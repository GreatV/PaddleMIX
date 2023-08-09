import paddle
import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import PIL
from packaging import version
from ...configuration_utils import FrozenDict
from ...image_processor import VaeImageProcessor
from ...loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from ...models import AutoencoderKL, UNet2DConditionModel
from ...schedulers import DDIMInverseScheduler, KarrasDiffusionSchedulers
from ...utils import PIL_INTERPOLATION, BaseOutput, deprecate, logging, randn_tensor, replace_example_docstring
from ..pipeline_utils import DiffusionPipeline
from . import StableDiffusionPipelineOutput
from .safety_checker import StableDiffusionSafetyChecker
logger = logging.get_logger(__name__)


@dataclass
class DiffEditInversionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        latents (`paddle.Tensor
            inverted latents tensor
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `num_timesteps * batch_size` or numpy array of shape `(num_timesteps,
            batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the
            diffusion pipeline.
    """
    latents: paddle.Tensor
    images: Union[List[PIL.Image.Image], np.ndarray]


EXAMPLE_DOC_STRING = """

        ```py
        >>> import PIL
        >>> import requests
        >>> import torch
        >>> from io import BytesIO

        >>> from diffusers import StableDiffusionDiffEditPipeline


        >>> def download_image(url):
        ...     response = requests.get(url)
        ...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")


        >>> img_url = "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"

        >>> init_image = download_image(img_url).resize((768, 768))

        >>> pipe = StableDiffusionDiffEditPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
        ... )

        >>> pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        >>> pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
        >>> pipeline.enable_model_cpu_offload()

        >>> mask_prompt = "A bowl of fruits"
        >>> prompt = "A bowl of pears"

        >>> mask_image = pipe.generate_mask(image=init_image, source_prompt=prompt, target_prompt=mask_prompt)
        >>> image_latents = pipe.invert(image=init_image, prompt=mask_prompt).latents
        >>> image = pipe(prompt=prompt, mask_image=mask_image, image_latents=image_latents).images[0]
        ```
"""
EXAMPLE_INVERT_DOC_STRING = """
        ```py
        >>> import PIL
        >>> import requests
        >>> import torch
        >>> from io import BytesIO

        >>> from diffusers import StableDiffusionDiffEditPipeline


        >>> def download_image(url):
        ...     response = requests.get(url)
        ...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")


        >>> img_url = "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"

        >>> init_image = download_image(img_url).resize((768, 768))

        >>> pipe = StableDiffusionDiffEditPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
        ... )

        >>> pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        >>> pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
        >>> pipeline.enable_model_cpu_offload()

        >>> prompt = "A bowl of fruits"

        >>> inverted_latents = pipe.invert(image=init_image, prompt=prompt).latents
        ```
"""


def auto_corr_loss(hidden_states, generator=None):
    reg_loss = 0.0
    for i in range(hidden_states.shape[0]):
        for j in range(hidden_states.shape[1]):
            noise = hidden_states[i:i + 1, j:j + 1, :, :]
            while True:
                roll_amount = paddle.randint(
                    low=0, high=noise.shape[2] // 2, shape=(1, )).item()
                reg_loss += (noise * paddle.roll(
                    x=noise, shifts=roll_amount, axis=2)).mean()**2
                reg_loss += (noise * paddle.roll(
                    x=noise, shifts=roll_amount, axis=3)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = paddle.nn.functional.avg_pool2d(
                    kernel_size=2, x=noise, exclusive=False)
    return reg_loss


def kl_divergence(hidden_states):
    return hidden_states.var() + hidden_states.mean()**2 - 1 - paddle.log(
        x=hidden_states.var() + 1e-07)


def preprocess(image):
    warnings.warn(
        'The preprocess method is deprecated and will be removed in a future version. Please use VaeImageProcessor.preprocess instead',
        FutureWarning)
    if isinstance(image, paddle.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]
    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = (x - x % 8 for x in (w, h))
        image = [
            np.array(i.resize(
                (w, h), resample=PIL_INTERPOLATION['lanczos']))[(None), :]
            for i in image
        ]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = paddle.to_tensor(data=image)
    elif isinstance(image[0], paddle.Tensor):
        image = paddle.concat(x=image, axis=0)
    return image


def preprocess_mask(mask, batch_size: int=1):
    if not isinstance(mask, paddle.Tensor):
        if isinstance(mask, PIL.Image.Image) or isinstance(mask, np.ndarray):
            mask = [mask]
        if isinstance(mask, list):
            if isinstance(mask[0], PIL.Image.Image):
                mask = [(np.array(m.convert('L')).astype(np.float32) / 255.0)
                        for m in mask]
            if isinstance(mask[0], np.ndarray):
                mask = np.stack(
                    mask, axis=0) if mask[0].ndim < 3 else np.concatenate(
                        mask, axis=0)
                mask = paddle.to_tensor(data=mask)
            elif isinstance(mask[0], paddle.Tensor):
                mask = paddle.stack(
                    x=mask, axis=0) if mask[0].ndim < 3 else paddle.concat(
                        x=mask, axis=0)
    if mask.ndim == 2:
        mask = mask.unsqueeze(axis=0).unsqueeze(axis=0)
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask.unsqueeze(axis=0)
        else:
            mask = mask.unsqueeze(axis=1)
    if batch_size > 1:
        if mask.shape[0] == 1:
            mask = paddle.concat(x=[mask] * batch_size)
        elif mask.shape[0] > 1 and mask.shape[0] != batch_size:
            raise ValueError(
                f'`mask_image` with batch size {mask.shape[0]} cannot be broadcasted to batch size {batch_size} inferred by prompt inputs'
            )
    if mask.shape[1] != 1:
        raise ValueError(
            f'`mask_image` must have 1 channel, but has {mask.shape[1]} channels'
        )
    if mask.min() < 0 or mask.max() > 1:
        raise ValueError('`mask_image` should be in [0, 1] range')
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    return mask


class StableDiffusionDiffEditPipeline(
        DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin):
    """
    <Tip warning={true}>

    This is an experimental feature!

    </Tip>

    Pipeline for text-guided image inpainting using Stable Diffusion and DiffEdit.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    In addition the pipeline inherits the following loading methods:
        - *Textual-Inversion*: [`loaders.TextualInversionLoaderMixin.load_textual_inversion`]
        - *LoRA*: [`loaders.LoraLoaderMixin.load_lora_weights`]

    as well as the following saving methods:
        - *LoRA*: [`loaders.LoraLoaderMixin.save_lora_weights`]

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer (`CLIPTokenizer`):
            A [`~transformers.CLIPTokenizer`] to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A [`UNet2DConditionModel`] to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        inverse_scheduler (`[DDIMInverseScheduler]`):
            A scheduler to be used in combination with `unet` to fill in the unmasked part of the input latents.
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`CLIPImageProcessor`]):
            A [`CLIPImageProcessor`] to extract features from generated images; used as inputs to the `safety_checker`.
    """
    _optional_components = [
        'safety_checker', 'feature_extractor', 'inverse_scheduler'
    ]

    def __init__(self,
                 vae: AutoencoderKL,
                 text_encoder: paddlenlp.transformers.CLIPTextModel,
                 tokenizer: paddlenlp.transformers.CLIPTokenizer,
                 unet: UNet2DConditionModel,
                 scheduler: KarrasDiffusionSchedulers,
                 safety_checker: StableDiffusionSafetyChecker,
                 feature_extractor: paddlenlp.transformers.CLIPImageProcessor,
                 inverse_scheduler: DDIMInverseScheduler,
                 requires_safety_checker: bool=True):
        super().__init__()
        if hasattr(scheduler.config,
                   'steps_offset') and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f'The configuration file of this scheduler: {scheduler} is outdated. `steps_offset` should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure to update the config accordingly as leaving `steps_offset` might led to incorrect results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json` file'
            )
            deprecate(
                'steps_offset!=1',
                '1.0.0',
                deprecation_message,
                standard_warn=False)
            new_config = dict(scheduler.config)
            new_config['steps_offset'] = 1
            scheduler._internal_dict = FrozenDict(new_config)
        if hasattr(
                scheduler.config,
                'skip_prk_steps') and scheduler.config.skip_prk_steps is False:
            deprecation_message = (
                f'The configuration file of this scheduler: {scheduler} has not set the configuration `skip_prk_steps`. `skip_prk_steps` should be set to True in the configuration file. Please make sure to update the config accordingly as not setting `skip_prk_steps` in the config might lead to incorrect results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json` file'
            )
            deprecate(
                'skip_prk_steps not set',
                '1.0.0',
                deprecation_message,
                standard_warn=False)
            new_config = dict(scheduler.config)
            new_config['skip_prk_steps'] = True
            scheduler._internal_dict = FrozenDict(new_config)
        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f'You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .'
            )
        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )
        is_unet_version_less_0_9_0 = hasattr(
            unet.config, '_diffusers_version') and version.parse(
                version.parse(unet.config._diffusers_version)
                .base_version) < version.parse('0.9.0.dev0')
        is_unet_sample_size_less_64 = hasattr(
            unet.config, 'sample_size') and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = """The configuration file of the unet has set the default `sample_size` to smaller than 64 which seems highly unlikely .If you're checkpoint is a fine-tuned version of any of the following: 
- CompVis/stable-diffusion-v1-4 
- CompVis/stable-diffusion-v1-3 
- CompVis/stable-diffusion-v1-2 
- CompVis/stable-diffusion-v1-1 
- runwayml/stable-diffusion-v1-5 
- runwayml/stable-diffusion-inpainting 
 you should change 'sample_size' to 64 in the configuration file. Please make sure to update the config accordingly as leaving `sample_size=32` in the config might lead to incorrect results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for the `unet/config.json` file"""
            deprecate(
                'sample_size<64',
                '1.0.0',
                deprecation_message,
                standard_warn=False)
            new_config = dict(unet.config)
            new_config['sample_size'] = 64
            unet._internal_dict = FrozenDict(new_config)
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            inverse_scheduler=inverse_scheduler)
        self.vae_scale_factor = 2**(len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def _encode_prompt(self,
                       prompt,
                       num_images_per_prompt,
                       do_classifier_free_guidance,
                       negative_prompt=None,
                       prompt_embeds: Optional[paddle.Tensor]=None,
                       negative_prompt_embeds: Optional[paddle.Tensor]=None,
                       lora_scale: Optional[float]=None):
        """
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`paddle.Tensoroptional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        if prompt_embeds is None:
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)
            text_inputs = self.tokenizer(
                prompt,
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors='pd')
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(
                prompt, padding='longest', return_tensors='pd').input_ids
            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                    -1] and not paddle.equal_all(
                        x=text_input_ids, y=untruncated_ids).item():
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1:-1])
                logger.warning(
                    f'The following part of your input was truncated because CLIP can only handle sequences up to {self.tokenizer.model_max_length} tokens: {removed_text}'
                )
            if hasattr(self.text_encoder.config, 'use_attention_mask'
                       ) and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask
            else:
                attention_mask = None
            prompt_embeds = self.text_encoder(
                text_input_ids, attention_mask=attention_mask)
            prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.cast(dtype=self.text_encoder.dtype)
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.tile(
            repeat_times=[1, num_images_per_prompt, 1])
        prompt_embeds = prompt_embeds.reshape(
            [bs_embed * num_images_per_prompt, seq_len, -1])
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [''] * batch_size
            elif prompt is not None and type(prompt) is not type(
                    negative_prompt):
                raise TypeError(
                    f'`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} != {type(prompt)}.'
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f'`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`: {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches the batch size of `prompt`.'
                )
            else:
                uncond_tokens = negative_prompt
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens,
                                                          self.tokenizer)
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding='max_length',
                max_length=max_length,
                truncation=True,
                return_tensors='pd')
            if hasattr(self.text_encoder.config, 'use_attention_mask'
                       ) and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask
            else:
                attention_mask = None
            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids, attention_mask=attention_mask)
            negative_prompt_embeds = negative_prompt_embeds[0]
        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.cast(
                dtype=self.text_encoder.dtype)
            negative_prompt_embeds = negative_prompt_embeds.tile(
                repeat_times=[1, num_images_per_prompt, 1])
            negative_prompt_embeds = negative_prompt_embeds.reshape(
                [batch_size * num_images_per_prompt, seq_len, -1])
            prompt_embeds = paddle.concat(
                x=[negative_prompt_embeds, prompt_embeds])
        return prompt_embeds

    def run_safety_checker(self, image, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if paddle.is_tensor(x=image):
                feature_extractor_input = self.image_processor.postprocess(
                    image, output_type='pil')
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(
                    image)
            safety_checker_input = self.feature_extractor(
                feature_extractor_input, return_tensors='pd')
            image, has_nsfw_concept = self.safety_checker(
                images=image,
                clip_input=safety_checker_input.pixel_values.cast(dtype))
        return image, has_nsfw_concept

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = 'eta' in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs['eta'] = eta
        accepts_generator = 'generator' in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs['generator'] = generator
        return extra_step_kwargs

    def decode_latents(self, latents):
        warnings.warn(
            'The decode_latents method is deprecated and will be removed in a future version. Please use VaeImageProcessor instead',
            FutureWarning)
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clip(min=0, max=1)
        image = image.cpu().transpose(perm=[0, 2, 3, 1]).astype(
            dtype='float32').numpy()
        return image

    def check_inputs(self,
                     prompt,
                     strength,
                     callback_steps,
                     negative_prompt=None,
                     prompt_embeds=None,
                     negative_prompt_embeds=None):
        if strength is None or strength is not None and (strength < 0 or
                                                         strength > 1):
            raise ValueError(
                f'The value of `strength` should in [0.0, 1.0] but is, but is {strength} of type {type(strength)}.'
            )
        if callback_steps is None or callback_steps is not None and (
                not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f'`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}.'
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f'Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to only forward one of the two.'
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                'Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined.'
            )
        elif prompt is not None and (not isinstance(prompt, str) and
                                     not isinstance(prompt, list)):
            raise ValueError(
                f'`prompt` has to be of type `str` or `list` but is {type(prompt)}'
            )
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f'Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to only forward one of the two.'
            )
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    f'`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds` {negative_prompt_embeds.shape}.'
                )

    def check_source_inputs(self,
                            source_prompt=None,
                            source_negative_prompt=None,
                            source_prompt_embeds=None,
                            source_negative_prompt_embeds=None):
        if source_prompt is not None and source_prompt_embeds is not None:
            raise ValueError(
                f'Cannot forward both `source_prompt`: {source_prompt} and `source_prompt_embeds`: {source_prompt_embeds}.  Please make sure to only forward one of the two.'
            )
        elif source_prompt is None and source_prompt_embeds is None:
            raise ValueError(
                'Provide either `source_image` or `source_prompt_embeds`. Cannot leave all both of the arguments undefined.'
            )
        elif source_prompt is not None and (
                not isinstance(source_prompt, str) and
                not isinstance(source_prompt, list)):
            raise ValueError(
                f'`source_prompt` has to be of type `str` or `list` but is {type(source_prompt)}'
            )
        if (source_negative_prompt is not None and
                source_negative_prompt_embeds is not None):
            raise ValueError(
                f'Cannot forward both `source_negative_prompt`: {source_negative_prompt} and `source_negative_prompt_embeds`: {source_negative_prompt_embeds}. Please make sure to only forward one of the two.'
            )
        if (source_prompt_embeds is not None and
                source_negative_prompt_embeds is not None):
            if (source_prompt_embeds.shape !=
                    source_negative_prompt_embeds.shape):
                raise ValueError(
                    f'`source_prompt_embeds` and `source_negative_prompt_embeds` must have the same shape when passed directly, but got: `source_prompt_embeds` {source_prompt_embeds.shape} != `source_negative_prompt_embeds` {source_negative_prompt_embeds.shape}.'
                )

    def get_timesteps(self, num_inference_steps, strength):
        init_timestep = min(
            int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order:]
        return timesteps, num_inference_steps - t_start

    def get_inverse_timesteps(self, num_inference_steps, strength):
        init_timestep = min(
            int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        if t_start == 0:
            return self.inverse_scheduler.timesteps, num_inference_steps
        timesteps = self.inverse_scheduler.timesteps[:-t_start]
        return timesteps, num_inference_steps - t_start

    def prepare_latents(self,
                        batch_size,
                        num_channels_latents,
                        height,
                        width,
                        dtype,
                        generator,
                        latents=None):
        shape = (batch_size, num_channels_latents, height //
                 self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, dtype=dtype)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_image_latents(self, image, batch_size, dtype, generator=None):
        if not isinstance(image, (paddle.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f'`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}'
            )
        image = image.cast(dtype=dtype)
        if image.shape[1] == 4:
            latents = image
        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
                )
            if isinstance(generator, list):
                latents = [
                    self.vae.encode(image[i:i + 1]).latent_dist.sample(
                        generator[i]) for i in range(batch_size)
                ]
                latents = paddle.concat(x=latents, axis=0)
            else:
                latents = self.vae.encode(image).latent_dist.sample(generator)
            latents = self.vae.config.scaling_factor * latents
        if batch_size != latents.shape[0]:
            if batch_size % latents.shape[0] == 0:
                deprecation_message = (
                    f'You have passed {batch_size} text prompts (`prompt`), but only {latents.shape[0]} initial images (`image`). Initial images are now duplicating to match the number of text prompts. Note that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update your script to pass as many initial images as text prompts to suppress this warning.'
                )
                deprecate(
                    'len(prompt) != len(image)',
                    '1.0.0',
                    deprecation_message,
                    standard_warn=False)
                additional_latents_per_image = batch_size // latents.shape[0]
                latents = paddle.concat(
                    x=[latents] * additional_latents_per_image, axis=0)
            else:
                raise ValueError(
                    f'Cannot duplicate `image` of batch size {latents.shape[0]} to {batch_size} text prompts.'
                )
        else:
            latents = paddle.concat(x=[latents], axis=0)
        return latents

    def get_epsilon(self,
                    model_output: paddle.Tensor,
                    sample: paddle.Tensor,
                    timestep: int):
        pred_type = self.inverse_scheduler.config.prediction_type
        alpha_prod_t = self.inverse_scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        if pred_type == 'epsilon':
            return model_output
        elif pred_type == 'sample':
            return (
                sample - alpha_prod_t**0.5 * model_output) / beta_prod_t**0.5
        elif pred_type == 'v_prediction':
            return (
                alpha_prod_t**0.5 * model_output + beta_prod_t**0.5 * sample)
        else:
            raise ValueError(
                f'prediction_type given as {pred_type} must be one of `epsilon`, `sample`, or `v_prediction`'
            )

    @paddle.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def generate_mask(
            self,
            image: Union[paddle.Tensor, PIL.Image.Image]=None,
            target_prompt: Optional[Union[str, List[str]]]=None,
            target_negative_prompt: Optional[Union[str, List[str]]]=None,
            target_prompt_embeds: Optional[paddle.Tensor]=None,
            target_negative_prompt_embeds: Optional[paddle.Tensor]=None,
            source_prompt: Optional[Union[str, List[str]]]=None,
            source_negative_prompt: Optional[Union[str, List[str]]]=None,
            source_prompt_embeds: Optional[paddle.Tensor]=None,
            source_negative_prompt_embeds: Optional[paddle.Tensor]=None,
            num_maps_per_mask: Optional[int]=10,
            mask_encode_strength: Optional[float]=0.5,
            mask_thresholding_ratio: Optional[float]=3.0,
            num_inference_steps: int=50,
            guidance_scale: float=7.5,
            generator: Optional[Union[torch.Generator, List[
                torch.Generator]]]=None,
            output_type: Optional[str]='np',
            cross_attention_kwargs: Optional[Dict[str, Any]]=None):
        """
        Generate a latent mask given a mask prompt, a target prompt, and an image.

        Args:
            image (`PIL.Image.Image`):
                `Image` or tensor representing an image batch to be used for computing the mask.
            target_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide semantic mask generation. If not defined, you need to pass
                `prompt_embeds`.
            target_negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            target_prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            target_negative_prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            source_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide semantic mask generation using DiffEdit. If not defined, you need to
                pass `source_prompt_embeds` or `source_image` instead.
            source_negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide semantic mask generation away from using DiffEdit. If not defined, you
                need to pass `source_negative_prompt_embeds` or `source_image` instead.
            source_prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated text embeddings to guide the semantic mask generation. Can be used to easily tweak text
                inputs (prompt weighting). If not provided, text embeddings are generated from `source_prompt` input
                argument.
            source_negative_prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated text embeddings to negatively guide the semantic mask generation. Can be used to easily
                tweak text inputs (prompt weighting). If not provided, text embeddings are generated from
                `source_negative_prompt` input argument.
            num_maps_per_mask (`int`, *optional*, defaults to 10):
                The number of noise maps sampled to generate the semantic mask using DiffEdit.
            mask_encode_strength (`float`, *optional*, defaults to 0.5):
                The strength of the noise maps sampled to generate the semantic mask using DiffEdit. Must be between 0
                and 1.
            mask_thresholding_ratio (`float`, *optional*, defaults to 3.0):
                The maximum multiple of the mean absolute difference used to clamp the semantic guidance map before
                mask binarization.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Examples:

        Returns:
            `List[PIL.Image.Image]` or `np.array`:
                When returning a `List[PIL.Image.Image]`, the list consists of a batch of single-channel binary images
                with dimensions `(height // self.vae_scale_factor, width // self.vae_scale_factor)`. If it's
                `np.array`, the shape is `(batch_size, height // self.vae_scale_factor, width //
                self.vae_scale_factor)`.
        """
        self.check_inputs(target_prompt, mask_encode_strength, 1,
                          target_negative_prompt, target_prompt_embeds,
                          target_negative_prompt_embeds)
        self.check_source_inputs(source_prompt, source_negative_prompt,
                                 source_prompt_embeds,
                                 source_negative_prompt_embeds)
        if num_maps_per_mask is None or num_maps_per_mask is not None and (
                not isinstance(num_maps_per_mask, int) or
                num_maps_per_mask <= 0):
            raise ValueError(
                f'`num_maps_per_mask` has to be a positive integer but is {num_maps_per_mask} of type {type(num_maps_per_mask)}.'
            )
        if mask_thresholding_ratio is None or mask_thresholding_ratio <= 0:
            raise ValueError(
                f'`mask_thresholding_ratio` has to be positive but is {mask_thresholding_ratio} of type {type(mask_thresholding_ratio)}.'
            )
        if target_prompt is not None and isinstance(target_prompt, str):
            batch_size = 1
        elif target_prompt is not None and isinstance(target_prompt, list):
            batch_size = len(target_prompt)
        else:
            batch_size = target_prompt_embeds.shape[0]
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
        do_classifier_free_guidance = guidance_scale > 1.0
        cross_attention_kwargs.get(
            'scale', None) if cross_attention_kwargs is not None else None
        target_prompt_embeds = self._encode_prompt(
            target_prompt,
            num_maps_per_mask,
            do_classifier_free_guidance,
            target_negative_prompt,
            prompt_embeds=target_prompt_embeds,
            negative_prompt_embeds=target_negative_prompt_embeds)
        source_prompt_embeds = self._encode_prompt(
            source_prompt,
            num_maps_per_mask,
            do_classifier_free_guidance,
            source_negative_prompt,
            prompt_embeds=source_prompt_embeds,
            negative_prompt_embeds=source_negative_prompt_embeds)
        image = self.image_processor.preprocess(image).repeat_interleave(
            repeats=num_maps_per_mask, axis=0)
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps, _ = self.get_timesteps(num_inference_steps,
                                          mask_encode_strength)
        encode_timestep = timesteps[0]
        image_latents = self.prepare_image_latents(
            image, batch_size * num_maps_per_mask, self.vae.dtype, generator)
        noise = randn_tensor(
            image_latents.shape, generator=generator, dtype=self.vae.dtype)
        image_latents = self.scheduler.add_noise(image_latents, noise,
                                                 encode_timestep)
        latent_model_input = paddle.concat(x=[image_latents] * (
            4 if do_classifier_free_guidance else 2))
        latent_model_input = self.scheduler.scale_model_input(
            latent_model_input, encode_timestep)
        prompt_embeds = paddle.concat(
            x=[source_prompt_embeds, target_prompt_embeds])
        noise_pred = self.unet(
            latent_model_input,
            encode_timestep,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs).sample
        if do_classifier_free_guidance:
            (noise_pred_neg_src, noise_pred_source, noise_pred_uncond,
             noise_pred_target) = noise_pred.chunk(chunks=4)
            noise_pred_source = noise_pred_neg_src + guidance_scale * (
                noise_pred_source - noise_pred_neg_src)
            noise_pred_target = noise_pred_uncond + guidance_scale * (
                noise_pred_target - noise_pred_uncond)
        else:
            noise_pred_source, noise_pred_target = noise_pred.chunk(chunks=2)
        mask_guidance_map = paddle.abs(
            x=noise_pred_target - noise_pred_source).reshape(
                batch_size, num_maps_per_mask,
                *noise_pred_target.shape[-3:]).mean(axis=[1, 2])
        clamp_magnitude = mask_guidance_map.mean() * mask_thresholding_ratio
        semantic_mask_image = mask_guidance_map.clip(
            min=0, max=clamp_magnitude) / clamp_magnitude
        semantic_mask_image = paddle.where(
            condition=semantic_mask_image <= 0.5, x=0, y=1)
        mask_image = semantic_mask_image.cpu().numpy()
        if output_type == 'pil':
            mask_image = self.image_processor.numpy_to_pil(mask_image)
        if hasattr(
                self,
                'final_offload_hook') and self.final_offload_hook is not None:
            self.final_offload_hook.offload()
        return mask_image

    @paddle.no_grad()
    @replace_example_docstring(EXAMPLE_INVERT_DOC_STRING)
    def invert(
            self,
            prompt: Optional[Union[str, List[str]]]=None,
            image: Union[paddle.Tensor, PIL.Image.Image]=None,
            num_inference_steps: int=50,
            inpaint_strength: float=0.8,
            guidance_scale: float=7.5,
            negative_prompt: Optional[Union[str, List[str]]]=None,
            generator: Optional[Union[torch.Generator, List[
                torch.Generator]]]=None,
            prompt_embeds: Optional[paddle.Tensor]=None,
            negative_prompt_embeds: Optional[paddle.Tensor]=None,
            decode_latents: bool=False,
            output_type: Optional[str]='pil',
            return_dict: bool=True,
            callback: Optional[Callable[[int, int, paddle.Tensor], None]]=None,
            callback_steps: Optional[int]=1,
            cross_attention_kwargs: Optional[Dict[str, Any]]=None,
            lambda_auto_corr: float=20.0,
            lambda_kl: float=20.0,
            num_reg_steps: int=0,
            num_auto_corr_rolls: int=5):
        """
        Generate inverted latents given a prompt and image.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`PIL.Image.Image`):
                `Image` or tensor representing an image batch to produce the inverted latents guided by `prompt`.
            inpaint_strength (`float`, *optional*, defaults to 0.8):
                Indicates extent of the noising process to run latent inversion. Must be between 0 and 1. When
                `strength` is 1, the inversion process iss ru for the full number of iterations specified in
                `num_inference_steps`. `image` is used as a reference for the inversion process, adding more noise the
                larger the `strength`. If `strength` is 0, no inpainting occurs.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            decode_latents (`bool`, *optional*, defaults to `False`):
                Whether or not to decode the inverted latents into a generated image. Setting this argument to `True`
                decodes all inverted latents for each timestep into a list of generated images.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.DiffEditInversionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: paddle.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            lambda_auto_corr (`float`, *optional*, defaults to 20.0):
                Lambda parameter to control auto correction.
            lambda_kl (`float`, *optional*, defaults to 20.0):
                Lambda parameter to control Kullback–Leibler divergence output.
            num_reg_steps (`int`, *optional*, defaults to 0):
                Number of regularization loss steps.
            num_auto_corr_rolls (`int`, *optional*, defaults to 5):
                Number of auto correction roll steps.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.pipeline_stable_diffusion_diffedit.DiffEditInversionPipelineOutput`] or
            `tuple`:
                If `return_dict` is `True`,
                [`~pipelines.stable_diffusion.pipeline_stable_diffusion_diffedit.DiffEditInversionPipelineOutput`] is
                returned, otherwise a `tuple` is returned where the first element is the inverted latents tensors
                ordered by increasing noise, and the second is the corresponding decoded images if `decode_latents` is
                `True`, otherwise `None`.
        """
        self.check_inputs(prompt, inpaint_strength, callback_steps,
                          negative_prompt, prompt_embeds,
                          negative_prompt_embeds)
        if image is None:
            raise ValueError('`image` input cannot be undefined.')
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
        do_classifier_free_guidance = guidance_scale > 1.0
        image = self.image_processor.preprocess(image)
        num_images_per_prompt = 1
        latents = self.prepare_image_latents(image,
                                             batch_size * num_images_per_prompt,
                                             self.vae.dtype, generator)
        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds)
        self.inverse_scheduler.set_timesteps(num_inference_steps)
        timesteps, num_inference_steps = self.get_inverse_timesteps(
            num_inference_steps, inpaint_strength)
        num_warmup_steps = len(
            timesteps) - num_inference_steps * self.inverse_scheduler.order
        inverted_latents = []
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = paddle.concat(
                    x=[latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.inverse_scheduler.scale_model_input(
                    latent_model_input, t)
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs).sample
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(
                        chunks=2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond)
                if num_reg_steps > 0:
                    with paddle.enable_grad():
                        for _ in range(num_reg_steps):
                            if lambda_auto_corr > 0:
                                for _ in range(num_auto_corr_rolls):
                                    var = noise_pred.detach().clone()
                                    var.stop_gradient = False
                                    var_epsilon = self.get_epsilon(
                                        var, latent_model_input.detach(), t)
                                    l_ac = auto_corr_loss(
                                        var_epsilon, generator=generator)
                                    l_ac.backward()
                                    grad = var.grad.detach(
                                    ) / num_auto_corr_rolls
                                    noise_pred = (
                                        noise_pred - lambda_auto_corr * grad)
                            if lambda_kl > 0:
                                var = noise_pred.detach().clone()
                                var.stop_gradient = False
                                var_epsilon = self.get_epsilon(
                                    var, latent_model_input.detach(), t)
                                l_kld = kl_divergence(var_epsilon)
                                l_kld.backward()
                                grad = var.grad.detach()
                                noise_pred = noise_pred - lambda_kl * grad
                            noise_pred = noise_pred.detach()
                latents = self.inverse_scheduler.step(noise_pred, t,
                                                      latents).prev_sample
                inverted_latents.append(latents.detach().clone())
                if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (
                        i + 1) % self.inverse_scheduler.order == 0:
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        assert len(inverted_latents) == len(timesteps)
        latents = paddle.stack(x=list(reversed(inverted_latents)), axis=1)
        image = None
        if decode_latents:
            image = self.decode_latents(
                latents.flatten(
                    start_axis=0, stop_axis=1))
        if decode_latents and output_type == 'pil':
            image = self.image_processor.numpy_to_pil(image)
        if hasattr(
                self,
                'final_offload_hook') and self.final_offload_hook is not None:
            self.final_offload_hook.offload()
        if not return_dict:
            return latents, image
        return DiffEditInversionPipelineOutput(latents=latents, images=image)

    @paddle.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
            self,
            prompt: Optional[Union[str, List[str]]]=None,
            mask_image: Union[paddle.Tensor, PIL.Image.Image]=None,
            image_latents: Union[paddle.Tensor, PIL.Image.Image]=None,
            inpaint_strength: Optional[float]=0.8,
            num_inference_steps: int=50,
            guidance_scale: float=7.5,
            negative_prompt: Optional[Union[str, List[str]]]=None,
            num_images_per_prompt: Optional[int]=1,
            eta: float=0.0,
            generator: Optional[Union[torch.Generator, List[
                torch.Generator]]]=None,
            latents: Optional[paddle.Tensor]=None,
            prompt_embeds: Optional[paddle.Tensor]=None,
            negative_prompt_embeds: Optional[paddle.Tensor]=None,
            output_type: Optional[str]='pil',
            return_dict: bool=True,
            callback: Optional[Callable[[int, int, paddle.Tensor], None]]=None,
            callback_steps: int=1,
            cross_attention_kwargs: Optional[Dict[str, Any]]=None):
        """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            mask_image (`PIL.Image.Image`):
                `Image` or tensor representing an image batch to mask the generated image. White pixels in the mask are
                repainted, while black pixels are preserved. If `mask_image` is a PIL image, it is converted to a
                single channel (luminance) before use. If it's a tensor, it should contain one color channel (L)
                instead of 3, so the expected shape would be `(B, 1, H, W)`.
            image_latents (`PIL.Image.Image` or `paddle.Tensor`):
                Partially noised image latents from the inversion process to be used as inputs for image generation.
            inpaint_strength (`float`, *optional*, defaults to 0.8):
                Indicates extent to inpaint the masked area. Must be between 0 and 1. When `strength` is 1, the
                denoising process is run on the masked area for the full number of iterations specified in
                `num_inference_steps`. `image_latents` is used as a reference for the masked area, adding more noise to
                that region the larger the `strength`. If `strength` is 0, no inpainting occurs.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`paddle.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: paddle.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        self.check_inputs(prompt, inpaint_strength, callback_steps,
                          negative_prompt, prompt_embeds,
                          negative_prompt_embeds)
        if mask_image is None:
            raise ValueError(
                '`mask_image` input cannot be undefined. Use `generate_mask()` to compute `mask_image` from text prompts.'
            )
        if image_latents is None:
            raise ValueError(
                '`image_latents` input cannot be undefined. Use `invert()` to compute `image_latents` from input images.'
            )
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
        do_classifier_free_guidance = guidance_scale > 1.0
        text_encoder_lora_scale = cross_attention_kwargs.get(
            'scale', None) if cross_attention_kwargs is not None else None
        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale)
        mask_image = preprocess_mask(mask_image, batch_size)
        latent_height, latent_width = mask_image.shape[-2:]
        mask_image = paddle.concat(x=[mask_image] * num_images_per_prompt)
        mask_image = mask_image.cast(dtype=prompt_embeds.dtype)
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps,
                                                            inpaint_strength)
        if isinstance(image_latents, list) and any(
                isinstance(l, paddle.Tensor) and l.ndim == 5
                for l in image_latents):
            image_latents = paddle.concat(x=image_latents).detach()
        elif isinstance(image_latents,
                        paddle.Tensor) and image_latents.ndim == 5:
            image_latents = image_latents.detach()
        else:
            image_latents = self.image_processor.preprocess(
                image_latents).detach()
        latent_shape = (self.vae.config.latent_channels, latent_height,
                        latent_width)
        if image_latents.shape[-3:] != latent_shape:
            raise ValueError(
                f'Each latent image in `image_latents` must have shape {latent_shape}, but has shape {image_latents.shape[-3:]}'
            )
        if image_latents.ndim == 4:
            image_latents = image_latents.reshape(batch_size,
                                                  len(timesteps), *latent_shape)
        if image_latents.shape[:2] != (batch_size, len(timesteps)):
            raise ValueError(
                f'`image_latents` must have batch size {batch_size} with latent images from {len(timesteps)} timesteps, but has batch size {image_latents.shape[0]} with latent images from {image_latents.shape[1]} timesteps.'
            )
        x = image_latents
        perm_4 = list(range(x.ndim))
        perm_4[0] = 1
        perm_4[1] = 0
        image_latents = x.transpose(perm=perm_4).repeat_interleave(
            repeats=num_images_per_prompt, axis=1)
        image_latents = image_latents.cast(dtype=prompt_embeds.dtype)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        latents = image_latents[0].clone()
        num_warmup_steps = len(
            timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = paddle.concat(
                    x=[latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t)
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs).sample
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(
                        chunks=2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample
                latents = latents * mask_image + image_latents[i] * (1 -
                                                                     mask_image)
                if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (
                        i + 1) % self.scheduler.order == 0:
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        if not output_type == 'latent':
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(
                image, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None
        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [(not has_nsfw) for has_nsfw in has_nsfw_concept]
        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize)
        if not return_dict:
            return image, has_nsfw_concept
        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept)
