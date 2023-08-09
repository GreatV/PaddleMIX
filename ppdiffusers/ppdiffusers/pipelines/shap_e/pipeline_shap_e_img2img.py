import paddle
from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np
import PIL
from ...models import PriorTransformer
from ...schedulers import HeunDiscreteScheduler
from ...utils import BaseOutput, logging, randn_tensor, replace_example_docstring
from ..pipeline_utils import DiffusionPipeline
from .renderer import ShapERenderer
import paddlenlp

logger = logging.get_logger(__name__)
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from PIL import Image
        >>> import torch
        >>> from diffusers import DiffusionPipeline
        >>> from diffusers.utils import export_to_gif, load_image

        >>> repo = "openai/shap-e-img2img"
        >>> pipe = DiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float16)

        >>> guidance_scale = 3.0
        >>> image_url = "https://hf.co/datasets/diffusers/docs-images/resolve/main/shap-e/corgi.png"
        >>> image = load_image(image_url).convert("RGB")

        >>> images = pipe(
        ...     image,
        ...     guidance_scale=guidance_scale,
        ...     num_inference_steps=64,
        ...     frame_size=256,
        ... ).images

        >>> gif_path = export_to_gif(images[0], "corgi_3d.gif")
        ```
"""


@dataclass
class ShapEPipelineOutput(BaseOutput):
    """
    Output class for [`ShapEPipeline`] and [`ShapEImg2ImgPipeline`].

    Args:
        images (`paddle.Tensor
            A list of images for 3D rendering.
    """
    images: Union[PIL.Image.Image, np.ndarray]


class ShapEImg2ImgPipeline(DiffusionPipeline):
    """
    Pipeline for generating latent representation of a 3D asset and rendering with NeRF method with Shap-E from an
    image.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        prior ([`PriorTransformer`]):
            The canonincal unCLIP prior to approximate the image embedding from the text embedding.
        image_encoder ([`CLIPVisionModel`]):
            Frozen image-encoder.
        image_processor (`CLIPImageProcessor`):
             A [`~transformers.CLIPImageProcessor`] to process images.
        scheduler ([`HeunDiscreteScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.
        shap_e_renderer ([`ShapERenderer`]):
            Shap-E renderer projects the generated latents into parameters of a MLP that's used to create 3D objects
            with the NeRF rendering method.
    """

    def __init__(self,
                 prior: PriorTransformer,
                 image_encoder: paddlenlp.transformers.CLIPVisionModel,
                 image_processor: paddlenlp.transformers.CLIPImageProcessor,
                 scheduler: HeunDiscreteScheduler,
                 shap_e_renderer: ShapERenderer):
        super().__init__()
        self.register_modules(
            prior=prior,
            image_encoder=image_encoder,
            image_processor=image_processor,
            scheduler=scheduler,
            shap_e_renderer=shap_e_renderer)

    def prepare_latents(self, shape, dtype, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(
                    f'Unexpected latents shape, got {latents.shape}, expected {shape}'
                )
        latents = latents * scheduler.init_noise_sigma
        return latents

    def _encode_image(self, image, num_images_per_prompt,
                      do_classifier_free_guidance):
        if isinstance(image, List) and isinstance(image[0], paddle.Tensor):
            image = paddle.concat(
                x=image, axis=0) if image[0].ndim == 4 else paddle.stack(
                    x=image, axis=0)
        if not isinstance(image, paddle.Tensor):
            image = self.image_processor(
                image, return_tensors='pd').pixel_values[0].unsqueeze(axis=0)
        image = image.cast(dtype=self.image_encoder.dtype)
        image_embeds = self.image_encoder(image)['last_hidden_state']
        image_embeds = image_embeds[:, 1:, :]
        image_embeds = image_embeds.repeat_interleave(
            repeats=num_images_per_prompt, axis=0)
        if do_classifier_free_guidance:
            negative_image_embeds = paddle.zeros_like(x=image_embeds)
            image_embeds = paddle.concat(
                x=[negative_image_embeds, image_embeds])
        return image_embeds

    @paddle.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(self,
                 image: Union[PIL.Image.Image, List[PIL.Image.Image]],
                 num_images_per_prompt: int=1,
                 num_inference_steps: int=25,
                 generator: Optional[Union[torch.Generator, List[
                     torch.Generator]]]=None,
                 latents: Optional[paddle.Tensor]=None,
                 guidance_scale: float=4.0,
                 frame_size: int=64,
                 output_type: Optional[str]='pil',
                 return_dict: bool=True):
        """
        The call function to the pipeline for generation.

        Args:
            image (`paddle.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[paddle.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image` or tensor representing an image batch to be used as the starting point. Can also accept image
                latents as `image`, if passing latents directly, it will not be encoded again.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`paddle.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            frame_size (`int`, *optional*, default to 64):
                The width and height of each image frame of the generated 3D output.
            output_type (`str`, *optional*, defaults to `"pt"`):
                (`np.array`),`"latent"` (`torch.Tensor`), mesh ([`MeshDecoderOutput`]).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.shap_e.pipeline_shap_e.ShapEPipelineOutput`] instead of a plain
                tuple.

        Examples:

        Returns:
            [`~pipelines.shap_e.pipeline_shap_e.ShapEPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.shap_e.pipeline_shap_e.ShapEPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images.
        """
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, paddle.Tensor):
            batch_size = image.shape[0]
        elif isinstance(image, list) and isinstance(image[0], (
                paddle.Tensor, PIL.Image.Image)):
            batch_size = len(image)
        else:
            raise ValueError(
                f'`image` has to be of type `PIL.Image.Image`, `torch.Tensor`, `List[PIL.Image.Image]` or `List[torch.Tensor]` but is {type(image)}'
            )
        batch_size = batch_size * num_images_per_prompt
        do_classifier_free_guidance = guidance_scale > 1.0
        image_embeds = self._encode_image(image, num_images_per_prompt,
                                          do_classifier_free_guidance)
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        num_embeddings = self.prior.config.num_embeddings
        embedding_dim = self.prior.config.embedding_dim
        latents = self.prepare_latents(
            (batch_size, num_embeddings * embedding_dim), image_embeds.dtype,
            generator, latents, self.scheduler)
        latents = latents.reshape(latents.shape[0], num_embeddings,
                                  embedding_dim)
        for i, t in enumerate(self.progress_bar(timesteps)):
            latent_model_input = paddle.concat(
                x=[latents] * 2) if do_classifier_free_guidance else latents
            scaled_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            noise_pred = self.prior(
                scaled_model_input, timestep=t,
                proj_embedding=image_embeds).predicted_image_embedding
            noise_pred, _ = noise_pred.split(scaled_model_input.shape[2], dim=2)
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred = noise_pred.chunk(chunks=2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred - noise_pred_uncond)
            latents = self.scheduler.step(
                noise_pred, timestep=t, sample=latents).prev_sample
        if output_type not in ['np', 'pil', 'latent', 'mesh']:
            raise ValueError(
                f'Only the output types `pil`, `np`, `latent` and `mesh` are supported not output_type={output_type}'
            )
        if output_type == 'latent':
            return ShapEPipelineOutput(images=latents)
        images = []
        if output_type == 'mesh':
            for i, latent in enumerate(latents):
                mesh = self.shap_e_renderer.decode_to_mesh(latent[(None), :], )
                images.append(mesh)
        else:
            for i, latent in enumerate(latents):
                image = self.shap_e_renderer.decode_to_image(
                    latent[(None), :], size=frame_size)
                images.append(image)
            images = paddle.stack(x=images)
            images = images.cpu().numpy()
            if output_type == 'pil':
                images = [self.numpy_to_pil(image) for image in images]
        if hasattr(
                self,
                'final_offload_hook') and self.final_offload_hook is not None:
            self.final_offload_hook.offload()
        if not return_dict:
            return images,
        return ShapEPipelineOutput(images=images)
