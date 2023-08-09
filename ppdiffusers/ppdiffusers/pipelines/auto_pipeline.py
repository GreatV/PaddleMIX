import inspect
from collections import OrderedDict
from ..configuration_utils import ConfigMixin
from .controlnet import StableDiffusionControlNetImg2ImgPipeline, StableDiffusionControlNetInpaintPipeline, StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline
from .deepfloyd_if import IFImg2ImgPipeline, IFInpaintingPipeline, IFPipeline
from .kandinsky import KandinskyCombinedPipeline, KandinskyImg2ImgCombinedPipeline, KandinskyImg2ImgPipeline, KandinskyInpaintCombinedPipeline, KandinskyInpaintPipeline, KandinskyPipeline
from .kandinsky2_2 import KandinskyV22CombinedPipeline, KandinskyV22Img2ImgCombinedPipeline, KandinskyV22Img2ImgPipeline, KandinskyV22InpaintCombinedPipeline, KandinskyV22InpaintPipeline, KandinskyV22Pipeline
from .stable_diffusion import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, StableDiffusionPipeline
from .stable_diffusion_xl import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLPipeline
AUTO_TEXT2IMAGE_PIPELINES_MAPPING = OrderedDict(
    [('stable-diffusion', StableDiffusionPipeline), ('stable-diffusion-xl',
                                                     StableDiffusionXLPipeline),
     ('if', IFPipeline), ('kandinsky', KandinskyCombinedPipeline), (
         'kandinsky22', KandinskyV22CombinedPipeline), (
             'stable-diffusion-controlnet', StableDiffusionControlNetPipeline),
     ('stable-diffusion-xl-controlnet', StableDiffusionXLControlNetPipeline)])
AUTO_IMAGE2IMAGE_PIPELINES_MAPPING = OrderedDict(
    [('stable-diffusion', StableDiffusionImg2ImgPipeline), (
        'stable-diffusion-xl', StableDiffusionXLImg2ImgPipeline),
     ('if', IFImg2ImgPipeline), ('kandinsky', KandinskyImg2ImgCombinedPipeline),
     ('kandinsky22', KandinskyV22Img2ImgCombinedPipeline),
     ('stable-diffusion-controlnet', StableDiffusionControlNetImg2ImgPipeline)])
AUTO_INPAINT_PIPELINES_MAPPING = OrderedDict(
    [('stable-diffusion', StableDiffusionInpaintPipeline),
     ('stable-diffusion-xl',
      StableDiffusionXLInpaintPipeline), ('if', IFInpaintingPipeline), (
          'kandinsky', KandinskyInpaintCombinedPipeline), (
              'kandinsky22', KandinskyV22InpaintCombinedPipeline),
     ('stable-diffusion-controlnet', StableDiffusionControlNetInpaintPipeline)])
_AUTO_TEXT2IMAGE_DECODER_PIPELINES_MAPPING = OrderedDict([(
    'kandinsky', KandinskyPipeline), ('kandinsky22', KandinskyV22Pipeline)])
_AUTO_IMAGE2IMAGE_DECODER_PIPELINES_MAPPING = OrderedDict(
    [('kandinsky', KandinskyImg2ImgPipeline),
     ('kandinsky22', KandinskyV22Img2ImgPipeline)])
_AUTO_INPAINT_DECODER_PIPELINES_MAPPING = OrderedDict(
    [('kandinsky', KandinskyInpaintPipeline),
     ('kandinsky22', KandinskyV22InpaintPipeline)])
SUPPORTED_TASKS_MAPPINGS = [
    AUTO_TEXT2IMAGE_PIPELINES_MAPPING, AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
    AUTO_INPAINT_PIPELINES_MAPPING, _AUTO_TEXT2IMAGE_DECODER_PIPELINES_MAPPING,
    _AUTO_IMAGE2IMAGE_DECODER_PIPELINES_MAPPING,
    _AUTO_INPAINT_DECODER_PIPELINES_MAPPING
]


def _get_connected_pipeline(pipeline_cls):
    if pipeline_cls in _AUTO_TEXT2IMAGE_DECODER_PIPELINES_MAPPING.values():
        return _get_task_class(
            AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
            pipeline_cls.__name__,
            throw_error_if_not_exist=False)
    if pipeline_cls in _AUTO_IMAGE2IMAGE_DECODER_PIPELINES_MAPPING.values():
        return _get_task_class(
            AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
            pipeline_cls.__name__,
            throw_error_if_not_exist=False)
    if pipeline_cls in _AUTO_INPAINT_DECODER_PIPELINES_MAPPING.values():
        return _get_task_class(
            AUTO_INPAINT_PIPELINES_MAPPING,
            pipeline_cls.__name__,
            throw_error_if_not_exist=False)


def _get_task_class(mapping,
                    pipeline_class_name,
                    throw_error_if_not_exist: bool=True):
    def get_model(pipeline_class_name):
        for task_mapping in SUPPORTED_TASKS_MAPPINGS:
            for model_name, pipeline in task_mapping.items():
                if pipeline.__name__ == pipeline_class_name:
                    return model_name

    model_name = get_model(pipeline_class_name)
    if model_name is not None:
        task_class = mapping.get(model_name, None)
        if task_class is not None:
            return task_class
    if throw_error_if_not_exist:
        raise ValueError(
            f"AutoPipeline can't find a pipeline linked to {pipeline_class_name} for {model_name}"
        )


def _get_signature_keys(obj):
    parameters = inspect.signature(obj.__init__).parameters
    required_parameters = {
        k: v
        for k, v in parameters.items() if v.default == inspect._empty
    }
    optional_parameters = set(
        {k
         for k, v in parameters.items() if v.default != inspect._empty})
    expected_modules = set(required_parameters.keys()) - {'self'}
    return expected_modules, optional_parameters


class AutoPipelineForText2Image(ConfigMixin):
    """

    AutoPipeline for text-to-image generation.

    [`AutoPipelineForText2Image`] is a generic pipeline class that will be instantiated as one of the text-to-image
    pipeline class in diffusers.

    The pipeline type (for example [`StableDiffusionPipeline`]) is automatically selected when created with the
    AutoPipelineForText2Image.from_pretrained(pretrained_model_name_or_path) or
    AutoPipelineForText2Image.from_pipe(pipeline) class methods .

    This class cannot be instantiated using __init__() (throws an error).

    Class attributes:

        - **config_name** (`str`) -- The configuration filename that stores the class and module names of all the
          diffusion pipeline's components.

    """
    config_name = 'model_index.json'

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f'{self.__class__.__name__} is designed to be instantiated using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or `{self.__class__.__name__}.from_pipe(pipeline)` methods.'
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, **kwargs):
        """
        Instantiates a text-to-image Pytorch diffusion pipeline from pretrained pipeline weight.

        The from_pretrained() method takes care of returning the correct pipeline class instance by:
            1. Detect the pipeline class of the pretrained_model_or_path based on the _class_name property of its
               config object
            2. Find the text-to-image pipeline linked to the pipeline class using pattern matching on pipeline class
               name.

        If a `controlnet` argument is passed, it will instantiate a [`StableDiffusionControlNetPipeline`] object.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If "auto" is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            custom_revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
                custom pipeline from GitHub, otherwise it defaults to `"main"` when loading from the Hub.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn’t need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Examples:

        ```py
        >>> from diffusers import AutoPipelineForTextToImage

        >>> pipeline = AutoPipelineForTextToImage.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> print(pipeline.__class__)
        ```
        """
        config = cls.load_config(pretrained_model_or_path)
        orig_class_name = config['_class_name']
        if 'controlnet' in kwargs:
            orig_class_name = config['_class_name'].replace(
                'Pipeline', 'ControlNetPipeline')
        text_2_image_cls = _get_task_class(AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
                                           orig_class_name)
        return text_2_image_cls.from_pretrained(pretrained_model_or_path,
                                                **kwargs)

    @classmethod
    def from_pipe(cls, pipeline, **kwargs):
        """
        Instantiates a text-to-image Pytorch diffusion pipeline from another instantiated diffusion pipeline class.

        The from_pipe() method takes care of returning the correct pipeline class instance by finding the text-to-image
        pipeline linked to the pipeline class using pattern matching on pipeline class name.

        All the modules the pipeline contains will be used to initialize the new pipeline without reallocating
        additional memoery.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        Parameters:
            pipeline (`DiffusionPipeline`):
                an instantiated `DiffusionPipeline` object

        ```py
        >>> from diffusers import AutoPipelineForTextToImage, AutoPipelineForImageToImage

        >>> pipe_i2i = AutoPipelineForImage2Image.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", requires_safety_checker=False
        ... )

        >>> pipe_t2i = AutoPipelineForTextToImage.from_pipe(pipe_t2i)
        ```
        """
        original_config = dict(pipeline.config)
        original_cls_name = pipeline.__class__.__name__
        text_2_image_cls = _get_task_class(AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
                                           original_cls_name)
        expected_modules, optional_kwargs = _get_signature_keys(
            text_2_image_cls)
        pretrained_model_name_or_path = original_config.pop('_name_or_path',
                                                            None)
        passed_class_obj = {
            k: kwargs.pop(k)
            for k in expected_modules if k in kwargs
        }
        original_class_obj = {
            k: pipeline.components[k]
            for k, v in pipeline.components.items()
            if k in expected_modules and k not in passed_class_obj
        }
        passed_pipe_kwargs = {
            k: kwargs.pop(k)
            for k in optional_kwargs if k in kwargs
        }
        original_pipe_kwargs = {
            k: original_config[k]
            for k, v in original_config.items()
            if k in optional_kwargs and k not in passed_pipe_kwargs
        }
        additional_pipe_kwargs = [
            k[1:] for k in original_config.keys()
            if k.startswith('_') and k[1:] in optional_kwargs and k[1:] not in
            passed_pipe_kwargs
        ]
        for k in additional_pipe_kwargs:
            original_pipe_kwargs[k] = original_config.pop(f'_{k}')
        text_2_image_kwargs = {
            ** passed_class_obj, ** original_class_obj, ** passed_pipe_kwargs,
            ** original_pipe_kwargs
        }
        unused_original_config = {
            f"{'' if k.startswith('_') else '_'}{k}": original_config[k]
            for k, v in original_config.items() if k not in text_2_image_kwargs
        }
        missing_modules = set(expected_modules) - set(
            pipeline._optional_components) - set(text_2_image_kwargs.keys())
        if len(missing_modules) > 0:
            raise ValueError(
                f'Pipeline {text_2_image_cls} expected {expected_modules}, but only {set(list(passed_class_obj.keys()) + list(original_class_obj.keys()))} were passed'
            )
        model = text_2_image_cls(**text_2_image_kwargs)
        model.register_to_config(_name_or_path=pretrained_model_name_or_path)
        model.register_to_config(**unused_original_config)
        return model


class AutoPipelineForImage2Image(ConfigMixin):
    """

    AutoPipeline for image-to-image generation.

    [`AutoPipelineForImage2Image`] is a generic pipeline class that will be instantiated as one of the image-to-image
    pipeline classes in diffusers.

    The pipeline type (for example [`StableDiffusionImg2ImgPipeline`]) is automatically selected when created with the
    `AutoPipelineForImage2Image.from_pretrained(pretrained_model_name_or_path)` or
    `AutoPipelineForImage2Image.from_pipe(pipeline)` class methods.

    This class cannot be instantiated using __init__() (throws an error).

    Class attributes:

        - **config_name** (`str`) -- The configuration filename that stores the class and module names of all the
          diffusion pipeline's components.

    """
    config_name = 'model_index.json'

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f'{self.__class__.__name__} is designed to be instantiated using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or `{self.__class__.__name__}.from_pipe(pipeline)` methods.'
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, **kwargs):
        """
        Instantiates a image-to-image Pytorch diffusion pipeline from pretrained pipeline weight.

        The from_pretrained() method takes care of returning the correct pipeline class instance by:
            1. Detect the pipeline class of the pretrained_model_or_path based on the _class_name property of its
               config object
            2. Find the image-to-image pipeline linked to the pipeline class using pattern matching on pipeline class
               name.

        If a `controlnet` argument is passed, it will instantiate a StableDiffusionControlNetImg2ImgPipeline object.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If "auto" is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            custom_revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
                custom pipeline from GitHub, otherwise it defaults to `"main"` when loading from the Hub.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn’t need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Examples:

        ```py
        >>> from diffusers import AutoPipelineForTextToImage

        >>> pipeline = AutoPipelineForImageToImage.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> print(pipeline.__class__)
        ```
        """
        config = cls.load_config(pretrained_model_or_path)
        orig_class_name = config['_class_name']
        if 'controlnet' in kwargs:
            orig_class_name = config['_class_name'].replace(
                'Pipeline', 'ControlNetPipeline')
        image_2_image_cls = _get_task_class(AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
                                            orig_class_name)
        return image_2_image_cls.from_pretrained(pretrained_model_or_path,
                                                 **kwargs)

    @classmethod
    def from_pipe(cls, pipeline, **kwargs):
        """
        Instantiates a image-to-image Pytorch diffusion pipeline from another instantiated diffusion pipeline class.

        The from_pipe() method takes care of returning the correct pipeline class instance by finding the
        image-to-image pipeline linked to the pipeline class using pattern matching on pipeline class name.

        All the modules the pipeline contains will be used to initialize the new pipeline without reallocating
        additional memoery.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        Parameters:
            pipeline (`DiffusionPipeline`):
                an instantiated `DiffusionPipeline` object

        Examples:

        ```py
        >>> from diffusers import AutoPipelineForTextToImage, AutoPipelineForImageToImage

        >>> pipe_t2i = AutoPipelineForText2Image.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", requires_safety_checker=False
        ... )

        >>> pipe_i2i = AutoPipelineForImageToImage.from_pipe(pipe_t2i)
        ```
        """
        original_config = dict(pipeline.config)
        original_cls_name = pipeline.__class__.__name__
        image_2_image_cls = _get_task_class(AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
                                            original_cls_name)
        expected_modules, optional_kwargs = _get_signature_keys(
            image_2_image_cls)
        pretrained_model_name_or_path = original_config.pop('_name_or_path',
                                                            None)
        passed_class_obj = {
            k: kwargs.pop(k)
            for k in expected_modules if k in kwargs
        }
        original_class_obj = {
            k: pipeline.components[k]
            for k, v in pipeline.components.items()
            if k in expected_modules and k not in passed_class_obj
        }
        passed_pipe_kwargs = {
            k: kwargs.pop(k)
            for k in optional_kwargs if k in kwargs
        }
        original_pipe_kwargs = {
            k: original_config[k]
            for k, v in original_config.items()
            if k in optional_kwargs and k not in passed_pipe_kwargs
        }
        additional_pipe_kwargs = [
            k[1:] for k in original_config.keys()
            if k.startswith('_') and k[1:] in optional_kwargs and k[1:] not in
            passed_pipe_kwargs
        ]
        for k in additional_pipe_kwargs:
            original_pipe_kwargs[k] = original_config.pop(f'_{k}')
        image_2_image_kwargs = {
            ** passed_class_obj, ** original_class_obj, ** passed_pipe_kwargs,
            ** original_pipe_kwargs
        }
        unused_original_config = {
            f"{'' if k.startswith('_') else '_'}{k}": original_config[k]
            for k, v in original_config.items() if k not in image_2_image_kwargs
        }
        missing_modules = set(expected_modules) - set(
            pipeline._optional_components) - set(image_2_image_kwargs.keys())
        if len(missing_modules) > 0:
            raise ValueError(
                f'Pipeline {image_2_image_cls} expected {expected_modules}, but only {set(list(passed_class_obj.keys()) + list(original_class_obj.keys()))} were passed'
            )
        model = image_2_image_cls(**image_2_image_kwargs)
        model.register_to_config(_name_or_path=pretrained_model_name_or_path)
        model.register_to_config(**unused_original_config)
        return model


class AutoPipelineForInpainting(ConfigMixin):
    """

    AutoPipeline for inpainting generation.

    [`AutoPipelineForInpainting`] is a generic pipeline class that will be instantiated as one of the inpainting
    pipeline class in diffusers.

    The pipeline type (for example [`IFInpaintingPipeline`]) is automatically selected when created with the
    AutoPipelineForInpainting.from_pretrained(pretrained_model_name_or_path) or
    AutoPipelineForInpainting.from_pipe(pipeline) class methods .

    This class cannot be instantiated using __init__() (throws an error).

    Class attributes:

        - **config_name** (`str`) -- The configuration filename that stores the class and module names of all the
          diffusion pipeline's components.

    """
    config_name = 'model_index.json'

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f'{self.__class__.__name__} is designed to be instantiated using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or `{self.__class__.__name__}.from_pipe(pipeline)` methods.'
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, **kwargs):
        """
        Instantiates a inpainting Pytorch diffusion pipeline from pretrained pipeline weight.

        The from_pretrained() method takes care of returning the correct pipeline class instance by:
            1. Detect the pipeline class of the pretrained_model_or_path based on the _class_name property of its
               config object
            2. Find the inpainting pipeline linked to the pipeline class using pattern matching on pipeline class name.

        If a `controlnet` argument is passed, it will instantiate a StableDiffusionControlNetInpaintPipeline object.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If "auto" is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            custom_revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
                custom pipeline from GitHub, otherwise it defaults to `"main"` when loading from the Hub.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn’t need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Examples:

        ```py
        >>> from diffusers import AutoPipelineForTextToImage

        >>> pipeline = AutoPipelineForImageToImage.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> print(pipeline.__class__)
        ```
        """
        config = cls.load_config(pretrained_model_or_path)
        orig_class_name = config['_class_name']
        if 'controlnet' in kwargs:
            orig_class_name = config['_class_name'].replace(
                'Pipeline', 'ControlNetPipeline')
        inpainting_cls = _get_task_class(AUTO_INPAINT_PIPELINES_MAPPING,
                                         orig_class_name)
        return inpainting_cls.from_pretrained(pretrained_model_or_path,
                                              **kwargs)

    @classmethod
    def from_pipe(cls, pipeline, **kwargs):
        """
        Instantiates a inpainting Pytorch diffusion pipeline from another instantiated diffusion pipeline class.

        The from_pipe() method takes care of returning the correct pipeline class instance by finding the inpainting
        pipeline linked to the pipeline class using pattern matching on pipeline class name.

        All the modules the pipeline class contain will be used to initialize the new pipeline without reallocating
        additional memoery.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        Parameters:
            pipeline (`DiffusionPipeline`):
                an instantiated `DiffusionPipeline` object

        Examples:

        ```py
        >>> from diffusers import AutoPipelineForTextToImage, AutoPipelineForInpainting

        >>> pipe_t2i = AutoPipelineForText2Image.from_pretrained(
        ...     "DeepFloyd/IF-I-XL-v1.0", requires_safety_checker=False
        ... )

        >>> pipe_inpaint = AutoPipelineForInpainting.from_pipe(pipe_t2i)
        ```
        """
        original_config = dict(pipeline.config)
        original_cls_name = pipeline.__class__.__name__
        inpainting_cls = _get_task_class(AUTO_INPAINT_PIPELINES_MAPPING,
                                         original_cls_name)
        expected_modules, optional_kwargs = _get_signature_keys(inpainting_cls)
        pretrained_model_name_or_path = original_config.pop('_name_or_path',
                                                            None)
        passed_class_obj = {
            k: kwargs.pop(k)
            for k in expected_modules if k in kwargs
        }
        original_class_obj = {
            k: pipeline.components[k]
            for k, v in pipeline.components.items()
            if k in expected_modules and k not in passed_class_obj
        }
        passed_pipe_kwargs = {
            k: kwargs.pop(k)
            for k in optional_kwargs if k in kwargs
        }
        original_pipe_kwargs = {
            k: original_config[k]
            for k, v in original_config.items()
            if k in optional_kwargs and k not in passed_pipe_kwargs
        }
        additional_pipe_kwargs = [
            k[1:] for k in original_config.keys()
            if k.startswith('_') and k[1:] in optional_kwargs and k[1:] not in
            passed_pipe_kwargs
        ]
        for k in additional_pipe_kwargs:
            original_pipe_kwargs[k] = original_config.pop(f'_{k}')
        inpainting_kwargs = {
            ** passed_class_obj, ** original_class_obj, ** passed_pipe_kwargs,
            ** original_pipe_kwargs
        }
        unused_original_config = {
            f"{'' if k.startswith('_') else '_'}{k}": original_config[k]
            for k, v in original_config.items() if k not in inpainting_kwargs
        }
        missing_modules = set(expected_modules) - set(
            pipeline._optional_components) - set(inpainting_kwargs.keys())
        if len(missing_modules) > 0:
            raise ValueError(
                f'Pipeline {inpainting_cls} expected {expected_modules}, but only {set(list(passed_class_obj.keys()) + list(original_class_obj.keys()))} were passed'
            )
        model = inpainting_cls(**inpainting_kwargs)
        model.register_to_config(_name_or_path=pretrained_model_name_or_path)
        model.register_to_config(**unused_original_config)
        return model