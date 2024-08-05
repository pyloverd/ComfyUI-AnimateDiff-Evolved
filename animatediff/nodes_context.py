import torch
from torch import Tensor
from typing import Union

import comfy.samplers
from comfy.model_patcher import ModelPatcher

from .context import (ContextFuseMethod, ContextOptions, ContextOptionsGroup, ContextSchedules,
                      generate_context_visualization)
from .context_extras import ContextExtrasGroup, ContextRef, ContextRefParams, ContextRefMode, NaiveReuse
from .utils_model import BIGMAX, MAX_RESOLUTION
from .utils_scheduling import convert_str_to_indexes


LENGTH_MAX = 128   # keep an eye on these max values;
STRIDE_MAX = 32    # would need to be updated
OVERLAP_MAX = 128  # if new motion modules come out


class LoopedUniformContextOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context_length": ("INT", {"default": 16, "min": 1, "max": LENGTH_MAX}),
                "context_stride": ("INT", {"default": 1, "min": 1, "max": STRIDE_MAX}),
                "context_overlap": ("INT", {"default": 4, "min": 0, "max": OVERLAP_MAX}),
                "closed_loop": ("BOOLEAN", {"default": False},),
                #"sync_context_to_pe": ("BOOLEAN", {"default": False},),
            },
            "optional": {
                "fuse_method": (ContextFuseMethod.LIST,),
                "use_on_equal_length": ("BOOLEAN", {"default": False},),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": BIGMAX}),
                "prev_context": ("CONTEXT_OPTIONS",),
                "view_opts": ("VIEW_OPTS",),
            }
        }
    
    RETURN_TYPES = ("CONTEXT_OPTIONS",)
    RETURN_NAMES = ("CONTEXT_OPTS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts"
    FUNCTION = "create_options"

    def create_options(self, context_length: int, context_stride: int, context_overlap: int, closed_loop: bool,
                       fuse_method: str=ContextFuseMethod.FLAT, use_on_equal_length=False, start_percent: float=0.0, guarantee_steps: int=1,
                       view_opts: ContextOptions=None, prev_context: ContextOptionsGroup=None):
        if prev_context is None:
            prev_context = ContextOptionsGroup()
        prev_context = prev_context.clone()

        context_options = ContextOptions(
            context_length=context_length,
            context_stride=context_stride,
            context_overlap=context_overlap,
            context_schedule=ContextSchedules.UNIFORM_LOOPED,
            closed_loop=closed_loop,
            fuse_method=fuse_method,
            use_on_equal_length=use_on_equal_length,
            start_percent=start_percent,
            guarantee_steps=guarantee_steps,
            view_options=view_opts,
            )
        #context_options.set_sync_context_to_pe(sync_context_to_pe)
        prev_context.add(context_options)
        return (prev_context,)


# This Legacy version exists to maintain compatiblity with old workflows
class LegacyLoopedUniformContextOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context_length": ("INT", {"default": 16, "min": 1, "max": LENGTH_MAX}),
                "context_stride": ("INT", {"default": 1, "min": 1, "max": STRIDE_MAX}),
                "context_overlap": ("INT", {"default": 4, "min": 0, "max": OVERLAP_MAX}),
                "context_schedule": (ContextSchedules.LEGACY_UNIFORM_SCHEDULE_LIST,),
                "closed_loop": ("BOOLEAN", {"default": False},),
                #"sync_context_to_pe": ("BOOLEAN", {"default": False},),
            },
            "optional": {
                "fuse_method": (ContextFuseMethod.LIST, {"default": ContextFuseMethod.FLAT}),
                "use_on_equal_length": ("BOOLEAN", {"default": False},),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": BIGMAX}),
                "prev_context": ("CONTEXT_OPTIONS",),
                "view_opts": ("VIEW_OPTS",),
                "deprecation_warning": ("ADEWARN", {"text": ""}),
            }
        }
    
    RETURN_TYPES = ("CONTEXT_OPTIONS",)
    RETURN_NAMES = ("CONTEXT_OPTS",)
    CATEGORY = ""  # No Category, so will not appear in menu
    FUNCTION = "create_options"

    def create_options(self, fuse_method: str=ContextFuseMethod.FLAT, context_schedule: str=None, **kwargs):
        return LoopedUniformContextOptionsNode.create_options(self, fuse_method=fuse_method, **kwargs)


class StandardUniformContextOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context_length": ("INT", {"default": 16, "min": 1, "max": LENGTH_MAX}),
                "context_stride": ("INT", {"default": 1, "min": 1, "max": STRIDE_MAX}),
                "context_overlap": ("INT", {"default": 4, "min": 0, "max": OVERLAP_MAX}),
            },
            "optional": {
                "fuse_method": (ContextFuseMethod.LIST,),
                "use_on_equal_length": ("BOOLEAN", {"default": False},),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": BIGMAX}),
                "prev_context": ("CONTEXT_OPTIONS",),
                "view_opts": ("VIEW_OPTS",),
            }
        }
    
    RETURN_TYPES = ("CONTEXT_OPTIONS",)
    RETURN_NAMES = ("CONTEXT_OPTS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts"
    FUNCTION = "create_options"

    def create_options(self, context_length: int, context_stride: int, context_overlap: int,
                       fuse_method: str=ContextFuseMethod.PYRAMID, use_on_equal_length=False, start_percent: float=0.0, guarantee_steps: int=1,
                       view_opts: ContextOptions=None, prev_context: ContextOptionsGroup=None):
        if prev_context is None:
            prev_context = ContextOptionsGroup()
        prev_context = prev_context.clone()

        context_options = ContextOptions(
            context_length=context_length,
            context_stride=context_stride,
            context_overlap=context_overlap,
            context_schedule=ContextSchedules.UNIFORM_STANDARD,
            closed_loop=False,
            fuse_method=fuse_method,
            use_on_equal_length=use_on_equal_length,
            start_percent=start_percent,
            guarantee_steps=guarantee_steps,
            view_options=view_opts,
            )
        prev_context.add(context_options)
        return (prev_context,)


class StandardStaticContextOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context_length": ("INT", {"default": 16, "min": 1, "max": LENGTH_MAX}),
                "context_overlap": ("INT", {"default": 4, "min": 0, "max": OVERLAP_MAX}),
            },
            "optional": {
                "fuse_method": (ContextFuseMethod.LIST_STATIC,),
                "use_on_equal_length": ("BOOLEAN", {"default": False},),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": BIGMAX}),
                "prev_context": ("CONTEXT_OPTIONS",),
                "view_opts": ("VIEW_OPTS",),
            }
        }
    
    RETURN_TYPES = ("CONTEXT_OPTIONS",)
    RETURN_NAMES = ("CONTEXT_OPTS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts"
    FUNCTION = "create_options"

    def create_options(self, context_length: int, context_overlap: int,
                       fuse_method: str=ContextFuseMethod.PYRAMID, use_on_equal_length=False, start_percent: float=0.0, guarantee_steps: int=1,
                       view_opts: ContextOptions=None, prev_context: ContextOptionsGroup=None):
        if prev_context is None:
            prev_context = ContextOptionsGroup()
        prev_context = prev_context.clone()
        
        context_options = ContextOptions(
            context_length=context_length,
            context_stride=None,
            context_overlap=context_overlap,
            context_schedule=ContextSchedules.STATIC_STANDARD,
            fuse_method=fuse_method,
            use_on_equal_length=use_on_equal_length,
            start_percent=start_percent,
            guarantee_steps=guarantee_steps,
            view_options=view_opts,
            )
        prev_context.add(context_options)
        return (prev_context,)


class BatchedContextOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context_length": ("INT", {"default": 16, "min": 1, "max": LENGTH_MAX}),
            },
            "optional": {
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": BIGMAX}),
                "prev_context": ("CONTEXT_OPTIONS",),
            }
        }
    
    RETURN_TYPES = ("CONTEXT_OPTIONS",)
    RETURN_NAMES = ("CONTEXT_OPTS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts"
    FUNCTION = "create_options"

    def create_options(self, context_length: int, start_percent: float=0.0, guarantee_steps: int=1,
                       prev_context: ContextOptionsGroup=None):
        if prev_context is None:
            prev_context = ContextOptionsGroup()
        prev_context = prev_context.clone()
        
        context_options = ContextOptions(
            context_length=context_length,
            context_overlap=0,
            context_schedule=ContextSchedules.BATCHED,
            start_percent=start_percent,
            guarantee_steps=guarantee_steps,
            )
        prev_context.add(context_options)
        return (prev_context,)


class ViewAsContextOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "view_opts_req": ("VIEW_OPTS",),
            },
            "optional": {
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": BIGMAX}),
                "prev_context": ("CONTEXT_OPTIONS",),
            }
        }
    
    RETURN_TYPES = ("CONTEXT_OPTIONS",)
    RETURN_NAMES = ("CONTEXT_OPTS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts"
    FUNCTION = "create_options"

    def create_options(self, view_opts_req: ContextOptions, start_percent: float=0.0, guarantee_steps: int=1,
                       prev_context: ContextOptionsGroup=None):
        if prev_context is None:
            prev_context = ContextOptionsGroup()
        prev_context = prev_context.clone()
        context_options = ContextOptions(
            context_schedule=ContextSchedules.VIEW_AS_CONTEXT,
            start_percent=start_percent,
            guarantee_steps=guarantee_steps,
            view_options=view_opts_req,
            use_on_equal_length=True
        )
        prev_context.add(context_options)
        return (prev_context,)


#########################
# View Options
class StandardStaticViewOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "view_length": ("INT", {"default": 16, "min": 1, "max": LENGTH_MAX}),
                "view_overlap": ("INT", {"default": 4, "min": 0, "max": OVERLAP_MAX}),
            },
            "optional": {
                "fuse_method": (ContextFuseMethod.LIST,),
            }
        }
    
    RETURN_TYPES = ("VIEW_OPTS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/view opts"
    FUNCTION = "create_options"

    def create_options(self, view_length: int, view_overlap: int,
                       fuse_method: str=ContextFuseMethod.FLAT,):
        view_options = ContextOptions(
            context_length=view_length,
            context_stride=None,
            context_overlap=view_overlap,
            context_schedule=ContextSchedules.STATIC_STANDARD,
            fuse_method=fuse_method,
            )
        return (view_options,)


class StandardUniformViewOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "view_length": ("INT", {"default": 16, "min": 1, "max": LENGTH_MAX}),
                "view_stride": ("INT", {"default": 1, "min": 1, "max": STRIDE_MAX}),
                "view_overlap": ("INT", {"default": 4, "min": 0, "max": OVERLAP_MAX}),
            },
            "optional": {
                "fuse_method": (ContextFuseMethod.LIST,),
            }
        }
    
    RETURN_TYPES = ("VIEW_OPTS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/view opts"
    FUNCTION = "create_options"

    def create_options(self, view_length: int, view_overlap: int, view_stride: int,
                       fuse_method: str=ContextFuseMethod.PYRAMID,):
        view_options = ContextOptions(
            context_length=view_length,
            context_stride=view_stride,
            context_overlap=view_overlap,
            context_schedule=ContextSchedules.UNIFORM_STANDARD,
            fuse_method=fuse_method,
            )
        return (view_options,)


class LoopedUniformViewOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "view_length": ("INT", {"default": 16, "min": 1, "max": LENGTH_MAX}),
                "view_stride": ("INT", {"default": 1, "min": 1, "max": STRIDE_MAX}),
                "view_overlap": ("INT", {"default": 4, "min": 0, "max": OVERLAP_MAX}),
                "closed_loop": ("BOOLEAN", {"default": False},),
            },
            "optional": {
                "fuse_method": (ContextFuseMethod.LIST,),
                "use_on_equal_length": ("BOOLEAN", {"default": False},),
            }
        }
    
    RETURN_TYPES = ("VIEW_OPTS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/view opts"
    FUNCTION = "create_options"

    def create_options(self, view_length: int, view_overlap: int, view_stride: int, closed_loop: bool,
                       fuse_method: str=ContextFuseMethod.PYRAMID, use_on_equal_length=False):
        view_options = ContextOptions(
            context_length=view_length,
            context_stride=view_stride,
            context_overlap=view_overlap,
            context_schedule=ContextSchedules.UNIFORM_LOOPED,
            closed_loop=closed_loop,
            fuse_method=fuse_method,
            use_on_equal_length=use_on_equal_length,
            )
        return (view_options,)


class VisualizeContextOptionsKAdv:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "context_opts": ("CONTEXT_OPTIONS",),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
            },
            "optional": {
                "visual_width": ("INT", {"min": 32, "max": MAX_RESOLUTION, "default": 1440}),
                "latents_length": ("INT", {"min": 1, "max": BIGMAX, "default": 32}),
                "steps": ("INT", {"min": 0, "max": BIGMAX, "default": 20}),
                "start_step": ("INT", {"min": 0, "max": BIGMAX, "default": 0}),
                "end_step": ("INT", {"min": 1, "max": BIGMAX, "default": 20}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/visualize"
    FUNCTION = "visualize"

    def visualize(self, model: ModelPatcher, context_opts: ContextOptionsGroup, sampler_name: str, scheduler: str,
                  visual_width: 1280, latents_length=32, steps=20, start_step=0, end_step=20):
        images = generate_context_visualization(context_opts=context_opts, model=model, width=visual_width, video_length=latents_length,
                                                sampler_name=sampler_name, scheduler=scheduler,
                                                steps=steps, start_step=start_step, end_step=end_step)
        return (images,)


class VisualizeContextOptionsK:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "context_opts": ("CONTEXT_OPTIONS",),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
            },
            "optional": {
                "visual_width": ("INT", {"min": 32, "max": MAX_RESOLUTION, "default": 1440}),
                "latents_length": ("INT", {"min": 1, "max": BIGMAX, "default": 32}),
                "steps": ("INT", {"min": 0, "max": BIGMAX, "default": 20}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/visualize"
    FUNCTION = "visualize"

    def visualize(self, model: ModelPatcher, context_opts: ContextOptionsGroup, sampler_name: str, scheduler: str,
                  visual_width: 1280, latents_length=32, steps=20, denoise=1.0):
        images = generate_context_visualization(context_opts=context_opts, model=model, width=visual_width, video_length=latents_length,
                                                sampler_name=sampler_name, scheduler=scheduler,
                                                steps=steps, denoise=denoise)
        return (images,)


class VisualizeContextOptionsSCustom:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "context_opts": ("CONTEXT_OPTIONS",),
                "sigmas": ("SIGMAS", ),
            },
            "optional": {
                "visual_width": ("INT", {"min": 32, "max": MAX_RESOLUTION, "default": 1440}),
                "latents_length": ("INT", {"min": 1, "max": BIGMAX, "default": 32}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/visualize"
    FUNCTION = "visualize"

    def visualize(self, model: ModelPatcher, context_opts: ContextOptionsGroup, sigmas,
                  visual_width: 1280, latents_length=32):
        images = generate_context_visualization(context_opts=context_opts, model=model, width=visual_width, video_length=latents_length,
                                                sigmas=sigmas)
        return (images,)


#########################
# Context Extras
class SetContextExtrasOnContextOptions:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context_opts": ("CONTEXT_OPTIONS",),
                "context_extras": ("CONTEXT_EXTRAS",),
            },
            "optional": {
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("CONTEXT_OPTIONS",)
    RETURN_NAMES = ("CONTEXT_OPTS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/context extras"
    FUNCTION = "set_context_extras"

    def set_context_extras(self, context_opts: ContextOptionsGroup, context_extras: ContextExtrasGroup):
        context_opts = context_opts.clone()
        context_opts.extras = context_extras.clone()
        return (context_opts,)


class ContextExtras_NaiveReuse:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "prev_extras": ("CONTEXT_EXTRAS",),
                "strength_multival": ("MULTIVAL",),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.001}),
                "weighted_mean": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.001}),
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("CONTEXT_EXTRAS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/context extras"
    FUNCTION = "create_context_extra"

    def create_context_extra(self, start_percent=0.0, end_percent=0.1, weighted_mean=0.95, strength_multival: Union[float, Tensor]=None,
                             prev_extras: ContextExtrasGroup=None):
        if prev_extras is None:
            prev_extras = prev_extras = ContextExtrasGroup()
        prev_extras = prev_extras.clone()
        # create extra
        naive_reuse = NaiveReuse(start_percent=start_percent, end_percent=end_percent, weighted_mean=weighted_mean, multival_opt=strength_multival)
        prev_extras.add(naive_reuse)
        return (prev_extras,)


class ContextExtras_ContextRef:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "prev_extras": ("CONTEXT_EXTRAS",),
                "strength_multival": ("MULTIVAL",),
                "contextref_mode": ("CONTEXTREF_MODE",),
                "contextref_tune": ("CONTEXTREF_TUNE",),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.001}),
                "autosize": ("ADEAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("CONTEXT_EXTRAS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/context extras"
    FUNCTION = "create_context_extra"

    def create_context_extra(self, start_percent=0.0, end_percent=0.1, strength_multival: Union[float, Tensor]=None,
                             contextref_mode: ContextRefMode=None,
                             contextref_tune: ContextRefParams=None,
                             prev_extras: ContextExtrasGroup=None):
        if prev_extras is None:
            prev_extras = prev_extras = ContextExtrasGroup()
        prev_extras = prev_extras.clone()
        # create extra
        # TODO: make customizable, and allow mask input
        if contextref_tune is None:
            contextref_tune = ContextRefParams(attn_style_fidelity=1.0, attn_ref_weight=1.0, attn_strength=1.0)
        if contextref_mode is None:
            contextref_mode = ContextRefMode.init_first()
        context_ref = ContextRef(start_percent=start_percent, end_percent=end_percent, params=contextref_tune, mode=contextref_mode)
        prev_extras.add(context_ref)
        return (prev_extras,)


class ContextRef_ModeFirst:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "autosize": ("ADEAUTOSIZE", {"padding": 25}),
            },
        }
    
    RETURN_TYPES = ("CONTEXTREF_MODE",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/context extras/ContextRef"
    FUNCTION = "create_contextref_mode"

    def create_contextref_mode(self):
        mode = ContextRefMode.init_first()
        return (mode,)


class ContextRef_ModeSliding:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "sliding_width": ("INT", {"default": 2, "min": 2, "max": BIGMAX, "step": 1}),
                "autosize": ("ADEAUTOSIZE", {"padding": 42}),
            }
        }
    
    RETURN_TYPES = ("CONTEXTREF_MODE",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/context extras/ContextRef"
    FUNCTION = "create_contextref_mode"

    def create_contextref_mode(self, sliding_width):
        mode = ContextRefMode.init_sliding(sliding_width=sliding_width)
        return (mode,)


class ContextRef_ModeIndexes:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "switch_on_idxs": ("STRING", {"default": ""}),
                "always_include_0": ("BOOLEAN", {"default": True},),
                "autosize": ("ADEAUTOSIZE", {"padding": 50}),
            },
        }
    
    RETURN_TYPES = ("CONTEXTREF_MODE",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/context extras/ContextRef"
    FUNCTION = "create_contextref_mode"

    def create_contextref_mode(self, switch_on_idxs: str, always_include_0: bool):
        idxs = set(convert_str_to_indexes(indexes_str=switch_on_idxs, length=0, allow_range=False))
        if always_include_0 and 0 not in idxs:
            idxs.add(0)
        mode = ContextRefMode.init_indexes(indexes=idxs)
        return (mode,)


class ContextRef_TuneAttnAdain:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "attn_style_fidelity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "attn_ref_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "attn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "adain_style_fidelity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "adain_ref_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "adain_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "autosize": ("ADEAUTOSIZE", {"padding": 65}),
            }
        }
    
    RETURN_TYPES = ("CONTEXTREF_TUNE",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/context extras/ContextRef"
    FUNCTION = "create_contextref_tune"

    def create_contextref_tune(self, attn_style_fidelity=1.0, attn_ref_weight=1.0, attn_strength=1.0,
                        adain_style_fidelity=1.0, adain_ref_weight=1.0, adain_strength=1.0):
        params = ContextRefParams(attn_style_fidelity=attn_style_fidelity, adain_style_fidelity=adain_style_fidelity,
                                  attn_ref_weight=attn_ref_weight, adain_ref_weight=adain_ref_weight,
                                  attn_strength=attn_strength, adain_strength=adain_strength)
        return (params,)


class ContextRef_TuneAttn:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "attn_style_fidelity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "attn_ref_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "attn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "autosize": ("ADEAUTOSIZE", {"padding": 15}),
            }
        }
    
    RETURN_TYPES = ("CONTEXTREF_TUNE",)
    CATEGORY = "Animate Diff 🎭🅐🅓/context opts/context extras/ContextRef"
    FUNCTION = "create_contextref_tune"

    def create_contextref_tune(self, attn_style_fidelity=1.0, attn_ref_weight=1.0, attn_strength=1.0):
        return ContextRef_TuneAttnAdain.create_contextref_tune(self,
                                                               attn_style_fidelity=attn_style_fidelity, attn_ref_weight=attn_ref_weight, attn_strength=attn_strength,
                                                               adain_ref_weight=0.0, adain_style_fidelity=0.0, adain_strength=0.0)
