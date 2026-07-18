from typing import Union

from .scheduling import (evaluate_prompt_schedule, evaluate_value_schedule, extract_cond_from_schedule, TensorInterp, PromptOptions,
                         verify_key_value)
from .utils_model import BIGMAX
from .logger import logger

class PromptSchedulingLatentsNode:
    NodeID = 'ADE_PromptSchedulingLatents'
    NodeName = 'Prompt Scheduling [Latents] 🎭🅐🅓'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompts": ("STRING", {"multiline": True, "default": ''}),
                "clip": ("CLIP",),
                "latent": ("LATENT",),
            },
            "optional": {
                "prepend_text": ("STRING", {"multiline": True, "default": '', "forceInput": True}),
                "append_text": ("STRING", {"multiline": True, "default": '', "forceInput": True}),
                "values_replace": ("VALUES_REPLACE",),
                "print_schedule": ("BOOLEAN", {"default": False}),
                "tensor_interp": (TensorInterp._LIST,)
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT",)
    CATEGORY = "Animate Diff 🎭🅐🅓/scheduling"
    FUNCTION = "create_schedule"
    DESCRIPTION = 'Encode a schedule of prompts with automatic interpolation, its length matching passed-in latent count.'

    def create_schedule(self, prompts: str, clip, latent: dict, print_schedule=False, tensor_interp=TensorInterp.LERP,
                        prepend_text='', append_text='', values_replace=None):
        options = PromptOptions(interp=tensor_interp, prepend_text=prepend_text, append_text=append_text,
                                values_replace=values_replace, print_schedule=print_schedule)
        conditioning = evaluate_prompt_schedule(prompts, latent["samples"].size(0), clip, options)
        return (conditioning, latent)


class PromptSchedulingNode:
    NodeID = 'ADE_PromptScheduling'
    NodeName = 'Prompt Scheduling 🎭🅐🅓'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompts": ("STRING", {"multiline": True, "default": ''}),
                "clip": ("CLIP",),
            },
            "optional": {
                "prepend_text": ("STRING", {"multiline": True, "default": '', "forceInput": True}),
                "append_text": ("STRING", {"multiline": True, "default": '', "forceInput": True}),
                "values_replace": ("VALUES_REPLACE",),
                "print_schedule": ("BOOLEAN", {"default": False}),
                "max_length": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "tensor_interp": (TensorInterp._LIST,)
            },
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "Animate Diff 🎭🅐🅓/scheduling"
    FUNCTION = "create_schedule"
    DESCRIPTION = 'Encode a schedule of prompts with automatic interpolation.'

    def create_schedule(self, prompts: str, clip, print_schedule=False, max_length: int=0, tensor_interp=TensorInterp.LERP,
                        prepend_text='', append_text='', values_replace=None):
        options = PromptOptions(interp=tensor_interp, prepend_text=prepend_text, append_text=append_text,
                                values_replace=values_replace, print_schedule=print_schedule)
        conditioning = evaluate_prompt_schedule(prompts, max_length, clip, options)
        return (conditioning,)


class ValueSchedulingLatentsNode:
    NodeID = 'ADE_ValueSchedulingLatents'
    NodeName = 'Value Scheduling [Latents] 🎭🅐🅓'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "values": ("STRING", {"multiline": True, "default": ""}),
                "latent": ("LATENT",),
            },
            "optional": {
                "print_schedule": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("FLOAT", "FLOATS", "INT", "INTS")
    CATEGORY = "Animate Diff 🎭🅐🅓/scheduling"
    FUNCTION = "create_schedule"
    DESCRIPTION = 'Create a list of values with automatic interpolation, its length matching passed-in latent count.'

    def create_schedule(self, values: str, latent: dict, print_schedule=False):
        float_vals = evaluate_value_schedule(values, latent["samples"].size(0))
        int_vals = [round(x) for x in float_vals]
        if print_schedule:
            logger.info(f"ValueScheduling ({len(float_vals)} values):")
            for i, val in enumerate(float_vals):
                logger.info(f"{i} = {val}")
        return (float_vals, float_vals, int_vals, int_vals)


class ValueSchedulingNode:
    NodeID = 'ADE_ValueScheduling'
    NodeName = 'Value Scheduling 🎭🅐🅓'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "values": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "print_schedule": ("BOOLEAN", {"default": False}),
                "max_length": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
            },
        }

    RETURN_TYPES = ("FLOAT", "FLOATS", "INT", "INTS")
    CATEGORY = "Animate Diff 🎭🅐🅓/scheduling"
    FUNCTION = "create_schedule"
    DESCRIPTION = 'Create a list of values with automatic interpolation.'

    def create_schedule(self, values: str, max_length: int, print_schedule=False):
        float_vals = evaluate_value_schedule(values, max_length)
        int_vals = [round(x) for x in float_vals]
        if print_schedule:
            logger.info(f"ValueScheduling ({len(float_vals)} values):")
            for i, val in enumerate(float_vals):
                logger.info(f"{i} = {val}")
        return (float_vals, float_vals, int_vals, int_vals)


class AddValuesReplaceNode:
    NodeID = 'ADE_ValuesReplace'
    NodeName = 'Add Values Replace 🎭🅐🅓'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value_key": ("STRING", {"default": ""}),
                "floats": ("FLOATS",)
            },
            "optional": {
                "prev_replace": ("VALUES_REPLACE",),
            },
        }
    
    RETURN_TYPES = ("VALUES_REPLACE",)
    CATEGORY = "Animate Diff 🎭🅐🅓/scheduling"
    FUNCTION = "add_values_replace"
    DESCRIPTION = 'Add a values schedule bound to a key to be used in Prompt Scheduling node.'

    def add_values_replace(self, value_key: str, floats: Union[list[float]], prev_replace: dict=None):
        # key can only have a-z, A-Z, 0-9, and _ characters
        verify_key_value(key=value_key)
        # add/replace value floats
        if prev_replace is None:
            prev_replace = {}
        prev_replace = prev_replace.copy()
        if value_key in prev_replace:
            logger.warn(f"Value key '{value_key}' is already present - corresponding floats value will be overriden.")
        prev_replace[value_key] = floats
        return (prev_replace,)


class FloatToFloatsNode:
    NodeID = 'ADE_FloatToFloats'
    NodeName = 'Float to Floats 🎭🅐🅓'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "FLOAT": ("FLOAT", {"default": 39, "forceInput": True}),
            },
        }
    
    RETURN_TYPES = ("FLOATS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/scheduling"
    FUNCTION = "convert_to_floats"

    def convert_to_floats(self, FLOAT: Union[float, list[float]]):
        floats = None
        if isinstance(FLOAT, float):
            floats = [float(FLOAT)]
        else:
            floats = list(FLOAT)
        return (floats,)

class ConditionExtractionNode:
    NodeID = 'ADE_ConditionExtraction'
    NodeName = 'Condition Step Extraction 🎭🅐🅓'
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "index": ("INT", {"default": 0, "min": 0, "step": 1})
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "Animate Diff 🎭🅐🅓/scheduling"
    FUNCTION = "extract_conditioning"
    DESCRIPTION = 'Extract a single conditioning step from a schedule of prompts.'

    def extract_conditioning(self, conditioning, index: int=0):
        conditioning_step = extract_cond_from_schedule(conditioning, index)
        return (conditioning_step,)
