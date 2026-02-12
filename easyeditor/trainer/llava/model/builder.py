#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from . import *
from ..constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

@dataclass
class ModelVisonArguments:
    vision_tower: Optional[str] = field(default='openai/clip-vit-large-patch14-336')
    mm_vision_select_layer: Optional[int] = field(default=-2)
    pretrain_mm_mlp_adapter: Optional[str] = field(default='hugging_cache/llava-v1.5-7b/mm_projector.bin')
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
    mm_vision_select_feature: Optional[str] = field(default="patch")

def load_pretrained_model(model_path, 
                          load_8bit=False, 
                          load_4bit=False, 
                          device_map="auto", 
                          device="cuda", 
                          use_lora=False,
                          lora_rank=8,
                          lora_alpha=32,
                          lora_dropout=0.1,
                          lora_target_modules=['down_proj', 'up_proj'],
                          for_eval: Optional[str] = None,
                          adapter_path: Optional[str] = None,
                          one_lora: bool = False,  # Added: One LoRA baseline support
                          **kwargs) -> LlavaLlamaForCausalLM:
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    # delete keys related to LoRA
    for key in ["lora_r", "lora_alpha", "lora_dropout", "lora_target_modules", "use_lora", "inner_params"]:
        if key in kwargs:
            print(kwargs.pop(key))

    # Load LLaVA model
    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    
    if use_lora:
        from peft import PeftMixedModel, get_peft_model, LoraConfig, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules
        )
        
        if one_lora:  # One LoRA baseline: single adapter for all editing
            model = get_peft_model(model, lora_config)
            print("One LoRA baseline builded (single adapter)")
        elif for_eval:  # Evaluation with 3 adapters (visual/textual/connector)
            model = PeftMixedModel.from_pretrained(model, os.path.join(adapter_path, "visual"), "visual")
            model.load_adapter(os.path.join(adapter_path, "textual"), adapter_name="textual")
            model.load_adapter(os.path.join(adapter_path, "connector"), adapter_name="connector")
            print('adapters loaded from ' + adapter_path)
        else:  # Training with 3 adapters (visual/textual/connector)
            model = PeftMixedModel(model, lora_config, adapter_name="visual")
            model.add_adapter(peft_config=lora_config, adapter_name="textual")

            # Attention connector
            connector_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj"]
            )

            model.add_adapter(peft_config=connector_config, adapter_name="connector")
            print("LORA(vis/text lora + connector) builded")
    
    # initialize vision modeles
    if '13b' in model_path:
        model_args = ModelVisonArguments(pretrain_mm_mlp_adapter='hugging_cache/llava-v1.5-13b/mm_projector.bin')
    else:
        model_args = ModelVisonArguments()
    model.get_model().initialize_vision_modules(model_args)
    return model
