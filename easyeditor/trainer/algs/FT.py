import copy
import logging
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from collections import deque
from tqdm import tqdm, trange

from . import local_nn
from .editable_model import EditableModel
from ..utils import _inner_params, _logits

LOG = logging.getLogger(__name__)


class FT(EditableModel):
    """
    Fine-Tuning (FT) baseline for multimodal editing.
    Also supports LoRA mode with peft=True and mode='visual'/'textual'.
    """
    def __init__(self, model, config, model_constructor):
        super().__init__(model, config, model_constructor)

        if not str(self.config.device).startswith('cuda'):
            self.config.device = f'cuda:{self.config.device}'
        self.model = self.model.to(torch.float32)
        self.save_weight = None
        
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(
            prefix=prefix, keep_vars=keep_vars
        )  # Get default state dict
        state_dict["model_config"] = self.model.config  # Include model config
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        config = state_dict["model_config"]
        del state_dict["model_config"]
        if config != self.model.config:
            LOG.info("Loaded model config doesn't match current model config.")
            LOG.info(f"Loaded: {config}")
            LOG.info(f"Current: {self.model.config}")

        res = super().load_state_dict(state_dict, True)
        assert len(res.unexpected_keys) == 0, "Shouldn't have any unexpected keys"
        return res

    # Inference
    def forward(self, *inputs, **kwargs):
        if 'minigpt4' in self.config.model_name.lower() or 'blip' in self.config.model_name.lower() or 'llava' in self.config.model_name.lower():
            use_lora = getattr(self.config, 'use_lora', False)
            lora_connector_type = getattr(self.config, 'lora_connector_type', None)
            
            if use_lora:
                if 'blip' in self.config.model_name.lower() or 'minigpt4' in self.config.model_name.lower():
                    outputs = self.model(*inputs, **kwargs)
                else:  # LLAVA - Unwrapping PeftModelForCausalLM
                    outputs = self.model.base_model(*inputs, **kwargs)
            else:
                outputs = self.model(*inputs, **kwargs)  # FT, custom model
        else:
            raise NotImplementedError("Model not supported")
        return outputs
    
    def outer_parameters(self):
        return None

    # Edit(Update Model)
    def edit(self, batch, condition=None, detach_history=False, return_factors=False, connector_mode=False, mode=None, peft=None):
        self.model.train()
        
        # Safe config access
        use_lora = getattr(self.config, 'use_lora', False)
        lora_connector_type = getattr(self.config, 'lora_connector_type', None)
        inner_params = getattr(self.config, 'inner_params', [])

        ## Update target parameters: inner_params / LoRA... ##
        if not inner_params:  # inner_params is empty
            if connector_mode: 
                weights = {
                    n: p
                    for n, p in self.model.named_parameters()
                    if ("connector" in n)  # MLP parameters only
                }
            else:  # without Connector 
                if peft:  # for peft 
                    if mode == "visual":
                        # Select visual adapter parameters only (exclude default)
                        weights = {n: p for n, p in self.model.named_parameters() 
                                if "lora" in n and "visual" in n}
                    elif mode == "textual":
                        # Select textual adapter parameters only
                        weights = {n: p for n, p in self.model.named_parameters() 
                                if "lora" in n and "textual" in n}
                    elif mode == "fusion":
                        print("Not implemented - connector to be written")
                        weights = {}
                    else:  # "one" lora 
                        weights = {
                            n: p
                            for n, p in self.model.named_parameters()
                            if "lora" in n  # LoRA parameters only
                        }
                    
                else:  # for custom code
                    if mode == "visual":
                        weights = {
                            n: p
                            for n, p in self.model.named_parameters()
                            if "lora_visual" in n
                        }
                    elif mode == "textual":
                        weights = {
                            n: p
                            for n, p in self.model.named_parameters()
                            if "lora_textual" in n
                        }
                    else:  # "one" lora 
                        weights = {
                            n: p
                            for n, p in self.model.named_parameters()
                            if "lora" in n
                        }
        
        elif inner_params[0] in ['Qformer', 'mm_projector']:
            weights = {
                n: p
                for n, p in self.model.named_parameters()
                if n.find(inner_params[0]) != -1
            }
        else:
            names = set([n for n, p in self.model.named_parameters()])
            pset = set(inner_params)
            for p in pset:
                assert p in names, f"inner param {p} not in model"

            weights = {
                n: p
                for n, p in self.model.named_parameters()
                if n in pset
            }

        # Set edit learning rate
        if connector_mode: 
            edit_lr = self.config.edit_lr / 5
        else:
            edit_lr = self.config.edit_lr

        opt = torch.optim.AdamW(
            [v for _, v in weights.items()],
            lr=edit_lr
        )
                   
        # Set requires_grad for target parameters
        for name, w in self.model.named_parameters():
            w.requires_grad = name in weights

        if 'minigpt4' in self.config.model_name.lower() or 'blip' in self.config.model_name.lower() or 'llava' in self.config.model_name.lower():
            pbar = trange(self.config.num_steps, ncols=120)
            for it in pbar:
                opt.zero_grad()

                ### For Edit with LoRA, !Unwrapping! is required ###
                if use_lora or lora_connector_type in ["attention", "ffn"]: 
                    if 'blip' in self.config.model_name.lower() or 'minigpt' in self.config.model_name.lower():
                        outputs = self.model(batch)
                    elif 'llava' in self.config.model_name.lower():
                        outputs = self.model.model(batch)  # PeftModelForCausalLM -> LlavaLlamaForCausalLM
                
                elif lora_connector_type == "one":
                    if connector_mode:
                        for module in self.model.modules():
                            if hasattr(module, "use_connector"):
                                module.use_connector()

                    outputs = self.model(batch)

                elif lora_connector_type == "two":
                    for module in self.model.modules():
                        if hasattr(module, "use_vis_adapter"):
                            if mode == "visual":
                                module.use_vis_adapter()
                            elif mode == "textual" and hasattr(module, "use_text_adapter"):
                                module.use_text_adapter()
                            elif mode == "fusion" and hasattr(module, "use_connector"):
                                module.use_connector()

                    outputs = self.model(batch)

                else:
                    outputs = self.model(batch)  # FT: LlavaLlamaForCausalLM

                if not isinstance(outputs, torch.Tensor):
                    outputs = outputs.logits
                loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]
                pbar.set_postfix({"loss": loss.item()})
                
                torch.autograd.set_detect_anomaly(True)
                loss.backward()

                opt.step()

                if connector_mode and it >= 2:  # connector updates 3 times only
                    break

        else:
            raise NotImplementedError("Model not supported")

        edited_model = self.model

        return (
            FT(
                edited_model,
                self.config,
                self.model_constructor,
            ),
            {}
        )
