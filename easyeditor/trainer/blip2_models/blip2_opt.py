"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
import logging

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from .blip2 import Blip2Base, disabled_train
from .modeling_opt import OPTForCausalLM, OPTConfig
from transformers import AutoTokenizer
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class BLIP2Output(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    labels: torch.IntTensor = None
    attention_mask: torch.IntTensor = None


class Blip2OPT(Blip2Base):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",
        "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",
        "caption_coco_opt2.7b": "configs/models/blip2/blip2_caption_opt2.7b.yaml",
        "caption_coco_opt6.7b": "configs/models/blip2/blip2_caption_opt6.7b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        opt_model="facebook/opt-2.7b",
        prompt="",
        max_txt_len=32,
        state_dict_file=None,
        qformer_name_or_path="bert-base-uncased",
        qformer_checkpoint="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_opt2.7b.pth",
        # Added
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: list = ["q_proj", "v_proj"],
        connector_type: Optional[str] = None, 
        for_eval: Optional[bool] = None,
        adapter_path: Optional[str] = None
    ):
        super().__init__()
        self.config = None
        self.tokenizer = self.init_tokenizer(qformer_name_or_path)

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision, state_dict_file
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, qformer_name_or_path
        ) # query_token?
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model, torch_dtype=torch.float16
        )

        # Added
        if use_lora:
            print('Loading Adapters')
            from peft import get_peft_model, LoraConfig, TaskType, PeftMixedModel
            lora_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM, # 
                        r= lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        target_modules=lora_target_modules
                )
            
            if for_eval: # test(load pretrained weights)
                self.opt_model = PeftMixedModel.from_pretrained(self.opt_model, os.path.join(adapter_path, "visual"), "visual")
                self.opt_model.load_adapter(os.path.join(adapter_path, "textual"), adapter_name="textual")
                self.opt_model.load_adapter(os.path.join(adapter_path, "connector"), adapter_name="connector")
            else: # train(add new adapter)
                self.opt_model = PeftMixedModel(self.opt_model, lora_config, adapter_name="visual")
                self.opt_model.add_adapter(peft_config=lora_config, adapter_name="textual")

                connector_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_r,
                    lora_alpha=16,
                    lora_dropout=0.1,
                    target_modules=["q_proj", "k_proj"]
                    )

                self.opt_model.add_adapter(peft_config=connector_config, adapter_name="connector")



        # for name, param in self.opt_model.named_parameters():
        #     param.requires_grad = False
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )
        
        print('Loading Q-Former and Linear')
        self.load_from_pretrained(url_or_filename=qformer_checkpoint)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        print('Loading Q-Former and Linear Done')
        
        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

    def forward(self, samples):
        if samples['image'] is not None:
            image = samples["image"] # bsz, 3, image_size, image_size
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.opt_proj(query_output.last_hidden_state) # project image emb from 768（Bert size）to OPT size 2560
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

            self.opt_tokenizer.padding_side = "right"

            text = [t for t in samples["text_input"]]
            text_labels = [t for t in samples["labels"]]

            opt_tokens = self.opt_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                # truncation=True,
                # max_length=self.max_txt_len,
            ).to(image.device)

            targets = opt_tokens.input_ids.masked_fill(
                opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
            )
            if samples['prompts_len']:
                # targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt
                for i, prompt_len in enumerate(samples['prompts_len']):
                    targets[i, :prompt_len] = -100

            empty_targets = (
                torch.ones(atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100)
            )
            targets = torch.cat([empty_targets, targets], dim=1)

            from peft import PeftMixedModel
            if isinstance(self.opt_model, PeftMixedModel):
                embed_layer   = self.opt_model.get_input_embeddings()      
                inputs_embeds = embed_layer(opt_tokens.input_ids) 
            else:
                inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
            # print('input_image', inputs_opt.size())
            inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
        else:
            text = [t for t in samples["text_input"]]

            opt_tokens = self.opt_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                # truncation=True,
                # max_length=self.max_txt_len,
            ).to(self.opt_model.device)

            targets = opt_tokens.input_ids.masked_fill(
                opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
            )
            
            if samples['prompts_len']:
                for i, prompt_len in enumerate(samples['prompts_len']):
                    targets[i, :prompt_len] = -100

            from peft import PeftMixedModel
            if isinstance(self.opt_model, PeftMixedModel):
                embed_layer   = self.opt_model.get_input_embeddings()      
                inputs_embeds = embed_layer(opt_tokens.input_ids) 
            else:
                inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
            attention_mask = opt_tokens.attention_mask

        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds, # inputs_embeds is the fusion of the image embeddings and the caption embeddings
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        # print('input', inputs_embeds.size())
        # print('output', outputs.logits.size())

        # return {"loss": loss, "logits": outputs.logits}
        if torch.isnan(outputs.logits).any():
            print("blip logits has nan!!!!!!!!!!!!")
        return BLIP2Output(
            loss=loss,
            logits=outputs.logits,
            labels=targets,
            attention_mask=attention_mask
        )


