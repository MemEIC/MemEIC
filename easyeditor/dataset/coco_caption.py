"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
from collections import OrderedDict

from .processor.base_dataset import BaseDataset
from .processor.blip_processors import BlipImageEvalProcessor
from ..trainer.utils import dict_to
from PIL import Image
import random
import typing
import torch
import transformers
from transformers import AutoTokenizer
from tqdm import tqdm
from copy import deepcopy


class CaptionDataset(BaseDataset):
    def __init__(self, data_dir: str, size:  typing.Optional[int] = None, config=None, no_image=False, hop=None, *args, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # get tokenizer and vis_processor
        if config.model_class == "Blip2OPT":
            vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
        elif config.model_class == "LLaVA":
            vis_processor = transformers.CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        else:
            raise NotImplementedError("unknown model class")

        if config.alg == 'OURS':
            self.vis_processor_classifier = getattr(transformers, config.cls_image_proc_class).from_pretrained(config.cls_image_name)

        if (config is not None and hasattr(config, 'tokenizer_name')):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.name
            )
            tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                tok_name, trust_remote_code=True
            )            
            if tokenizer.pad_token == None or tokenizer.pad_token == '':
                tokenizer.pad_token = tokenizer.eos_token  
                
        vis_root = config.coco_image
        rephrase_root = config.rephrase_image
        super().__init__(vis_processor, vis_root, rephrase_root, [data_dir])

        self.config = config
        self.tok = tokenizer
        self.max_length = 32

        self.prompt = "Question: {} Short answer: "

        if config.alg == 'OURS' and config.query_generate == False:
            file_dir = os.path.dirname(data_dir)
            file_name = os.path.basename(data_dir)
            self.annotation = json.load(open(os.path.join(file_dir, 'prompt', file_name), 'r'))

        data = []
        if size is not None:
            self.annotation = self.annotation[:size]
        if hop:
            self.hop = hop
            assert int(hop) in [1, 2, 3, 4], "hop should be 1, 2, 3, or 4"
            port_types = ['', '1-hop', '2-hop', '3-hop', '4-hop']
            port_type = port_types[int(hop)]
        elif hop == None:
            self.hop = 1
            port_type = '1-hop'

        for record in tqdm(self.annotation, ncols=120, desc='Loading Data'):
            
            if record['alt'] == "":
                continue
            if hop and 'port_new' not in record.keys():
                continue
            
            image_path = os.path.join(self.vis_root, record["image"])
            rephrase_image_path = os.path.join(self.rephrase_root, record["image_rephrase"])
            locality_image_path = os.path.join(self.vis_root, record['m_loc'])
            
            item = {
                'prompt': record['src'],
                'pred': record['pred'],
                'target': record['alt'],
                'rephrase_prompt': record['rephrase'],
                # 'image': image,
                # 'image_rephrase': rephrase_image,
                'image': image_path,
                'image_rephrase': rephrase_image_path,
                'cond': "{} >> {} || {}".format(
                    record['pred'],
                    record['alt'],
                    record['src']
                )
            }
            
            item['locality_prompt'] = record['loc']
            item['locality_ground_truth'] = record['loc_ans']
            
            # item['multimodal_locality_image'] = locality_image
            item['multimodal_locality_image'] = locality_image_path

            item['multimodal_locality_prompt'] = record['m_loc_q']
            item['multimodal_locality_ground_truth'] = record['m_loc_a']

            if hop and 'port_new' in record.keys():
                item['portability_prompt'] = []
                item['portability_ground_truth'] = []
                find_hop = False
                for ports in record['port_new']:
                    if ports['port_type'] == port_type:
                        find_hop = True
                        port_q = ports['Q&A']['Question']
                        port_a = ports['Q&A']['Answer']
                        item['portability_prompt'].append(port_q)
                        item['portability_ground_truth'].append(port_a)
                        if config.alg == 'OURS' and config.query_generate == False:
                            item['portability_prompt_query'] = ports['Q&A']['Query']
                        break
                
                if not find_hop:
                    continue
            
            if config.alg == 'OURS' and config.query_generate == False:
                item['prompt_query'] = record['src_q']
                item['rephrase_prompt_query'] = record['rephrase_q']
                item['multimodal_locality_prompt_query'] = record['m_loc_q_q']

            data.append(item)
            
        # if size is not None:
        #     data = data[:size]        
        self._data = data
        self.no_image = no_image

    def __getitem__(self, index):
        if self.no_image:
            return self._data[index]

        data = deepcopy(self._data[index])        
        # load image
        image_path = data['image']
        rephrase_image_path = data['image_rephrase']
        locality_image_path = data['multimodal_locality_image']
        
        image = Image.open(image_path).convert("RGB")
        rephrase_image = Image.open(rephrase_image_path).convert("RGB")
        locality_image = Image.open(locality_image_path).convert("RGB")

        if self.config.alg == 'OURS':
            image_cls = self.vis_processor_classifier(image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            rephrase_image_cls = self.vis_processor_classifier(rephrase_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            locality_image_cls = self.vis_processor_classifier(locality_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
        if self.config.model_class == "Blip2OPT":
            image = self.vis_processor(image)
            rephrase_image = self.vis_processor(rephrase_image)
            locality_image = self.vis_processor(locality_image)
        elif self.config.model_class == "LLaVA":
            image = self.vis_processor(image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            rephrase_image = self.vis_processor(rephrase_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            locality_image = self.vis_processor(locality_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
        else:
            raise NotImplementedError

        data['image'] = image
        data['image_rephrase'] = rephrase_image
        data['multimodal_locality_image'] = locality_image

        if self.config.alg == 'OURS':
            data['image_cls'] = image_cls
            data['image_rephrase_cls'] = rephrase_image_cls
            data['multimodal_locality_image_cls'] = locality_image_cls

        return data
    
    def __len__(self):
        return len(self._data)

    def collate_fn(self, batch):
        src = [b['prompt'] for b in batch]
        trg = [b['target'] for b in batch]
        cond = [b['cond'] for b in batch]
        rephrase = [b['rephrase_prompt'] for b in batch]
        image = [b['image'] for b in batch]
        image_rephrase = [b['image_rephrase'] for b in batch]
        loc_q = [b["locality_prompt"] for b in batch]
        loc_a = [b["locality_ground_truth"] for b in batch]
        m_loc_image = [b['multimodal_locality_image'] for b in batch]
        m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
        m_loc_a = [b['multimodal_locality_ground_truth'] for b in batch]

        if self.config.alg == 'OURS' and self.config.query_generate == False:
            src_q = [b['prompt_query'] for b in batch]
            rephrase_q = [b['rephrase_prompt_query'] for b in batch]
            m_loc_q_q = [b['multimodal_locality_prompt_query'] for b in batch]
            image_cls = [b['image_cls'] for b in batch]
            image_rephrase_cls = [b['image_rephrase_cls'] for b in batch]
            m_loc_image_cls = [b['multimodal_locality_image_cls'] for b in batch]
            if 'portability_prompt' in batch[0].keys():
                port_q_q = [b['portability_prompt_query'] for b in batch]

        tokenizer = AutoTokenizer.from_pretrained(self.config.name, use_fast=False) if self.config.model_name == "owl-2" else None
        
        # edit_inner
        edit_inner = {}
        edit_inner['image'] = torch.stack(image, dim=0)
        edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        edit_inner['labels'] = trg
        edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        if self.config.alg == 'SERAC_MULTI':
            edit_inner['prompt'] = src
        if self.config.alg == 'OURS' and self.config.query_generate == False:
            edit_inner['prompt'] = src
            edit_inner['answer'] = trg
            edit_inner['answers_len'] = [len(self.tok.encode(t, add_special_tokens=False)) for t in trg]
            edit_inner['image_cls'] = torch.stack(image_cls, dim=0)
            edit_inner['text_input_query'] = [s for s in src_q]
            #TODO: 여기에 필요한거 더 추가하기
            pass
        
        # edit_outer
        edit_outer = {}
        edit_outer['image'] = torch.stack(image, dim=0)
        edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(rephrase, trg)]
        edit_outer['labels'] = trg
        edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in rephrase]
        edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        if self.config.alg == 'SERAC_MULTI':
            edit_outer['prompt'] = rephrase
        if self.config.alg == 'OURS' and self.config.query_generate == False:
            edit_outer['prompt'] = rephrase
            edit_outer['answer'] = trg
            edit_outer['answers_len'] = [len(self.tok.encode(t, add_special_tokens=False)) for t in trg]
            edit_outer['image_cls'] = torch.stack(image_cls, dim=0)
            edit_outer['text_input_query'] = [r for r in rephrase_q]
            #TODO: 여기에 필요한거 더 추가하기
            pass
            
        # edit_outer_image
        edit_outer_image = {}
        edit_outer_image['image'] = torch.stack(image_rephrase, dim=0)
        edit_outer_image['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        edit_outer_image['labels'] = trg
        edit_outer_image['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        if self.config.alg == 'SERAC_MULTI':
            edit_outer_image['prompt'] = src
        if self.config.alg == 'OURS' and self.config.query_generate == False:
            edit_outer_image['prompt'] = src
            edit_outer_image['answer'] = trg
            edit_outer_image['answers_len'] = [len(self.tok.encode(t, add_special_tokens=False)) for t in trg]
            edit_outer_image['image_cls'] = torch.stack(image_rephrase_cls, dim=0)
            edit_outer_image['text_input_query'] = [r for r in src_q]
            #TODO: 여기에 필요한거 더 추가하기
            pass

        
        # loc
        loc = {}
        loc['image'] = None
        loc['text_input'] = [" ".join([q, a]) for q, a in zip(loc_q, loc_a)]
        loc['labels'] = loc_a
        loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
        loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        if self.config.alg == 'SERAC_MULTI':
            loc['prompt'] = loc_q
        if self.config.alg == 'OURS' and self.config.query_generate == False:
            loc['prompt'] = loc_q
            loc['answer'] = loc_a
            loc['answers_len'] = [len(self.tok.encode(a, add_special_tokens=False)) for a in loc_a]
            loc['image_cls'] = None
            loc['text_input_query'] = ['Image Level: None\nText Level: ' + q for q in loc_q]

        # m_loc
        loc_image = {}
        loc_image['image'] = torch.stack(m_loc_image, dim=0)
        loc_image['text_input'] = [" ".join([q, a]) for q, a in zip(m_loc_q, m_loc_a)]
        loc_image['labels'] = m_loc_a
        loc_image['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in m_loc_q]
        loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        if self.config.alg == 'SERAC_MULTI':
            loc_image['prompt'] = m_loc_q
        if self.config.alg == 'OURS' and self.config.query_generate == False:
            loc_image['prompt'] = m_loc_q
            loc_image['answer'] = m_loc_a
            loc_image['answers_len'] = [len(self.tok.encode(a, add_special_tokens=False)) for a in m_loc_a]
            loc_image['image_cls'] = torch.stack(m_loc_image_cls, dim=0)
            loc_image['text_input_query'] = [r for r in m_loc_q_q]

        # cond
        cond = self.tok(
            cond,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.config.device)

        edit_ports = None
        if 'portability_prompt' in batch[0].keys():
            edit_ports = []
            for port_q, port_a in zip(batch[0]['portability_prompt'], batch[0]['portability_ground_truth']):
                port = {}
                port['image'] = torch.stack(image, dim=0) if ("qwen-vl" not in self.config.model_name and "owl-2" not in self.config.model_name) else image
                port['text_input'] = [' '.join([port_q, port_a])]
                port['labels'] = [port_a]
                port['prompts_len'] = [len(self.tok.encode(port_q, add_special_tokens=False))]
                port['labels'] = self.tok([port_a], add_special_tokens=False, return_tensors="pt",)["input_ids"]
                if self.config.alg == 'SERAC_MULTI':
                    port['prompt'] = [port_q]
                if self.config.alg == 'OURS' and self.config.query_generate == False:
                    port['prompt'] = [port_q]
                    port['answer'] = [port_a]
                    port['answers_len'] = [len(self.tok.encode(port_a, add_special_tokens=False))]
                    port['image_cls'] = torch.stack(image_cls, dim=0) if ("qwen-vl" not in self.config.model_name and "owl-2" not in self.config.model_name) else image_cls
                    port['text_input_query'] = [q for q in port_q_q]
                edit_ports.append(port)

        
        batch_ = {
            "edit_inner": edit_inner,
            "edit_outer": edit_outer,
            "edit_outer_image": edit_outer_image,
            "loc": loc,
            "loc_image": loc_image,
            'port': edit_ports,
            "cond": cond
        }
        
        return dict_to(batch_, self.config.device)

class CompositionalCaptionDataset(BaseDataset):
    def __init__(self, data_dir: str, size:  typing.Optional[int] = None, config=None, no_image=False, hop=None, *args, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # get tokenizer and vis_processor
        if config.model_class == "Blip2OPT":
            vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
        elif config.model_class == "LLaVA":
            vis_processor = transformers.CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        elif config.model_class ==  "qwen-vl":
            vis_processor = BlipImageEvalProcessor(image_size=448, mean=None, std=None)
        elif "owl-2" in config.model_name.lower():
            from transformers.models.clip.image_processing_clip import CLIPImageProcessor
            vis_processor = CLIPImageProcessor.from_pretrained(config.name, trust_remote_code=True)
        else:
            raise NotImplementedError("unknown model class")

        if config.alg == 'OURS':
            self.vis_processor_classifier = getattr(transformers, config.cls_image_proc_class).from_pretrained(config.cls_image_name)

        if (config is not None and hasattr(config, 'tokenizer_name')):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.name
            )
            if config.tokenizer_class == "QWenTokenizer":
                tokenizer = AutoTokenizer.from_pretrained(config.name, trust_remote_code=True, pad_token='<|endoftext|>')
            elif config.model_name == "owl-2":
                tokenizer = AutoTokenizer.from_pretrained(config.name, use_fast=False, trust_remote_code=True)
            else:
                tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                    tok_name, trust_remote_code=True
                )            
            if tokenizer.pad_token == None or tokenizer.pad_token == '':
                tokenizer.pad_token = tokenizer.eos_token  
                
        vis_root = config.coco_image
        rephrase_root = config.rephrase_image
        super().__init__(vis_processor, vis_root, rephrase_root, [data_dir])

        self.config = config
        self.tok = tokenizer
        self.max_length = 32

        self.prompt = "Question: {} Short answer: "

        if config.alg == 'OURS' and config.query_generate == False:
            file_dir = os.path.dirname(data_dir)
            file_name = os.path.basename(data_dir)
            self.annotation = json.load(open(os.path.join(file_dir, 'prompt', file_name), 'r'))

        data = []
        if size is not None:
            self.annotation = self.annotation[:size]
        if hop:
            self.hop = hop
            assert int(hop) in [1, 2, 3, 4], "hop should be 1, 2, 3, or 4"
            port_types = ['', 'comp', '2-hop', '3-hop', '4-hop']
            port_type = port_types[int(hop)]
        elif hop == None:
            hop = 1
            self.hop = hop
            port_type = 'comp'

        for record in tqdm(self.annotation, ncols=120, desc='Loading Data'):
            
            if record['alt'] == "":
                continue
            if hop and 'port_new' not in record.keys():
                continue
            
            '''
            Visual Edit Dataset
            '''

            image_path = os.path.join(self.vis_root, record["image"])
            rephrase_image_path = os.path.join(self.rephrase_root, record["image_rephrase"])
            locality_image_path = os.path.join(self.vis_root, record['m_loc'])
            
            item = {
                'prompt': record['src'],
                'pred': record['pred'],
                'target': record['alt'],
                'rephrase_prompt': record['rephrase'],
                'image': image_path,
                'image_rephrase': rephrase_image_path,
                'cond': "{} >> {} || {}".format(
                    record['pred'],
                    record['alt'],
                    record['src']
                ),
                'locality_prompt': record['loc'],
                'locality_ground_truth': record['loc_ans'],
                'multimodal_locality_image': locality_image_path,
                'multimodal_locality_prompt': record['m_loc_q'],
                'multimodal_locality_ground_truth': record['m_loc_a']
            }

            if hop and 'port_new' in record.keys():
                item['portability_prompt'] = []
                item['portability_ground_truth'] = []
                find_hop = False
                for ports in record['port_new']:
                    if ports['port_type'] == port_type:
                        find_hop = True
                        port_q = ports['Q&A']['Question']
                        port_a = ports['Q&A']['Answer']
                        item['portability_prompt'].append(port_q)
                        item['portability_ground_truth'].append(port_a)
                        if config.alg == 'OURS' and config.query_generate == False:
                            item['portability_prompt_query'] = ports['Q&A']['Query']
                        break
                
                if not find_hop:
                    continue
            
            if config.alg == 'OURS' and config.query_generate == False:
                item['prompt_query'] = record['src_q']
                item['rephrase_prompt_query'] = record['rephrase_q']
                item['multimodal_locality_prompt_query'] = record['m_loc_q_q']

            '''
            Textual Edit dataset
            '''
            
            if "textual_edit" in record.keys():
                t_pred = " ".join(record["textual_edit"]['pred']) if isinstance(record["textual_edit"]['pred'], list) else record["textual_edit"]['pred']
                t_alt = " ".join(record["textual_edit"]['alt']) if isinstance(record["textual_edit"]['alt'], list) else record["textual_edit"]['alt']
                
                item["textual_edit"] = {
                    "prompt": record["textual_edit"]["src"],
                    "pred": t_pred,
                    "target": t_alt, 
                    "rephrase_prompt": record["textual_edit"]["rephrase"],
                    'cond': "{} >> {} || {}".format(
                        t_pred,
                        t_alt,
                        record["textual_edit"]['src']
                        ),
                    'locality_prompt': record["textual_edit"]['loc'],
                    'locality_ground_truth': record["textual_edit"]['loc_ans']
                }

                if hop and 'port_new' in record["textual_edit"].keys():
                    item["textual_edit"]['portability_prompt'] = []
                    item["textual_edit"]['portability_ground_truth'] = []
                    find_hop = False
                    for ports in record['port_new']:
                        if ports['port_type'] == port_type:
                            find_hop = True
                            port_q = ports['Q&A']['Question']
                            port_a = t_alt
                            item["textual_edit"]['portability_prompt'].append(port_q)
                            item["textual_edit"]['portability_ground_truth'].append(port_a)
                            if config.alg == 'OURS' and config.query_generate == False:
                                item["textual_edit"]['portability_prompt_query'] = ports['Q&A']['Query']
                            break
                    
                    if not find_hop:
                        continue

            data.append(item)
            
        # if size is not None:
        #     data = data[:size]        
        self._data = data
        self.no_image = no_image

    def __getitem__(self, index):
        if self.no_image:
            return self._data[index]

        data = deepcopy(self._data[index])        
        # load image
        image_path = data['image']
        rephrase_image_path = data['image_rephrase']
        locality_image_path = data['multimodal_locality_image']
        
        image = Image.open(image_path).convert("RGB")
        rephrase_image = Image.open(rephrase_image_path).convert("RGB")
        locality_image = Image.open(locality_image_path).convert("RGB")

        if self.config.alg == 'OURS':
            image_cls = self.vis_processor_classifier(image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            rephrase_image_cls = self.vis_processor_classifier(rephrase_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            locality_image_cls = self.vis_processor_classifier(locality_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
        if self.config.model_class == "Blip2OPT":
            image = self.vis_processor(image)
            rephrase_image = self.vis_processor(rephrase_image)
            locality_image = self.vis_processor(locality_image)
        elif self.config.model_class == "LLaVA":
            image = self.vis_processor(image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            rephrase_image = self.vis_processor(rephrase_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            locality_image = self.vis_processor(locality_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
        elif self.config.model_class == "qwen-vl":
            image = os.path.join(self.vis_root, image_path)
            rephrase_image = os.path.join(self.rephrase_root, rephrase_image_path)
            locality_image = os.path.join(self.vis_root, locality_image_path)
        else:
            raise NotImplementedError

        data['image'] = image
        data['image_rephrase'] = rephrase_image
        data['multimodal_locality_image'] = locality_image

        if self.config.alg == 'OURS':
            data['image_cls'] = image_cls
            data['image_rephrase_cls'] = rephrase_image_cls
            data['multimodal_locality_image_cls'] = locality_image_cls

        return data
    
    def __len__(self):
        return len(self._data)

    def collate_fn(self, batch):
        '''
        Visual Edit Dataset
        '''
        src = [b['prompt'] for b in batch]
        trg = [b['target'] for b in batch]
        cond = [b['cond'] for b in batch]
        rephrase = [b['rephrase_prompt'] for b in batch]
        image = [b['image'] for b in batch] if "owl-2" not in self.config.model_name else [b['image'] for b in batch][0]
        image_rephrase = [b['image_rephrase'] for b in batch] if "owl-2" not in self.config.model_name else [b['image_rephrase'] for b in batch][0]
        loc_q = [b["locality_prompt"] for b in batch]
        loc_a = [b["locality_ground_truth"] for b in batch]
        m_loc_image = [b['multimodal_locality_image'] for b in batch] if "owl-2" not in self.config.model_name else [b['multimodal_locality_image'] for b in batch][0]
        m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
        m_loc_a = [b['multimodal_locality_ground_truth'] for b in batch]

        if self.config.alg == 'OURS' and self.config.query_generate == False:
            src_q = [b['prompt_query'] for b in batch]
            rephrase_q = [b['rephrase_prompt_query'] for b in batch]
            m_loc_q_q = [b['multimodal_locality_prompt_query'] for b in batch]
            image_cls = [b['image_cls'] for b in batch]
            image_rephrase_cls = [b['image_rephrase_cls'] for b in batch]
            m_loc_image_cls = [b['multimodal_locality_image_cls'] for b in batch]
            if 'portability_prompt' in batch[0].keys():
                port_q_q = [b['portability_prompt_query'] for b in batch]
            if "textual_edit" in batch[0].keys() and "portability_prompt" in batch[0]["textual_edit"].keys():
                cport_q_q = [b["textual_edit"]["portability_prompt_query"] for b in batch]

        tokenizer = AutoTokenizer.from_pretrained(self.config.name, use_fast=False) if self.config.model_name == "owl-2" else None
        
        # edit_inner
        edit_inner = {}
        edit_inner['image'] = torch.stack(image, dim=0)
        edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        edit_inner['labels'] = trg
        edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        if self.config.alg == 'WISE':

            tok_output = self.tok(edit_inner['text_input'][0], add_special_tokens=False, return_tensors="pt",)
            edit_inner['text_tokens'] = {
                'input_ids': tok_output["input_ids"],
                'attention_mask': tok_output["attention_mask"],
                'labels': tok_output["input_ids"].clone()
            }
            prompt_len = self.tok(src[0], add_special_tokens=False, return_tensors="pt")["input_ids"].shape[1] 
            edit_inner['text_tokens']["labels"][:, :prompt_len] = -100

            edit_inner['ans_token_len'] = len(self.tok.encode(trg[0], add_special_tokens=False))
            
        if self.config.alg == 'SERAC_MULTI':
            edit_inner['prompt'] = src
        if self.config.alg == 'OURS' and self.config.query_generate == False:
            edit_inner['prompt'] = src
            edit_inner['answer'] = trg
            edit_inner['answers_len'] = [len(self.tok.encode(t, add_special_tokens=False)) for t in trg]
            edit_inner['image_cls'] = torch.stack(image_cls, dim=0)
            edit_inner['text_input_query'] = [s for s in src_q]
        
        # edit_outer
        edit_outer = {}
        edit_outer['image'] = torch.stack(image, dim=0)
        edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(rephrase, trg)]
        edit_outer['labels'] = trg
        edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in rephrase]
        edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        if self.config.alg == 'SERAC_MULTI':
            edit_outer['prompt'] = rephrase
        if self.config.alg == 'OURS' and self.config.query_generate == False:
            edit_outer['prompt'] = rephrase
            edit_outer['answer'] = trg
            edit_outer['answers_len'] = [len(self.tok.encode(t, add_special_tokens=False)) for t in trg]
            edit_outer['image_cls'] = torch.stack(image_cls, dim=0)
            edit_outer['text_input_query'] = [r for r in rephrase_q]
            
        # edit_outer_image
        edit_outer_image = {}
        edit_outer_image['image'] = torch.stack(image_rephrase, dim=0)
        edit_outer_image['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        edit_outer_image['labels'] = trg
        edit_outer_image['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        if self.config.alg == 'SERAC_MULTI':
            edit_outer_image['prompt'] = src
        if self.config.alg == 'OURS' and self.config.query_generate == False:
            edit_outer_image['prompt'] = src
            edit_outer_image['answer'] = trg
            edit_outer_image['answers_len'] = [len(self.tok.encode(t, add_special_tokens=False)) for t in trg]
            edit_outer_image['image_cls'] = torch.stack(image_rephrase_cls, dim=0)
            edit_outer_image['text_input_query'] = [r for r in src_q]

        
        # loc
        loc = {}
        loc['image'] = None
        loc['text_input'] = [" ".join([q, a]) for q, a in zip(loc_q, loc_a)]
        loc['labels'] = loc_a
        loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
        loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        if self.config.alg == 'SERAC_MULTI' or self.config.alg == 'WISE':
            loc['prompt'] = loc_q
            loc['answer'] = loc_a
        if self.config.alg == 'OURS' and self.config.query_generate == False:
            loc['prompt'] = loc_q
            loc['answer'] = loc_a
            loc['answers_len'] = [len(self.tok.encode(a, add_special_tokens=False)) for a in loc_a]
            loc['image_cls'] = None
            loc['text_input_query'] = ['Image Level: None\nText Level: ' + q for q in loc_q]

        # m_loc
        loc_image = {}
        loc_image['image'] = torch.stack(m_loc_image, dim=0)
        loc_image['text_input'] = [" ".join([q, a]) for q, a in zip(m_loc_q, m_loc_a)]
        loc_image['labels'] = m_loc_a
        loc_image['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in m_loc_q]
        loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        if self.config.alg == 'SERAC_MULTI' or self.config.alg == 'WISE':
            loc_image['prompt'] = m_loc_q
            loc_image['answer'] = m_loc_a
        if self.config.alg == 'OURS' and self.config.query_generate == False:
            loc_image['prompt'] = m_loc_q
            loc_image['answer'] = m_loc_a
            loc_image['answers_len'] = [len(self.tok.encode(a, add_special_tokens=False)) for a in m_loc_a]
            loc_image['image_cls'] = torch.stack(m_loc_image_cls, dim=0)
            loc_image['text_input_query'] = [r for r in m_loc_q_q]

        # cond
        cond = self.tok(
            cond,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.config.device)

        edit_ports = None
        if 'portability_prompt' in batch[0].keys():
            edit_ports = []
            for port_q, port_a in zip(batch[0]['portability_prompt'], batch[0]['portability_ground_truth']):
                port = {}
                port['image'] = torch.stack(image, dim=0) if ("qwen-vl" not in self.config.model_name and "owl-2" not in self.config.model_name) else image
                port['text_input'] = [' '.join([port_q, port_a])]
                port['labels'] = [port_a]
                port['prompts_len'] = [len(self.tok.encode(port_q, add_special_tokens=False))]
                port['labels'] = self.tok([port_a], add_special_tokens=False, return_tensors="pt",)["input_ids"]
                if self.config.alg == 'SERAC_MULTI':
                    port['prompt'] = [port_q]
                if self.config.alg == 'OURS' and self.config.query_generate == False:
                    port['prompt'] = [port_q]
                    port['answer'] = [port_a]
                    port['answers_len'] = [len(self.tok.encode(port_a, add_special_tokens=False))]
                    port['image_cls'] = torch.stack(image_cls, dim=0) if ("qwen-vl" not in self.config.model_name and "owl-2" not in self.config.model_name) else image_cls
                    port['text_input_query'] = [q for q in port_q_q]
                edit_ports.append(port)

        '''
        Textual Edit Dataset
        '''
        if "textual_edit" in batch[0].keys():
            t_src = [b['textual_edit']['prompt'] for b in batch]
            t_trg = [b['textual_edit']['target'] for b in batch]
            t_cond = [b['textual_edit']['cond'] for b in batch]
            t_rephrase = [b['textual_edit']['rephrase_prompt'] for b in batch]
            t_loc_q = [b['textual_edit']['locality_prompt'] for b in batch]
            t_loc_a = [b['textual_edit']['locality_ground_truth'] for b in batch]

            # edit_inner
            t_edit_inner = {}
            t_edit_inner['image'] = None
            t_edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(t_src, t_trg)]
            t_edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in t_src]
            t_edit_inner['labels'] = self.tok(t_trg, add_special_tokens=False, return_tensors="pt")["input_ids"]
            if self.config.alg == 'WISE':

                tok_output = self.tok(t_edit_inner['text_input'][0], add_special_tokens=False, return_tensors="pt",)
                t_edit_inner['text_tokens'] = {
                    'input_ids': tok_output["input_ids"],
                    'attention_mask': tok_output["attention_mask"],
                    'labels': tok_output["input_ids"].clone()
                }
                prompt_len = self.tok(t_src[0], add_special_tokens=False, return_tensors="pt")["input_ids"].shape[1] 
                t_edit_inner['text_tokens']["labels"][:, :prompt_len] = -100

                t_edit_inner['ans_token_len'] = len(self.tok.encode(t_trg[0], add_special_tokens=False))
            
            if self.config.alg == 'SERAC_MULTI':
                t_edit_inner['prompt'] = t_src
            if self.config.alg == 'OURS' and self.config.query_generate == False:
                t_edit_inner['prompt'] = t_src
                t_edit_inner['answer'] = t_trg
                t_edit_inner['answers_len'] = [len(self.tok.encode(t, add_special_tokens=False)) for t in t_trg]
                t_edit_inner['image_cls'] = None
                t_edit_inner['text_input_query'] = ['Image Level: None\nText Level: ' + q for q in t_src]

            # edit_outer
            t_edit_outer = {}
            t_edit_outer['image'] = None
            t_edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(t_rephrase, t_trg)] if t_rephrase else []
            t_edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in t_rephrase] if t_rephrase else []
            t_edit_outer['labels'] = self.tok(t_trg, add_special_tokens=False, return_tensors="pt")["input_ids"]
            if self.config.alg == 'SERAC_MULTI':
                t_edit_outer['prompt'] = t_rephrase
            if self.config.alg == 'OURS' and self.config.query_generate == False:
                t_edit_outer['prompt'] = t_rephrase
                t_edit_outer['answer'] = t_trg
                t_edit_outer['answers_len'] = [len(self.tok.encode(t, add_special_tokens=False)) for t in t_trg]
                t_edit_outer['image_cls'] = None
                t_edit_outer['text_input_query'] = ['Image Level: None\nText Level: ' + q for q in t_rephrase]


            # loc
            t_loc = {}
            t_loc['image'] = None
            t_loc['text_input'] = [" ".join([q, a]) for q, a in zip(t_loc_q, t_loc_a)]
            t_loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in t_loc_q]
            t_loc['labels'] = self.tok(t_loc_a, add_special_tokens=False, return_tensors="pt")["input_ids"]
            if self.config.alg == 'SERAC_MULTI':
                t_loc['prompt'] = t_loc_q
            if self.config.alg == 'OURS' and self.config.query_generate == False:
                t_loc['prompt'] = t_loc_q
                t_loc['answer'] = t_loc_a
                t_loc['answers_len'] = [len(self.tok.encode(t, add_special_tokens=False)) for t in t_loc_q]
                t_loc['image_cls'] = None
                t_loc['text_input_query'] = ['Image Level: None\nText Level: ' + q for q in t_loc_q]

            t_cond = self.tok(
                t_cond,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).to(self.config.device)

            # comp_port
            t_edit_port = None
            if 'portability_prompt' in batch[0]['textual_edit'].keys():
                t_edit_port = []
                for port_q, port_a in zip(batch[0]["textual_edit"]['portability_prompt'], batch[0]["textual_edit"]['portability_ground_truth']):
                    port = {}
                    port['image'] = torch.stack(image, dim=0) if ("qwen-vl" not in self.config.model_name and "owl-2" not in self.config.model_name) else image
                    port['text_input'] = [' '.join([port_q, port_a])]
                    port['labels'] = [port_a]
                    port['prompts_len'] = [len(self.tok.encode(port_q, add_special_tokens=False))]
                    port['labels'] = self.tok([port_a], add_special_tokens=False, return_tensors="pt",)["input_ids"]
                    if self.config.alg == 'SERAC_MULTI':
                        port['prompt'] = [port_q]
                    if self.config.alg == 'OURS' and self.config.query_generate == False:
                        port['prompt'] = [port_q]
                        port['answer'] = [port_a]
                        port['answers_len'] = [len(self.tok.encode(port_a, add_special_tokens=False))]
                        port['image_cls'] = torch.stack(image_cls, dim=0) if ("qwen-vl" not in self.config.model_name and "owl-2" not in self.config.model_name) else image_cls
                        port['text_input_query'] = [q for q in cport_q_q] 
                    t_edit_port.append(port)


        batch_ = {
            "edit_inner": edit_inner,
            "edit_outer": edit_outer,
            "edit_outer_image": edit_outer_image,
            "loc": loc,
            "loc_image": loc_image,
            'port': edit_ports,
            "cond": cond
        }
        if "textual_edit" in batch[0].keys():
            batch_["textual_edit"] = {
                "edit_inner": t_edit_inner,
                "edit_outer": t_edit_outer,
                "loc": t_loc,
                "cond": t_cond
            }
            if t_edit_port is not None:
                batch_["textual_edit"]["port"] = t_edit_port

        return dict_to(batch_, self.config.device)    

class CompositionalDataset_RAG_70(CompositionalCaptionDataset):
    """
    Compositional Dataset with RAG simulation (70% accuracy).
    Used for Knowledge Connector (KC) training (Stage 2).
    Inherits from CompositionalCaptionDataset, only overrides collate_fn for RAG simulation.
    """
    
    def collate_fn(self, batch):
        '''
        Visual Edit Dataset with RAG simulation (70% accuracy)
        '''
        src = [b['prompt'] for b in batch]
        trg = [b['target'] for b in batch]
        cond = [b['cond'] for b in batch]
        rephrase = [b['rephrase_prompt'] for b in batch]
        image = [b['image'] for b in batch] if "owl-2" not in self.config.model_name else [b['image'] for b in batch][0]
        image_rephrase = [b['image_rephrase'] for b in batch] if "owl-2" not in self.config.model_name else [b['image_rephrase'] for b in batch][0]
        loc_q = [b["locality_prompt"] for b in batch]
        loc_a = [b["locality_ground_truth"] for b in batch]
        m_loc_image = [b['multimodal_locality_image'] for b in batch] if "owl-2" not in self.config.model_name else [b['multimodal_locality_image'] for b in batch][0]
        m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
        m_loc_a = [b['multimodal_locality_ground_truth'] for b in batch]

        # edit_inner (Visual)
        v_edit_inner = {}
        v_edit_inner['image'] = torch.stack(image, dim=0)
        v_edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        v_edit_inner['labels'] = trg
        v_edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        v_edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # edit_outer (Visual)
        v_edit_outer = {}
        v_edit_outer['image'] = torch.stack(image, dim=0)
        v_edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(rephrase, trg)]
        v_edit_outer['labels'] = trg
        v_edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in rephrase]
        v_edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # edit_outer_image (Visual)
        v_edit_outer_image = {}
        v_edit_outer_image['image'] = torch.stack(image_rephrase, dim=0)
        v_edit_outer_image['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        v_edit_outer_image['labels'] = trg
        v_edit_outer_image['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        v_edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # loc (Visual)
        v_loc = {}
        v_loc['image'] = None
        v_loc['text_input'] = [" ".join([q, a]) for q, a in zip(loc_q, loc_a)]
        v_loc['labels'] = loc_a
        v_loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
        v_loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # loc_image (Visual)
        v_loc_image = {}
        v_loc_image['image'] = torch.stack(m_loc_image, dim=0)
        v_loc_image['text_input'] = [" ".join([q, a]) for q, a in zip(m_loc_q, m_loc_a)]
        v_loc_image['labels'] = m_loc_a
        v_loc_image['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in m_loc_q]
        v_loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # Textual Edit Data
        t_src = [b['textual_edit']['prompt'] for b in batch]
        t_trg = [b['textual_edit']['target'] for b in batch]
        t_loc_q = [b.get("textual_edit", {}).get("locality_prompt", "") for b in batch]
        t_loc_a = [b.get("textual_edit", {}).get("locality_ground_truth", "") for b in batch]

        # edit_inner (Textual)
        t_edit_inner = {}
        t_edit_inner['image'] = None
        t_edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(t_src, t_trg)]
        t_edit_inner['labels'] = t_trg
        t_edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in t_src]
        t_edit_inner['labels'] = self.tok(t_trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # edit_outer (Textual)
        t_rephrase = [b.get("textual_edit", {}).get("rephrase_prompt", "") for b in batch]
        t_edit_outer = {}
        t_edit_outer['image'] = None
        t_edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(t_rephrase, t_trg)] if t_rephrase else []
        t_edit_outer['labels'] = t_trg
        t_edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in t_rephrase] if t_rephrase else []
        t_edit_outer['labels'] = self.tok(t_trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # loc (Textual)
        t_loc = {}
        t_loc['image'] = None
        t_loc['text_input'] = [" ".join([q, a]) for q, a in zip(t_loc_q, t_loc_a)]
        t_loc['labels'] = t_loc_a
        t_loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in t_loc_q]
        t_loc['labels'] = self.tok(t_loc_a, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # Compositional Portability with RAG simulation (70% accuracy = 84% * 84%)
        prompt_template = (
            "Visual Editing Knowledge: {visual_info}\n"
            "Text Editing Knowledge: {textual_info}\n"
            "--------------------------------\n"
            "Question: {question}\n"
            "Answer: "
        )
        edit_ports = None
        if 'portability_prompt' in batch[0].keys():
            edit_ports = []
            for port_q, port_a in zip(batch[0]['portability_prompt'], batch[0]['portability_ground_truth']):
                port = {}
                
                # RAG simulation: 84% accuracy for visual edit data
                if random.random() < 0.84:
                    visual_src = batch[0]['prompt']
                    visual_trg = batch[0]['target']
                else:
                    rand_idx = random.choice(range(len(self._data)))
                    random_item = self._data[rand_idx]
                    visual_src = random_item['prompt']
                    visual_trg = random_item['target']

                # RAG simulation: 84% accuracy for textual edit data
                if random.random() < 0.84:
                    t_s = batch[0].get('textual_edit', {}).get('prompt', '')
                    t_t = batch[0].get('textual_edit', {}).get('target', '')
                else:
                    rand_idx = random.choice(range(len(self._data)))
                    random_item = self._data[rand_idx]
                    t_s = random_item.get('textual_edit', {}).get('prompt', '')
                    t_t = random_item.get('textual_edit', {}).get('target', '')

                visual_info = f"{visual_src} -> {visual_trg}"
                textual_info = f"{t_s} -> {t_t}"
                question = port_q

                prompt = prompt_template.format(
                    visual_info=visual_info,
                    textual_info=textual_info,
                    question=question
                )

                port['text_input'] = [" ".join([prompt, port_a])]
                port['labels'] = self.tok([port_a], add_special_tokens=False, return_tensors="pt")["input_ids"]
                port['prompts_len'] = [len(self.tok.encode(prompt, add_special_tokens=False))]
                port['image'] = torch.stack(image, dim=0)

                edit_ports.append(port)

        # cond
        cond = self.tok(
            cond,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.config.device)

        # Final batch structure (compatible with VLKEB)
        visual_edit = {
            "edit_inner": v_edit_inner,
            "edit_outer": v_edit_outer,
            "edit_outer_image": v_edit_outer_image,
            "loc": v_loc,
            "loc_image": v_loc_image,
        }

        textual_edit = {
            "edit_inner": t_edit_inner,
            "edit_outer": t_edit_outer,
            "loc": t_loc,
            "port": None,
        }

        batch_dict = {
            "visual_edit": visual_edit,
            "textual_edit": textual_edit,
            "cond": cond,
            "port": edit_ports
        }

        return dict_to(batch_dict, self.config.device)

class CompositionalDataset_RAG_50(CompositionalCaptionDataset):
    """
    Compositional Dataset with RAG simulation (50% accuracy).
    Used for Knowledge Connector (KC) training (Stage 2) for MiniGPT4.
    Inherits from CompositionalCaptionDataset, only overrides collate_fn for RAG simulation.
    """
    
    def collate_fn(self, batch):
        '''
        Visual Edit Dataset with RAG simulation (50% accuracy)
        '''
        src = [b['prompt'] for b in batch]
        trg = [b['target'] for b in batch]
        cond = [b['cond'] for b in batch]
        rephrase = [b['rephrase_prompt'] for b in batch]
        image = [b['image'] for b in batch] if "owl-2" not in self.config.model_name else [b['image'] for b in batch][0]
        image_rephrase = [b['image_rephrase'] for b in batch] if "owl-2" not in self.config.model_name else [b['image_rephrase'] for b in batch][0]
        loc_q = [b["locality_prompt"] for b in batch]
        loc_a = [b["locality_ground_truth"] for b in batch]
        m_loc_image = [b['multimodal_locality_image'] for b in batch] if "owl-2" not in self.config.model_name else [b['multimodal_locality_image'] for b in batch][0]
        m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
        m_loc_a = [b['multimodal_locality_ground_truth'] for b in batch]

        # edit_inner (Visual)
        v_edit_inner = {}
        v_edit_inner['image'] = torch.stack(image, dim=0)
        v_edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        v_edit_inner['labels'] = trg
        v_edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        v_edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # edit_outer (Visual)
        v_edit_outer = {}
        v_edit_outer['image'] = torch.stack(image, dim=0)
        v_edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(rephrase, trg)]
        v_edit_outer['labels'] = trg
        v_edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in rephrase]
        v_edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # edit_outer_image (Visual)
        v_edit_outer_image = {}
        v_edit_outer_image['image'] = torch.stack(image_rephrase, dim=0)
        v_edit_outer_image['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        v_edit_outer_image['labels'] = trg
        v_edit_outer_image['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        v_edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # loc (Visual)
        v_loc = {}
        v_loc['image'] = None
        v_loc['text_input'] = [" ".join([q, a]) for q, a in zip(loc_q, loc_a)]
        v_loc['labels'] = loc_a
        v_loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
        v_loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # loc_image (Visual)
        v_loc_image = {}
        v_loc_image['image'] = torch.stack(m_loc_image, dim=0)
        v_loc_image['text_input'] = [" ".join([q, a]) for q, a in zip(m_loc_q, m_loc_a)]
        v_loc_image['labels'] = m_loc_a
        v_loc_image['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in m_loc_q]
        v_loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # Textual Edit Data
        t_src = [b['textual_edit']['prompt'] for b in batch]
        t_trg = [b['textual_edit']['target'] for b in batch]
        t_loc_q = [b.get("textual_edit", {}).get("locality_prompt", "") for b in batch]
        t_loc_a = [b.get("textual_edit", {}).get("locality_ground_truth", "") for b in batch]

        # edit_inner (Textual)
        t_edit_inner = {}
        t_edit_inner['image'] = None
        t_edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(t_src, t_trg)]
        t_edit_inner['labels'] = t_trg
        t_edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in t_src]
        t_edit_inner['labels'] = self.tok(t_trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # edit_outer (Textual)
        t_rephrase = [b.get("textual_edit", {}).get("rephrase_prompt", "") for b in batch]
        t_edit_outer = {}
        t_edit_outer['image'] = None
        t_edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(t_rephrase, t_trg)] if t_rephrase else []
        t_edit_outer['labels'] = t_trg
        t_edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in t_rephrase] if t_rephrase else []
        t_edit_outer['labels'] = self.tok(t_trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # loc (Textual)
        t_loc = {}
        t_loc['image'] = None
        t_loc['text_input'] = [" ".join([q, a]) for q, a in zip(t_loc_q, t_loc_a)]
        t_loc['labels'] = t_loc_a
        t_loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in t_loc_q]
        t_loc['labels'] = self.tok(t_loc_a, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # Compositional Portability with RAG simulation (50% accuracy = 70% * 70%)
        prompt_template = (
            "Visual Editing Knowledge: {visual_info}\n"
            "Text Editing Knowledge: {textual_info}\n"
            "--------------------------------\n"
            "Question: {question}\n"
            "Answer: "
        )
        edit_ports = None
        if 'portability_prompt' in batch[0].keys():
            edit_ports = []
            for port_q, port_a in zip(batch[0]['portability_prompt'], batch[0]['portability_ground_truth']):
                port = {}
                
                # RAG simulation: 70% accuracy for visual edit data
                if random.random() < 0.70:
                    visual_src = batch[0]['prompt']
                    visual_trg = batch[0]['target']
                else:
                    rand_idx = random.choice(range(len(self._data)))
                    random_item = self._data[rand_idx]
                    visual_src = random_item['prompt']
                    visual_trg = random_item['target']

                # RAG simulation: 70% accuracy for textual edit data
                if random.random() < 0.70:
                    t_s = batch[0].get('textual_edit', {}).get('prompt', '')
                    t_t = batch[0].get('textual_edit', {}).get('target', '')
                else:
                    rand_idx = random.choice(range(len(self._data)))
                    random_item = self._data[rand_idx]
                    t_s = random_item.get('textual_edit', {}).get('prompt', '')
                    t_t = random_item.get('textual_edit', {}).get('target', '')

                visual_info = f"{visual_src} -> {visual_trg}"
                textual_info = f"{t_s} -> {t_t}"
                question = port_q

                prompt = prompt_template.format(
                    visual_info=visual_info,
                    textual_info=textual_info,
                    question=question
                )

                port['text_input'] = [" ".join([prompt, port_a])]
                port['labels'] = self.tok([port_a], add_special_tokens=False, return_tensors="pt")["input_ids"]
                port['prompts_len'] = [len(self.tok.encode(prompt, add_special_tokens=False))]
                port['image'] = torch.stack(image, dim=0)

                edit_ports.append(port)

        # cond
        cond = self.tok(
            cond,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.config.device)

        # Final batch structure (compatible with VLKEB)
        visual_edit = {
            "edit_inner": v_edit_inner,
            "edit_outer": v_edit_outer,
            "edit_outer_image": v_edit_outer_image,
            "loc": v_loc,
            "loc_image": v_loc_image,
        }

        textual_edit = {
            "edit_inner": t_edit_inner,
            "edit_outer": t_edit_outer,
            "loc": t_loc,
            "port": None,
        }

        batch_dict = {
            "visual_edit": visual_edit,
            "textual_edit": textual_edit,
            "cond": cond,
            "port": edit_ports
        }

        return dict_to(batch_dict, self.config.device)

class CompositionalDataset(BaseDataset):
    """
    Compositional Dataset from VLKEB project.
    Used for Baselines (FT, LoRA) without Query Decomposition.
    Uses CCKEB_train/eval.json (clean version without query decomposition).
    """
    def __init__(self, data_dir: str, size:  typing.Optional[int] = None, config=None, no_image=False, hop=None, *args, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # get tokenizer and vis_processor
        if config.model_class == "Blip2OPT":
            vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
        elif config.model_class == "LLaVA":
            vis_processor = transformers.CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        else:
            raise NotImplementedError("unknown model class")
        # Load Tokenizer
        if (config is not None and hasattr(config, 'tokenizer_name')):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.name
            )
            tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                tok_name, trust_remote_code=True
            )            
            if tokenizer.pad_token == None or tokenizer.pad_token == '':
                tokenizer.pad_token = tokenizer.eos_token  
                
        vis_root = config.coco_image
        rephrase_root = config.rephrase_image
        super().__init__(vis_processor, vis_root, rephrase_root, [data_dir])

        self.config = config
        self.tok = tokenizer
        self.max_length = 32

        self.prompt = "Question: {} Short answer: "

        data = []
        if size is not None:
            self.annotation = self.annotation[:size]
        if hop:
            self.hop = hop
            assert int(hop) in [1, 2, 3, 4], "hop should be 1, 2, 3, or 4"
            port_types = ['', 'comp', '2-hop', '3-hop', '4-hop']  # 'comp' for compositional
            port_type = port_types[int(hop)]
        else:
            self.hop = 1
            port_type = 'comp'  # default to 'comp'
            
        for record in tqdm(self.annotation, ncols=120, desc='Loading Data'):
            
            if record['alt'] == "":
                continue
            if hop and 'port_new' not in record.keys():
                continue
            
            ## Visual Edit Data ##
            item = {
                'prompt': record['src'],
                'pred': record['pred'],
                'target': record['alt'],
                'rephrase_prompt': record['rephrase'],
                'image': os.path.join(self.vis_root, record["image"]),
                'image_rephrase': os.path.join(self.rephrase_root, record["image_rephrase"]),
                'cond': "{} >> {} || {}".format(record['pred'], record['alt'], record['src']),
                'locality_prompt': record['loc'],
                'locality_ground_truth': record['loc_ans'],
                'multimodal_locality_image': os.path.join(self.vis_root, record['m_loc']),
                'multimodal_locality_prompt': record['m_loc_q'],
                'multimodal_locality_ground_truth': record['m_loc_a']
            }

            ##  Textual Edit Data ##
            if "textual_edit" in record:
                textual_pred = " ".join(record["textual_edit"]['pred']) if isinstance(record["textual_edit"]['pred'], list) else record["textual_edit"]['pred']
                textual_alt = " ".join(record["textual_edit"]['alt']) if isinstance(record["textual_edit"]['alt'], list) else record["textual_edit"]['alt']

                item["textual_edit"] = {
                    "prompt": record["textual_edit"]["src"],
                    "pred": textual_pred,
                    "target": textual_alt, 
                    "rephrase_prompt": record["textual_edit"]["rephrase"],
                    'cond': "{} >> {} || {}".format(
                        textual_pred,
                        textual_alt,
                        record["textual_edit"]['src']
                        ),
                    'locality_prompt': record["textual_edit"]['loc'],
                    'locality_ground_truth': record["textual_edit"]['loc_ans']
                }
                
            # Compositional portability
            if 'port_new' in record.keys():
                item['portability_prompt'] = []
                item['portability_ground_truth'] = []
                find_hop = False
                for ports in record['port_new']:
                    if ports['port_type'] == port_type:
                        find_hop = True
                        port_q = ports['Q&A']['Question']
                        port_a = textual_alt  # e': after textual edit
                        item['portability_prompt'].append(port_q)
                        item['portability_ground_truth'].append(port_a)
                        break
                
                if not find_hop:
                    continue

            data.append(item)
     
        self._data = data
        self.no_image = no_image

    def __getitem__(self, index):
        if self.no_image:
            return self._data[index]

        data = deepcopy(self._data[index])        
        # load image
        image_path = data['image']
        rephrase_image_path = data['image_rephrase']
        locality_image_path = data['multimodal_locality_image']
        
        image = Image.open(image_path).convert("RGB")
        rephrase_image = Image.open(rephrase_image_path).convert("RGB")
        locality_image = Image.open(locality_image_path).convert("RGB")
        
        if self.config.model_class == "Blip2OPT":
            image = self.vis_processor(image)
            rephrase_image = self.vis_processor(rephrase_image)
            locality_image = self.vis_processor(locality_image)
        elif self.config.model_class == "LLaVA":
            image = self.vis_processor(image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            rephrase_image = self.vis_processor(rephrase_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            locality_image = self.vis_processor(locality_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
        else:
            raise NotImplementedError

        data['image'] = image
        data['image_rephrase'] = rephrase_image
        data['multimodal_locality_image'] = locality_image

        return data
    
    def __len__(self):
        return len(self._data)

    def collate_fn(self, batch):
        # -----------------------------------------------------
        # 1) Visual Edit (top-level keys for Trainer compatibility)
        # -----------------------------------------------------
        src = [b['prompt'] for b in batch]
        trg = [b['target'] for b in batch]
        rephrase = [b['rephrase_prompt'] for b in batch]

        image = [b['image'] for b in batch]
        image_rephrase = [b['image_rephrase'] for b in batch]

        loc_q = [b["locality_prompt"] for b in batch]
        loc_a = [b["locality_ground_truth"] for b in batch]

        m_loc_image = [b['multimodal_locality_image'] for b in batch]
        m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
        m_loc_a = [b['multimodal_locality_ground_truth'] for b in batch]

        # edit_inner (Visual)
        edit_inner = {}
        edit_inner['image'] = torch.stack(image, dim=0)
        edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        edit_inner['labels'] = trg
        edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # edit_outer (Visual)
        edit_outer = {}
        edit_outer['image'] = torch.stack(image, dim=0)
        edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(rephrase, trg)]
        edit_outer['labels'] = trg
        edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in rephrase]
        edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # edit_outer_image (Visual)
        edit_outer_image = {}
        edit_outer_image['image'] = torch.stack(image_rephrase, dim=0)
        edit_outer_image['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        edit_outer_image['labels'] = trg
        edit_outer_image['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # loc (Visual)
        loc = {}
        loc['image'] = None
        loc['text_input'] = [" ".join([q, a]) for q, a in zip(loc_q, loc_a)]
        loc['labels'] = loc_a
        loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
        loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt")["input_ids"]
        # For OURS
        loc['prompt'] = loc_q
        loc['answer'] = loc_a

        # loc_image (Visual)
        loc_image = {}
        loc_image['image'] = torch.stack(m_loc_image, dim=0)
        loc_image['text_input'] = [" ".join([q, a]) for q, a in zip(m_loc_q, m_loc_a)]
        loc_image['labels'] = m_loc_a
        loc_image['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in m_loc_q]
        loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt")["input_ids"]
        # For OURS
        loc_image['prompt'] = m_loc_q
        loc_image['answer'] = m_loc_a

        # -----------------------------------------------------
        # 2) Textual Edit
        # -----------------------------------------------------
        textual_edit = None
        if "textual_edit" in batch[0].keys():
            t_src = [b['textual_edit']['prompt'] for b in batch]
            t_trg = [b['textual_edit']['target'] for b in batch]
            t_loc_q = [b.get("textual_edit", {}).get("locality_prompt", "") for b in batch]
            t_loc_a = [b.get("textual_edit", {}).get("locality_ground_truth", "") for b in batch]
            t_rephrase = [b.get("textual_edit", {}).get("rephrase_prompt", "") for b in batch]
            t_cond = [b['textual_edit']['cond'] for b in batch]

            # edit_inner (Textual)
            t_edit_inner = {}
            t_edit_inner['image'] = None
            t_edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(t_src, t_trg)]
            t_edit_inner['labels'] = t_trg
            t_edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in t_src]
            t_edit_inner['labels'] = self.tok(t_trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

            # edit_outer (Textual)
            t_edit_outer = {}
            t_edit_outer['image'] = None
            t_edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(t_rephrase, t_trg)] if t_rephrase else []
            t_edit_outer['labels'] = t_trg
            t_edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in t_rephrase] if t_rephrase else []
            if t_rephrase:
                t_edit_outer['labels'] = self.tok(t_trg, add_special_tokens=False, return_tensors="pt")["input_ids"]
            else:
                t_edit_outer['labels'] = None

            # loc (Textual)
            t_loc = {}
            if t_loc_q and t_loc_a:
                t_loc['image'] = None
                t_loc['text_input'] = [" ".join([q, a]) for q, a in zip(t_loc_q, t_loc_a)]
                t_loc['labels'] = t_loc_a
                t_loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in t_loc_q]
                t_loc['labels'] = self.tok(t_loc_a, add_special_tokens=False, return_tensors="pt")["input_ids"]
                # For OURS
                t_loc['prompt'] = t_loc_q
                t_loc['answer'] = t_loc_a
            else:
                t_loc['image'] = None
                t_loc['text_input'] = []
                t_loc['labels'] = []
                t_loc['prompts_len'] = []

            # cond (Textual)
            t_cond = self.tok(
                t_cond,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).to(self.config.device)

            textual_edit = {
                "edit_inner": t_edit_inner,
                "edit_outer": t_edit_outer,
                "loc": t_loc,
                "cond": t_cond,
                "port": None,
            }

        # -----------------------------------------------------
        # 3) Compositional Portability
        # -----------------------------------------------------
        edit_ports = None
        if 'portability_prompt' in batch[0].keys():
            edit_ports = []
            for port_q, port_a in zip(batch[0]['portability_prompt'], batch[0]['portability_ground_truth']):
                port = {}
                port['image'] = torch.stack(image, dim=0)
                port['text_input'] = [' '.join([port_q, port_a])]
                port['labels'] = [port_a]
                port['prompts_len'] = [len(self.tok.encode(port_q, add_special_tokens=False))]
                port['labels'] = self.tok([port_a], add_special_tokens=False, return_tensors="pt")["input_ids"]
                edit_ports.append(port)

        # -----------------------------------------------------
        # 4) cond (Visual)
        # -----------------------------------------------------
        cond = [b['cond'] for b in batch]
        cond = self.tok(
            cond,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.config.device)

        # -----------------------------------------------------
        # 5) Final Batch (Trainer-compatible structure)
        # -----------------------------------------------------
        visual_edit = {
            "edit_inner": edit_inner,
            "edit_outer": edit_outer,
            "edit_outer_image": edit_outer_image,
            "loc": loc,
            "loc_image": loc_image,
        }

        batch_dict = {
            "visual_edit": visual_edit,
            "cond": cond,
            "port": edit_ports
        }
        
        if textual_edit is not None:
            batch_dict["textual_edit"] = textual_edit

        return dict_to(batch_dict, self.config.device)
