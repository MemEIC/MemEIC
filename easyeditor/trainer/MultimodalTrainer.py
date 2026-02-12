from .BaseTrainer import *
import json
import logging
import os
import shutil
import tempfile
import time

import torch
from .losses import kl_loc_loss
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from .utils import (
    EarlyStopper,
    RunningStatAverager,
    _logits,
    formatted_timestamp,
    safe_backward,
    time_delta_seconds,
)
from .algs.utils import multimodal_tokenize
from tqdm import tqdm
from transformers import AutoTokenizer

LOG = logging.getLogger(__name__)


class MultimodalTrainer(BaseTrainer):
    def __init__(self, config, train_set: Dataset, val_set: Dataset):
        super().__init__(config, train_set, val_set)

        if hasattr(self.model, "edit_lrs") and not self.config.eval_only:
            self.lr_opt = self.OptimizerClass([self.model.edit_lrs], config.lr_lr)
            if self.archive is not None:
                self.lr_opt.load_state_dict(self.archive["lr_opt"])
        else:
            self.lr_opt = None

        if hasattr(self.config, "ft"):
            if getattr(self.config.ft, "use_locality", False):
                batch = next(self.edit_gen)
                self.model.loc_ids = batch["loc"]["input_ids"]
                self.model.loc_masks = batch["loc"]["attention_mask"]

    def edit_step(self, batch, training: bool):
        self.model.train(training)
        self.original_model.train(training)

        # Do the edit
        start = time.time()
        edited_model, model_info = self.model.edit(batch["edit_inner"], batch["cond"])
        edit_time = time.time() - start

        l_total, l_edit, l_loc, l_base = 0, 0, 0, 0
        info_dict = {}

        ################ portability #################
        if batch['port'] is not None:
            port_acc = 0
            assert len(batch['port']) == 1, "batch['port'] should have only one element"
            for port in batch['port']:
                with torch.no_grad():
                    port_outputs = edited_model(port)
                    port_labels = port["labels"]
                    if not isinstance(port_outputs, torch.Tensor):
                        port_logits = port_outputs.logits
                    else:
                        port_logits = port_outputs
                    if port_logits.shape[1] > port_labels.shape[1]:
                        port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                    else:
                        port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                    port_acc += port_dict["acc"].item()
                    info_dict['grad/port_pred_ids'] = port_dict['pred_ids']
                    info_dict['grad/port_targ_ids'] = port_dict['targ_ids']
            port_acc /= len(batch['port'])
            info_dict['port/acc'] = port_acc
        ################ portability #################
        
        info_dict = {**info_dict, **model_info}

        return l_total, l_edit, l_loc, l_base, info_dict

    def train_step(self, batch):
        l_total, l_edit, l_loc, l_base, info_dict = self.edit_step(
            batch, training=True
        )

        if self.global_iter > 0 and self.global_iter % self.config.accumulate_bs == 0:
            grad = torch.nn.utils.clip_grad_norm_(
                self.model.outer_parameters(),
                self.config.grad_clip,
                error_if_nonfinite=True,
            )
            info_dict['grad'] = grad.item()

            self.opt.step()
            self.opt.zero_grad()

            if self.lr_opt is not None:
                self.lr_opt.step()
                self.lr_opt.zero_grad()

                for lr_idx, lr in enumerate(self.model.edit_lrs):
                    info_dict[f'lr/lr{lr_idx}'] = lr.item()

        return info_dict

    def _inline_validation_log(self, step, stats, start_time, steps):
        elapsed = (time.time() - start_time) / (step + 1)
        prog = f"{step+1}/{steps}".ljust(20)
        #################################################################
        # inner_acc = f"{stats['inner/acc_val']:<12.5f}"
        # outer_acc = f"{stats['edit/acc_val']:<12.5f}"
        # image_acc = f"{stats['image_rephrase/acc_val']:<12.5f}"
        # loc_acc = f"{stats['loc/acc_val']:<12.5f}"
        # loc_image_acc = f"{stats['image_loc/acc_val']:<12.5f}"

        # LOG.info(
        #   f"Step {prog} outer_acc: {outer_acc} image_acc: {image_acc} inner_acc: {inner_acc} it_time: {elapsed:.4f} loc_acc: {loc_acc}, image_loc: {loc_image_acc}"
        # )
        #################################################################
        if 'port/acc_val' in stats:
            LOG.info(f"step {prog} port_acc: {stats['port/acc_val']:<12.5f} it_time: {elapsed:.4f}")
       
        if 'knowledge/acc_val' in stats:
            LOG.info(f"step {prog} knowledge_acc: {stats['knowledge/acc_val']:<12.5f} it_time: {elapsed:.4f}")

    def validate(self, steps=None, log: bool = False, result_name: str = None):
        if steps is None or steps > len(self.val_set):
            steps = len(self.val_set)

        if log:
            LOG.info(f"Beginning evaluation for {steps} steps...")
        averager = RunningStatAverager("val")

        start_time = time.time()
        if result_name is not None:
            port_result = []
        for val_step, batch in tqdm(enumerate(self.val_loader), total=steps, desc="Validation", ncols=100):
            if val_step >= steps:
                break
            
            if (log and (val_step) % self.config.log_interval == 0):
                self._inline_validation_log(
                    val_step, averager.average(), start_time, steps
                )
           
            if batch['port'] is None:
                continue
            _, _, _, _, info_dict = self.edit_step(batch, training=False)
            averager.add(info_dict) 

            # append write to txt file info_dict['port/acc']
            if result_name is not None:
                edit_inputs = batch['edit_inner']['text_input']
                port_inputs = batch['port'][0]['text_input']
                port_acc = info_dict['port/acc']
                port_pred_ids = info_dict['grad/port_pred_ids'].cpu().numpy()
                port_targ_ids = info_dict['grad/port_targ_ids'].cpu().numpy()
                # with open(f'results/results_multihop/{result_name}_port_hop{self.val_set.hop}.txt', 'a') as f:
                #     f.write(f'{edit_inputs}\n{port_inputs}\n{port_acc}\npred: {port_pred_ids}\ntarget: {port_targ_ids}\n\n')
                port_result.append({
                    'edit_input': edit_inputs,
                    'port_input': port_inputs,
                    'port_acc': port_acc,
                    'port_pred_ids': port_pred_ids.tolist(),
                    'port_targ_ids': port_targ_ids.tolist()
                })
        
        if result_name is not None:
            with open(f'results/results_multihop/{result_name}_port_hop{self.val_set.hop}.json', 'w') as f:
                json.dump(port_result, f, indent=2)

        if log:
            self._inline_validation_log(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        return stats
    
    def knowledge_qa(self, batch, training: bool):
        self.model.train(training)
        self.original_model.train(training)

        l_total, l_edit, l_loc, l_base = 0, 0, 0, 0
        info_dict = {}

        assert batch['port'] is not None, "portability edit must be provided"
        assert len(batch['port']) == 1, "batch['port'] should have only one element"

        knowledge = batch['port'][0]
        with torch.no_grad():
            knowledge_outputs = self.model(knowledge)
            knowledge_labels = knowledge["labels"]
            if not isinstance(knowledge_outputs, torch.Tensor):
                knowledge_logits = knowledge_outputs.logits
            else:
                knowledge_logits = knowledge_outputs
            if knowledge_logits.shape[1] > knowledge_labels.shape[1]:
                knowledge_dict = self.model.edit_loss_fn(self.config, knowledge_logits, knowledge_labels)
            else:
                knowledge_dict = self.model.edit_loss_fn(self.config, knowledge_logits, knowledge_labels[:, -knowledge_logits.shape[1]-1:])
            knowledge_acc = knowledge_dict["acc"].item()
        info_dict['knowledge/acc'] = knowledge_acc
        
        info_dict = {**info_dict, **{}}

        return l_total, l_edit, l_loc, l_base, info_dict
    
    def test_knowledge(self, steps=None, log: bool = False):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.eval()

        if steps is None or steps > len(self.val_set):
            steps = len(self.val_set)

        if log:
            LOG.info(f"Beginning evaluation for {steps} steps...")
        averager = RunningStatAverager("val")

        start_time = time.time()
        for val_step, batch in tqdm(enumerate(self.val_loader), total=steps, desc="Validation", ncols=100):
            if val_step >= steps:
                break
            
            if (log and (val_step) % self.config.log_interval == 0):
                self._inline_validation_log(
                    val_step, averager.average(), start_time, steps
                )

            _, _, _, _, info_dict = self.knowledge_qa(batch, training=False)
            averager.add(info_dict) 

        if log:
            self._inline_validation_log(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps


        results_path = f"results/results_base_port/{cur_time}_{self.config.model_name}_port{self.val_set.hop}_questiontest.json"

        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats

    
    def _inline_seq_log(self, step, stats, start_time, steps, comp):
        elapsed = (time.time() - start_time) / (step + 1)
        prog = f"{step+1}/{steps}".ljust(20)
        inner_acc = f"{stats['inner/acc_val']:<12.5f}"
        outer_acc = f"{stats['edit/acc_val']:<12.5f}"
        image_acc = f"{stats['image_rephrase/acc_val']:<12.5f}"
        loc_acc = f"{stats['loc/acc_val']:<12.5f}"
        loc_image_acc = f"{stats['image_loc/acc_val']:<12.5f}"
        if comp:
            text_inner_acc = f"{stats['text_inner/acc_val']:<12.5f}"
            text_outer_acc = f"{stats['text_edit/acc_val']:<12.5f}"
            text_loc_acc = f"{stats['text_loc/acc_val']:<12.5f}"
        port_acc = f"{stats['port/acc_val']:<12.5f}"
        if not comp:
            LOG.info(
            f"Step {prog} outer_acc: {outer_acc} image_acc: {image_acc} inner_acc: {inner_acc} it_time: {elapsed:.4f} loc_acc: {loc_acc}, image_loc: {loc_image_acc}, port_acc: {port_acc}"
            )
        else:
            LOG.info(
            f"Step {prog} outer_acc: {outer_acc} image_acc: {image_acc} inner_acc: {inner_acc} it_time: {elapsed:.4f} loc_acc: {loc_acc}, image_loc: {loc_image_acc}, text_inner_acc: {text_inner_acc} text_outer_acc: {text_outer_acc} text_loc_acc: {text_loc_acc} port_acc: {port_acc}"
            )

    def _inline_seq_log_CompositionalEdit(self, step, stats, start_time, steps):
        elapsed = (time.time() - start_time) / (step + 1)
        prog = f"{step+1}/{steps}".ljust(20)
        # visual
        v_inner_acc = f"{stats['vis/inner/acc_val']:<12.5f}"
        v_outer_acc = f"{stats['vis/edit/acc_val']:<12.5f}"
        v_image_acc = f"{stats['vis/image_rephrase/acc_val']:<12.5f}"
        v_loc_acc = f"{stats['vis/loc/acc_val']:<12.5f}"
        v_loc_image_acc = f"{stats['vis/image_loc/acc_val']:<12.5f}"

        # textual
        t_inner_acc = f"{stats['text/inner/acc_val']:<12.5f}"
        t_outer_acc = f"{stats['text/edit/acc_val']:<12.5f}"
        t_loc_acc = f"{stats['text/loc/acc_val']:<12.5f}"
        
        # compositional
        port_acc = f"{stats['port/acc_val']:<12.5f}"
        if hasattr(self.config, 'for_eval') and self.config.for_eval:
            port_ratio = f"{stats['port/ratio_val']:<12.5f}"
            LOG.info(
                f"Step {prog} | "
                f"[Visual Edit] - inner_acc: {v_inner_acc} outer_acc: {v_outer_acc} img_acc: {v_image_acc}| "
                f"loc_acc: {v_loc_acc} img_loc_acc: {v_loc_image_acc}                                     | "
                f"[Textual Edit] - inner_acc: {t_inner_acc} outer_acc: {t_outer_acc} loc_acc: {t_loc_acc} | "
                f"[Compositional Edit] - Port_acc: {port_acc} Port_ratio: {port_ratio}                    | "                       
                f"it_time: {elapsed:.4f}s"
            )
        else:
            LOG.info(
                f"Step {prog} | "
                f"[Visual Edit] - inner_acc: {v_inner_acc} outer_acc: {v_outer_acc} img_acc: {v_image_acc}| "
                f"loc_acc: {v_loc_acc} img_loc_acc: {v_loc_image_acc}                                     | "
                f"[Textual Edit] - inner_acc: {t_inner_acc} outer_acc: {t_outer_acc} loc_acc: {t_loc_acc} | "
                f"[Compositional Edit] - Port_acc: {port_acc}                                             | " 
                f"it_time: {elapsed:.4f}s"
            )


    def test_sequencial_step(self, batch, edited_model, base_logits, base_image_logits):
        info_dict = {}

        ##############################################################################
        with torch.no_grad():
            # inner
            inner_edit_outputs = edited_model(batch["edit_inner"])
            inner_batch_labels = batch["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase
            post_edit_outputs = edited_model(batch["edit_outer"])
            post_batch_labels = batch["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # image rephrase
            post_image_edit_outputs = edited_model(batch["edit_outer_image"])
            post_image_batch_labels = batch["edit_outer_image"]["labels"]
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs

            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
            else:
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
            del post_image_edit_outputs, post_image_edit_logits
            torch.cuda.empty_cache()

            # text loc
            post_base_outputs = edited_model(batch["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

            # image loc
            post_image_base_outputs = edited_model(batch["loc_image"])
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
            else:
                post_image_base_logits = post_image_base_outputs
            post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices
            del post_image_base_outputs, post_image_base_logits
            torch.cuda.empty_cache()

        info_dict['inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['edit/acc'] = post_edit_dict["acc"].item()
        info_dict['image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ################ portability #################
        assert len(batch['port']) == 1, "batch['port'] exist and have only one element"
        port = batch['port'][0]
        with torch.no_grad():
            port_outputs = edited_model(port)
            port_labels = port["labels"]
            if not isinstance(port_outputs, torch.Tensor):
                port_logits = port_outputs.logits
            else:
                port_logits = port_outputs
            if port_logits.shape[1] > port_labels.shape[1]:
                port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
            else:
                port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
            port_acc = port_dict["acc"].item()
        info_dict['port/acc'] = port_acc
        ################ portability #################

        return info_dict
   
    def test_sequential_step_comp(self, batch, edited_model, base_logits_vis, base_image_logits_vis, base_logits_text):
        info_dict = {}
        edited_model.eval()
        
        ##############################################################################
        with torch.no_grad():
            '''
            Visual Metric (in VLKEB dataset)
            '''

            # inner
            inner_edit_outputs = edited_model(batch["edit_inner"])
            inner_batch_labels = batch["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase
            post_edit_outputs = edited_model(batch["edit_outer"])
            post_batch_labels = batch["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # image rephrase
            post_image_edit_outputs = edited_model(batch["edit_outer_image"])
            post_image_batch_labels = batch["edit_outer_image"]["labels"]
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs

            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
            else:
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
            del post_image_edit_outputs, post_image_edit_logits
            torch.cuda.empty_cache()

            # text loc
            post_base_outputs = edited_model(batch["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_vis, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()


            # image loc
            post_image_base_outputs = edited_model(batch["loc_image"])
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
            else:
                post_image_base_logits = post_image_base_outputs
            post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits_vis, dim=-1), k=10, dim=-1).indices
            del post_image_base_outputs, post_image_base_logits
            torch.cuda.empty_cache()
        
        info_dict['inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['edit/acc'] = post_edit_dict["acc"].item()
        info_dict['image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################
        
        with torch.no_grad():
            '''
            Textual Metric
            '''

            # inner (text)
            text_inner_edit_outputs = edited_model(batch["textual_edit"]["edit_inner"])
            text_inner_batch_labels = batch["textual_edit"]["edit_inner"]["labels"]
            if not isinstance(text_inner_edit_outputs, torch.Tensor):
                text_inner_edit_logits = text_inner_edit_outputs.logits
            else:
                text_inner_edit_logits = text_inner_edit_outputs

            if text_inner_edit_logits.shape[1] > text_inner_batch_labels.shape[1]:
                text_inner_edit_dict = self.model.edit_loss_fn(self.config, text_inner_edit_logits, text_inner_batch_labels)
            else:
                text_inner_edit_dict = self.model.edit_loss_fn(self.config, text_inner_edit_logits, text_inner_batch_labels[:, -text_inner_edit_logits.shape[1]-1:])
            del text_inner_edit_outputs, text_inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase (text)
            text_post_edit_outputs = edited_model(batch["textual_edit"]["edit_outer"])
            text_post_batch_labels = batch["textual_edit"]["edit_outer"]["labels"]
            if not isinstance(text_post_edit_outputs, torch.Tensor):
                text_post_edit_logits = text_post_edit_outputs.logits
            else:
                text_post_edit_logits = text_post_edit_outputs
            
            if text_post_edit_logits.shape[1] > text_post_batch_labels.shape[1]:
                text_post_edit_dict = self.model.edit_loss_fn(self.config, text_post_edit_logits, text_post_batch_labels)
            else:
                text_post_edit_dict = self.model.edit_loss_fn(self.config, text_post_edit_logits, text_post_batch_labels[:, -text_post_edit_logits.shape[1]-1:])
            del text_post_edit_outputs, text_post_edit_logits
            torch.cuda.empty_cache()

            # text loc (text)
            text_post_base_outputs = edited_model(batch["textual_edit"]["loc"])
            if not isinstance(text_post_base_outputs, torch.Tensor):
                text_post_base_logits = text_post_base_outputs.logits
            else:
                text_post_base_logits = text_post_base_outputs
            text_post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(text_post_base_logits, dim=-1), k=1, dim=-1).indices
            text_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_text, dim=-1), k=1, dim=-1).indices
            del text_post_base_outputs, text_post_base_logits
            torch.cuda.empty_cache()

        info_dict["text_inner/acc"] = text_inner_edit_dict["acc"].item()
        info_dict["text_edit/acc"] = text_post_edit_dict["acc"].item()
        info_dict["text_loc/acc"] = sum(text_post_base_logits_softmax_top_k.view(-1) == text_base_logits_softmax_top_k.view(-1))/text_post_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ################ Compositional portability #################

        # set lora: visual & textual inference

        assert len(batch["textual_edit"]['port']) == 1, "batch['textual_edit']['port'] exist and have only one element"
        port = batch["textual_edit"]['port'][0]
        with torch.no_grad():
            port_outputs = edited_model(port)
            port_labels = port["labels"]
            if not isinstance(port_outputs, torch.Tensor):
                port_logits = port_outputs.logits
            else:
                port_logits = port_outputs
            if port_logits.shape[1] > port_labels.shape[1]:
                port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
            else:
                port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
            port_acc = port_dict["acc"].item()
            del port_outputs, port_logits
            torch.cuda.empty_cache()
        
        info_dict['port/acc'] = port_acc
        ################ Compositional portability #################

        return info_dict


    ## Baselines: LoRA & FT ##
    def test_sequencial_compositional_ft(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") # 궁금한게, 200개에 대한 batch
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## 저장할 내용
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual 둘다 저장 & 출력)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()

        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first)
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], batch["cond"], detach_history=True)

            # 2.1.2) Textual Edit(second) ★☆★☆★☆ --> __getitem__ 시 확인. collate_fn
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], batch["cond"], detach_history=True)
            ## **순서상 Textual Edit이 나중에 되고, 바로 평가가 되니 textual edit이 조금이나마 더 잘나오지 않을까 생각 ##

            # 2.2) Test with GAP
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text 모두 다 평가해야 함.
                info_dict = self.test_sequencial_compositional_step(
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit( ## ★☆★ 수정 필요 ★☆★ ##
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps) ## ★☆★ 수정 필요 ★☆★ ##
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        if 'llava' in self.config.model_name.lower():
            if os.path.basename(self.config.name) == "llava-v1.5-13b":
                print("-> llava 13B 저장중...")
                result_dir = f"results/results_sequencial/llava1.5v_13b/composition/ft"
            elif os.path.basename(self.config.name) == "llava-v1.5-7b":
                result_dir = f"results/results_sequencial/composition/ft"
                print("-> llava 7B 저장중...")
        elif 'blip2' in self.config.model_name.lower() :
            result_dir = f"results/results_sequencial/blip2/composition/ft"
        elif 'minigpt4' in self.config.model_name.lower() :
            result_dir = f"results/results_sequencial/minigpt4/composition/ft"
        
        results_path = os.path.join(result_dir, f"{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json")
        

        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats

    ## Stage2: Training Knowledge Connector 
    def test_sequencial_compositional_connector_attention_rag_70(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...")
        averager = RunningStatAverager("val")

        start_time = time.time()

        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual both store & output)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch data
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference  # self.model -> ft
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()

        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first)
            if self.config.model_name == 'llava':
                self.model.model.set_adapter("visual")
            elif self.config.model_name == 'minigpt4':
                self.model.model.llama_model.set_adapter("visual")
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], mode = "visual" , peft = True)

            # 2.1.2) Textual Edit(second) 
            if self.config.model_name == 'llava':
                self.model.model.set_adapter("textual")
            elif self.config.model_name == 'minigpt4':
                self.model.model.llama_model.set_adapter("textual")
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], mode = "textual" , peft = True)

            # 2.1.3) Compositional Edit(second) ★ 
            if val_step > 5:
                if self.config.model_name == 'llava':
                    edited_model.model.set_adapter(["textual","visual","connector"])
                elif self.config.model_name == 'minigpt4':
                    edited_model.model.llama_model.set_adapter(["textual","visual","connector"])
                edited_model, _ = edited_model.edit(batch["port"][0], connector_mode=True)


            # 2.2) Test with GAP
            if val_step >= gap_num: 
                # load previous batch, t-loc & i-loc-logit . For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text both evaluation
                info_dict = self.test_sequencial_compositional_connector_attention_rag_step(
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit(
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps) 
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps


        if 'llava' in self.config.model_name.lower():
            result_dir = f"results/results_sequencial/composition/stage2_two_lora_connect_attention_rag_70"
        elif 'blip2' in self.config.model_name.lower() :
            result_dir = f"results/results_sequencial/blip2/composition/stage2_two_lora_connect_attention_rag_70"
        elif 'minigpt4' in self.config.model_name.lower() :
            result_dir = f"results/results_sequencial/minigpt4/composition/stage2_two_lora_connect_attention_rag_70"
        
        results_path = os.path.join(result_dir, f"{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json")
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        if gap_num == 0:
            try: # Store Knowledge Connector & Mem-I 
                from peft import LoraConfig, TaskType, get_peft_model, PeftConfig, PeftModel
                connector_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=8,
                        lora_alpha=16,
                        lora_dropout=0.05,
                        target_modules=["q_proj", "k_proj"]
                    )
                
                # model branch
                if 'llava' in self.config.model_name.lower() :
                    peft_model = get_peft_model(self.model.model.base_model.model, connector_config)

                elif 'blip2' in self.config.model_name.lower() :
                    peft_model = get_peft_model(self.model.model.opt_model.model, connector_config)

                elif 'minigpt4' in self.config.model_name.lower() :
                    peft_model = get_peft_model(self.model.model.llama_model.model, connector_config)

                peft_model.delete_adapter("default")
                peft_model = peft_model.cpu()
                peft_model.save_pretrained(result_dir)
                # 저장 후 메모리 해제
                del peft_model

                torch.cuda.empty_cache()
                print("Complete Model Saving LoRA + Connector (Gap 0 with train_compositional_edit.json) ->", result_dir)
            except Exception as e:
                print(f"Store Failed LoRA + Connector: {e}")

        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats
    

        ## TEST - compositonal - two      
    def test_sequencial_compositional_connector_attention_rag_50(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True) 

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") 
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## stored data
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual data store & output)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # store batch 
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # store T-Loc inference # self.model -> ft
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # store I-Loc inference 
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # store T-Loc inference 
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()

        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first)
            if self.config.model_name == 'llava':
                self.model.model.set_adapter("visual")
            elif self.config.model_name == 'minigpt4':
                self.model.model.llama_model.set_adapter("visual")
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], mode = "visual" , peft = True)

            # 2.1.2) Textual Edit(second) 
            if self.config.model_name == 'llava':
                self.model.model.set_adapter("textual")
            elif self.config.model_name == 'minigpt4':
                self.model.model.llama_model.set_adapter("textual")
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], mode = "textual" , peft = True)

            # 2.1.3) Compositional Edit(second) 
            if val_step > 5:
                if self.config.model_name == 'llava':
                    edited_model.model.set_adapter(["textual","visual","connector"])
                elif self.config.model_name == 'minigpt4':
                    edited_model.model.llama_model.set_adapter(["textual","visual","connector"])
                edited_model, _ = edited_model.edit(batch["port"][0], connector_mode=True)


            # 2.2) Test with GAP
            if val_step >= gap_num: 
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - both vis / text 
                info_dict = self.test_sequencial_compositional_connector_attention_rag_step(
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit( 
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        if 'llava' in self.config.model_name.lower():
            if os.path.basename(self.config.name) == "llava-v1.5-13b":
                print("-> Storing llava 13B ...")
                result_dir = f"results/results_sequencial/llava1.5v_13b/composition/stage2_two_lora_connect_attention_rag_50"
            elif os.path.basename(self.config.name) == "llava-v1.5-7b":
                result_dir = f"results/results_sequencial/composition/stage2_two_lora_connect_attention_rag_50"
                print("-> Storing llava 7B ...")
            
        elif 'blip2' in self.config.model_name.lower() :
            result_dir = f"results/results_sequencial/blip2/composition/stage2_two_lora_connect_attention_rag_50"
        elif 'minigpt4' in self.config.model_name.lower() :
            result_dir = f"results/results_sequencial/minigpt4/composition/stage2_two_lora_connect_attention_rag_50"

        results_path = os.path.join(result_dir, f"{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json")
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        if gap_num == 0:
            try: # Save Knowledge Connector & Mem-I
                from peft import LoraConfig, TaskType, get_peft_model, PeftConfig, PeftModel
                connector_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=8,
                        lora_alpha=16,
                        lora_dropout=0.05,
                        target_modules=["q_proj", "k_proj"]
                    )
  
                if 'llava' in self.config.model_name.lower() :
                    peft_model = get_peft_model(self.model.model.base_model.model, connector_config)

                elif 'blip2' in self.config.model_name.lower() :
                    peft_model = get_peft_model(self.model.model.opt_model.model, connector_config)

                elif 'minigpt4' in self.config.model_name.lower() :
                    peft_model = get_peft_model(self.model.model.llama_model.model, connector_config)

                peft_model.delete_adapter("default")
                peft_model = peft_model.cpu()
                peft_model.save_pretrained(result_dir)
                
                # Pop Memory
                del peft_model
                torch.cuda.empty_cache()

                print("Complete Model Saving: LoRA + Connector (Gap 0 with train_compositional_edit.json) ->", result_dir)
            except Exception as e:
                print(f"Failed Model Saving: LoRA + Connector: {e}")

        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats
    
    ## TEST - compositonal - two      
    def test_sequencial_compositional_connector_attention_rag_step(self, batch, edited_model, base_logits_vis, base_image_logits_vis, base_logits_tex):
        info_dict = {}

        ##############################################################################
        # ----------------------------Test: Visual Edit------------------------------#
        with torch.no_grad():
            # set lora: visual inference
            if self.config.model_name == 'llava':
                edited_model.model.set_adapter("visual")
            elif self.config.model_name == 'minigpt4':
                edited_model.model.llama_model.set_adapter("visual")
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["visual_edit"]["edit_inner"])
            inner_batch_labels = batch["visual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels) # edit_loss_fn이 어딨지?
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(T-Gen)
            post_edit_outputs = edited_model(batch["visual_edit"]["edit_outer"])
            post_batch_labels = batch["visual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels) # edit_loss_fn -> vis, text 한번에 적용해도 되는가
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # image rephrase(I-Gen)
            post_image_edit_outputs = edited_model(batch["visual_edit"]["edit_outer_image"])
            post_image_batch_labels = batch["visual_edit"]["edit_outer_image"]["labels"]
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs

            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
            else:
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
            del post_image_edit_outputs, post_image_edit_logits
            torch.cuda.empty_cache()

            # text loc(T-Loc)
            post_base_outputs = edited_model(batch["visual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_vis, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

            # image loc(I-Loc)
            post_image_base_outputs = edited_model(batch["visual_edit"]["loc_image"])
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
            else:
                post_image_base_logits = post_image_base_outputs
            post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits_vis, dim=-1), k=10, dim=-1).indices
            del post_image_base_outputs, post_image_base_logits
            torch.cuda.empty_cache()

        info_dict['vis/inner/acc'] = inner_edit_dict["acc"].item() # copy안해도 되는가? -> item은 int/float이 직접 반환됨. 따라서 ㅇㅇ
        info_dict['vis/edit/acc'] = post_edit_dict["acc"].item()
        info_dict['vis/image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        # 디바이스 일치를 위해 base_logits_softmax_top_k를 post_base_logits_softmax_top_k와 같은 디바이스로 이동
        base_logits_softmax_top_k = base_logits_softmax_top_k.to(post_base_logits_softmax_top_k.device)
        info_dict["vis/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        # 디바이스 일치를 위해 base_image_logits_softmax_top_k를 post_image_base_logits_softmax_top_k와 같은 디바이스로 이동
        base_image_logits_softmax_top_k = base_image_logits_softmax_top_k.to(post_image_base_logits_softmax_top_k.device)
        info_dict["vis/image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ##############################################################################
        # ----------------------------Test: Textual Edit------------------------------#
        with torch.no_grad():
            # set lora: textual inference
            if self.config.model_name == 'llava':
                edited_model.model.set_adapter("textual")
            elif self.config.model_name == 'minigpt4':
                edited_model.model.llama_model.set_adapter("textual")
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["textual_edit"]["edit_inner"])
            inner_batch_labels = batch["textual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(Generality)
            post_edit_outputs = edited_model(batch["textual_edit"]["edit_outer"])
            post_batch_labels = batch["textual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # text loc(Locality)
            post_base_outputs = edited_model(batch["textual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_tex, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

        info_dict['text/inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['text/edit/acc'] = post_edit_dict["acc"].item()
        # 디바이스 일치를 위해 base_logits_softmax_top_k를 post_base_logits_softmax_top_k와 같은 디바이스로 이동
        base_logits_softmax_top_k = base_logits_softmax_top_k.to(post_base_logits_softmax_top_k.device)
        info_dict["text/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ################ portability #################
        # set lora: visual&textual inference
        if self.config.model_name == 'llava':
            edited_model.model.set_adapter(["textual","visual","connector"])
        elif self.config.model_name == 'minigpt4':
            edited_model.model.llama_model.set_adapter(["textual","visual","connector"])

        if batch['port'] is not None:
            assert len(batch['port']) == 1, "batch['port'] exist and have only one element"
            port = batch['port'][0]
            with torch.no_grad():
                port_outputs = edited_model(port)
                port_labels = port["labels"]
                if not isinstance(port_outputs, torch.Tensor):
                    port_logits = port_outputs.logits
                else:
                    port_logits = port_outputs
                if port_logits.shape[1] > port_labels.shape[1]:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                else:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                port_acc = port_dict["acc"].item()
                del port_outputs, port_logits
                torch.cuda.empty_cache()

            info_dict['port/acc'] = port_acc
            ################ portability #################

        return info_dict

    def test_sequencial_compositional_step(self, batch, edited_model, base_logits_vis, base_image_logits_vis, base_logits_tex):
        info_dict = {}

        ##############################################################################
        # ----------------------------Test: Visual Edit------------------------------#
        with torch.no_grad():
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["visual_edit"]["edit_inner"])
            inner_batch_labels = batch["visual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(T-Gen)
            post_edit_outputs = edited_model(batch["visual_edit"]["edit_outer"])
            post_batch_labels = batch["visual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels) # edit_loss_fn 
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # image rephrase(I-Gen)
            post_image_edit_outputs = edited_model(batch["visual_edit"]["edit_outer_image"])
            post_image_batch_labels = batch["visual_edit"]["edit_outer_image"]["labels"]
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs

            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
            else:
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
            del post_image_edit_outputs, post_image_edit_logits
            torch.cuda.empty_cache()

            # text loc(T-Loc)
            post_base_outputs = edited_model(batch["visual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_vis, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

            # image loc(I-Loc)
            post_image_base_outputs = edited_model(batch["visual_edit"]["loc_image"])
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
            else:
                post_image_base_logits = post_image_base_outputs
            post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits_vis, dim=-1), k=10, dim=-1).indices
            del post_image_base_outputs, post_image_base_logits
            torch.cuda.empty_cache()

        info_dict['vis/inner/acc'] = inner_edit_dict["acc"].item() 
        info_dict['vis/edit/acc'] = post_edit_dict["acc"].item()
        info_dict['vis/image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        
        base_logits_softmax_top_k = base_logits_softmax_top_k.to(post_base_logits_softmax_top_k.device)
        info_dict["vis/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]

        base_image_logits_softmax_top_k = base_image_logits_softmax_top_k.to(post_image_base_logits_softmax_top_k.device)
        info_dict["vis/image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ##############################################################################
        # ----------------------------Test: Textual Edit------------------------------#
        with torch.no_grad():
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["textual_edit"]["edit_inner"])
            inner_batch_labels = batch["textual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(Generality)
            post_edit_outputs = edited_model(batch["textual_edit"]["edit_outer"])
            post_batch_labels = batch["textual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # text loc(Locality)
            post_base_outputs = edited_model(batch["textual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_tex, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

        info_dict['text/inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['text/edit/acc'] = post_edit_dict["acc"].item()
        
        base_logits_softmax_top_k = base_logits_softmax_top_k.to(post_base_logits_softmax_top_k.device)
        info_dict["text/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ################ portability #################
        if batch['port'] is not None:
            assert len(batch['port']) == 1, "batch['port'] exist and have only one element"
            port = batch['port'][0]
            with torch.no_grad():
                port_outputs = edited_model(port)
                port_labels = port["labels"]
                if not isinstance(port_outputs, torch.Tensor):
                    port_logits = port_outputs.logits
                else:
                    port_logits = port_outputs
                if port_logits.shape[1] > port_labels.shape[1]:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                else:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                port_acc = port_dict["acc"].item()
            info_dict['port/acc'] = port_acc
            ################ portability #################

        return info_dict
    
    def test_sequencial_multi_gpus(self, log: bool = False, test_num=500, gap_num=0, comp=False, training=False, gpus=[0,1]):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            #LOG.info(f'Seed is {self.config.seed}')
            LOG.info(f"Beginning evaluation for {test_num} steps...")
        averager = RunningStatAverager("val")

        start_time = time.time()

        val_data_store = []
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        base_logits_store_text = []
        

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                val_data_store.append(batch)
                # Visual Edit
                with torch.no_grad():
                    base_outputs = self.model(batch["loc"])
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach().to('cuda:' + str(gpus[1])))
                        
                    base_image_outputs = self.model(batch["loc_image"])
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach().to('cuda:' + str(gpus[1])))
                
                # Textual Edit
                if comp and "textual_edit" in batch.keys():
                    with torch.no_grad():
                        base_outputs = self.model(batch["textual_edit"]["loc"])
                        if not isinstance(base_outputs, torch.Tensor):
                            base_logits = base_outputs.logits
                        else:  
                            base_logits = base_outputs
                        base_logits_store_text.append(base_logits.clone().detach().to('cuda:' + str(gpus[1])))

                pbar.update(1)
            else:
                break
        pbar.close()

        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            edited_model, _ = edited_model.edit(batch["edit_inner"], batch["cond"], detach_history=True)
            # Textual Edit
            if comp and "textual_edit" in batch.keys():
                edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], batch["textual_edit"]["cond"], detach_history=True)

            if val_step >= gap_num:
                stored_batch = val_data_store.pop(0)
                stored_base_logits_vis = base_logits_store_vis.pop(0).to('cuda:' + str(gpus[0]))
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0).to('cuda:' + str(gpus[0]))
                if comp:
                    stored_base_logits_text = base_logits_store_text.pop(0).to('cuda:' + str(gpus[0]))
                    # if self.config.alg_name == 'WISE':
                    #     info_dict = self.test_sequential_step_comp(stored_batch, self.model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_text)
                    # else:
                    info_dict = self.test_sequential_step_comp(stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_text)
                else:
                    info_dict = self.test_sequencial_step(stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis)
                averager.add(info_dict)

            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log(
                    val_step, averager.average(), start_time, steps, comp
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        if log:
            self._inline_seq_log(val_step, averager.average(), start_time, steps, comp)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        adapter_path = f"{self.config.results_dir}/{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}"
        results_path = f"{self.config.results_dir}/{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}.json"

        # Store Lora weights
        if training:
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            if gap_num == 0:
                try:
                    from peft import LoraConfig, TaskType, get_peft_model, PeftConfig, PeftModel
                    connector_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=8,
                        lora_alpha=16,
                        lora_dropout=0.05,
                        target_modules=["q_proj", "k_proj"]
                    )

                    peft_model = get_peft_model(self.model.model.base_model.model, connector_config)
                    peft_model.delete_adapter("default")
                    peft_model = peft_model.cpu()
                    peft_model.save_pretrained(adapter_path)
                    # Delete memory after saving
                    del peft_model

                    torch.cuda.empty_cache()
                    print('Saved LORA (gap0)')
                except Exception as e:
                    print(f"Failed to save LORA: {e}")

        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats
