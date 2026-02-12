import torch
import copy
import transformers
import logging
import re

from ..utils import scr, set_dropout, _logits, add_padding, add_sep
from .editable_model import EditableModel
from ..models import BertClassifier
from transformers import GPT2Tokenizer, GPT2TokenizerFast
from tqdm import tqdm, trange

LOG = logging.getLogger(__name__)

comp_prompt_template = (
"Visual Editing Knowledge: {visual_info}\n"
"Text Editing Knowledge: {textual_info}\n"
"--------------------------------\n"
"Question: {question}\n"
"Answer: "
)

base_prompt = (
    "Question: {question} Short answer: {answer}"
)

def translate_tokens(tokens, from_tok, to_tok):
    tokens = tokens.masked_fill(tokens == -100, from_tok.pad_token_id)
    text = from_tok.batch_decode(tokens, skip_special_tokens=True)
    return to_tok(text, return_tensors="pt")["input_ids"].to(tokens.device)


class OURS(EditableModel):
    def __init__(self, model, config, model_constructor, classifier=None, classifier_tok=None, cache_inputs=None, cache_labels=None, cache_questions=None,
                 cache_text_inputs=None, cache_text_labels=None, cache_text_questions=None, scale=None, cache_image_inputs=None, classifier_image=None, classifier_image_processor=None, tok=None):
        super().__init__(model, config, model_constructor)

        if classifier is None:
            if config.cross_attend and not config.cls_class.endswith("ForSequenceClassification"):
                LOG.warn(f"Switching {config.cls_class} to {config.cls_class}ForSequenceClassification for cross-attend")
                config.cls_class += "ForSequenceClassification"
            self.classifier = getattr(transformers, config.cls_class).from_pretrained(config.cls_name, cache_dir='./hugging_cache')
            if self.config.checkpoint_grad:
                LOG.info(f"Checking for checkpointing: {hasattr(self.classifier.config, 'gradient_checkpointing')}")
                self.classifier.config.gradient_checkpointing = True
            self.classifier_tok = transformers.AutoTokenizer.from_pretrained(config.cls_name, cache_dir='./hugging_cache')
            if not self.config.cross_attend and 'bert' in self.config.cls_name:
                self.classifier.pooler = None  # we don't need the classification head
            elif not self.config.cross_attend and "mpnet" not in self.config.cls_name:
                if hasattr(self.classifier, "pooler"):
                    self.classifier.pooler = None  # we don't need the classification head
            set_dropout(self.classifier, config.dropout)
        else:
            assert isinstance(classifier, torch.nn.Module), f"Classifier is a {type(classifier)}!"
            assert isinstance(classifier_tok, transformers.PreTrainedTokenizerBase), f"Classifier tok is {type(classifier_tok)}!"
            self.classifier, self.classifier_tok = classifier, classifier_tok

        if classifier_image is None:
            self.classifier_image = getattr(transformers, config.cls_image_class).from_pretrained(config.cls_image_name, cache_dir='./hugging_cache')
            if self.config.checkpoint_grad:
                LOG.info(f"Checking for checkpointing: {hasattr(self.classifier_image.config, 'gradient_checkpointing')}")
                self.classifier_image.config.gradient_checkpointing = True
            self.classifier_image_processor = getattr(transformers, config.cls_image_proc_class).from_pretrained(config.cls_image_name, cache_dir='./hugging_cache')
            # self.classifier_image.requires_grad_(True)
            set_dropout(self.classifier_image, config.dropout)
        else:
            assert isinstance(classifier_image, torch.nn.Module), f"Image Classifier is a {type(classifier_image)}!"
            assert isinstance(classifier_image_processor, transformers.models.clip.image_processing_clip.CLIPImageProcessor), f"Image Classifier tok is {type(classifier_image_processor)}!"
            self.classifier_image, self.classifier_image_processor = classifier_image, classifier_image_processor
        
        if tok is None:
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.name
            )
            if config.tokenizer_class == "QWenTokenizer":
                self.tok = transformers.AutoTokenizer.from_pretrained(config.name, trust_remote_code=True, pad_token='<|endoftext|>')
            elif config.model_name == "owl-2":
                self.tok = transformers.AutoTokenizer.from_pretrained(config.name, use_fast=False, trust_remote_code=True)
            else:
                self.tok = getattr(transformers, config.tokenizer_class).from_pretrained(
                    tok_name, trust_remote_code=True
                )            
            if self.tok.pad_token == None or self.tok.pad_token == '':
                self.tok.pad_token = self.tok.eos_token  
        else:
            assert isinstance(tok, transformers.PreTrainedTokenizerBase), f"Tokenizer is {type(tok)}!"
            self.tok = tok

        if config.model_name == 'qwen-vl':
            self.qwenvl_tok = transformers.AutoTokenizer.from_pretrained(config.name, trust_remote_code=True, pad_token='<|endoftext|>')

        if self.config.cross_attend:
            self.scale = None
        else:
            if scale is None:
                self.register_buffer("scale", torch.tensor(1.0))
            else:
                self.scale = scale
        
        if cache_inputs is None:
            self.cache_inputs = []
            self.cache_labels = []
            self.cache_questions = []
        else:
            assert isinstance(cache_inputs, list), f"Cache inputs is {cache_inputs}"
            assert isinstance(cache_labels, list), f"Cache labels is {cache_labels}"
            assert isinstance(cache_questions, list), f"Cache questions is {cache_questions}"
            self.cache_inputs = copy.deepcopy(cache_inputs)
            self.cache_labels = copy.deepcopy(cache_labels)
            self.cache_questions = copy.deepcopy(cache_questions)

        if cache_image_inputs is None or cache_image_inputs == []:
            self.cache_image_inputs = []
        else:
            assert isinstance(cache_image_inputs, torch.Tensor), f"Cache Image inputs is {cache_image_inputs}"
            self.cache_image_inputs = cache_image_inputs

        if cache_text_inputs is None:
            self.cache_text_inputs = []
            self.cache_text_labels = []
            self.cache_text_questions = []
        else:
            assert isinstance(cache_text_inputs, list), f"Cache text inputs is {cache_text_inputs}"
            assert isinstance(cache_text_labels, list), f"Cache text labels is {cache_text_labels}"
            assert isinstance(cache_text_questions, list), f"Cache text labels is {cache_text_questions}"
            self.cache_text_inputs = copy.deepcopy(cache_text_inputs)
            self.cache_text_labels = copy.deepcopy(cache_text_labels)
            self.cache_text_questions = copy.deepcopy(cache_text_questions)

        # For LORA
        if self.config.use_lora == True:
            self.model = self.model.to(torch.float32)
            self.save_weight = None

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(prefix=prefix, keep_vars=keep_vars)  # Get default state dict
        if self.config.use_lora == False:
            model_keys = self.model.state_dict(prefix=prefix, keep_vars=keep_vars).keys()  # Remove model params
            for k in model_keys:
                del state_dict[f"model.{k}"]
        state_dict["model_config"] = self.model.config  # Include model config
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        config = state_dict["model_config"]
        del state_dict["model_config"]
        if config != self.model.config:
            LOG.info("Loaded model config doesn't match current model config.")
            LOG.info(f"Loaded: {config}")
            LOG.info(f"Current: {self.model.config}")

        res = super().load_state_dict(state_dict, False)

        # We should only have missing keys for the model, and no unexpected keys
        def ok_to_miss(k):
            if self.config.model_name == "minigpt4" or self.config.model_name == "llava" or self.config.model_name == "qwen-vl" or self.config.model_name == "owl-2":
                return k.startswith("model.") or (self.config.freeze_cntr and k.startswith("replacement.")) or (k.startswith("replacement.") and ("31" not in k))
            return k.startswith("model.") or (self.config.freeze_cntr and k.startswith("replacement."))
        missing_keys = [k for k in res.missing_keys if not ok_to_miss(k)]
        assert len(missing_keys) == 0, f"Should only have missing keys for model: {missing_keys}."
        assert len(res.unexpected_keys) == 0, "Shouldn't have any unexpected keys"
        return res

    def outer_parameters(self, grouped=False):
        if self.config.freeze is not None:
            # Text Classifier
            modlist = None
            for m in self.classifier.modules():
                if isinstance(m, torch.nn.ModuleList):
                    modlist = m
                    break
            model_params = list(modlist[-self.config.freeze:].parameters())
            if self.config.only_text == False:
                # Image Classifier
                modlist = None
                for m in self.classifier_image.modules():
                    if isinstance(m, torch.nn.ModuleList):
                        modlist = m
                        break
                model_params.extend(list(modlist[-self.config.freeze:].parameters()))
        else:
            model_params = list(self.classifier.parameters())
            if self.config.only_text == False:
                model_params.extend(list(self.classifier_image.parameters()))

        if self.config.freeze is not None:
            # Text Classifier
            cls = self.classifier
            if hasattr(cls, "classifier"):
                model_params.extend(cls.classifier.parameters())
            if hasattr(cls, "pre_classifier"):
                model_params.extend(cls.pre_classifier.parameters())

        extra_params = []
        if grouped:
            return [
                dict(params=model_params, lr=self.config.lr),
                dict(params=extra_params, lr=self.config.lr_lr)
            ]
        else:
            return model_params + extra_params

    def textual_parameters(self, grouped=False):
        if self.config.freeze is not None:
            # Text Classifier
            modlist = None
            for m in self.classifier.modules():
                if isinstance(m, torch.nn.ModuleList):
                    modlist = m
                    break
            model_params = list(modlist[-self.config.freeze:].parameters())
        else:
            model_params = list(self.classifier.parameters())
            model_params.extend(list(self.classifier_image.parameters()))

        if self.config.freeze is not None:
            # Text Classifier
            cls = self.classifier
            if hasattr(cls, "classifier"):
                model_params.extend(cls.classifier.parameters())
            if hasattr(cls, "pre_classifier"):
                model_params.extend(cls.pre_classifier.parameters())

        extra_params = []
        if grouped:
            return [
                dict(params=model_params, lr=self.config.lr),
                dict(params=extra_params, lr=self.config.lr_lr)
            ]
        else:
            return model_params + extra_params

    def edit(self, batch, condition=None, detach_history=False, connector_mode=False):
        def detokenize(toks, tok):
            tokens = toks.masked_fill(toks == -100, tok.pad_token_id)
            return tok.batch_decode(tokens, skip_special_tokens=True)
        
        cache_inputs, cache_labels, cache_questions, cache_image_inputs = self.cache_inputs, self.cache_labels, self.cache_questions, self.cache_image_inputs
        cache_text_inputs, cache_text_labels, cache_text_questions = self.cache_text_inputs, self.cache_text_labels, self.cache_text_questions

        if not connector_mode and batch['image'] is not None:
            # Visual Edit
            inputs = batch["text_input"]
            labels = batch["labels"]
            questions = batch["prompt"]
            image_inputs = batch["image_cls"]

            if isinstance(labels, torch.Tensor):
                if self.config.model_name == "qwen-vl":
                    labels = detokenize(labels, self.qwenvl_tok)
                    labels = [s.strip() for s in labels]
                else:
                    labels = detokenize(labels, self.tok)

            # Preparing outputs
            with torch.no_grad():
                if type(image_inputs) is list or image_inputs.ndim == 5:
                    concat_images = torch.cat([image for image in image_inputs], dim=0)
                    cls_image_features = self.encode_images(concat_images)
                    # for Projector_layer
                    # split_sizes = [image.shape[0] for image in cls_ctxs]
                    # cls_image_features = torch.split(cls_image_features, split_sizes, dim=0)
                    # cls_image_features = [x.flatten(0, 1).to(self.config.device) for x in cls_image_features]
                else:
                    cls_image_features = self.encode_images(image_inputs)
                image_inputs = cls_image_features.last_hidden_state[:, 0].unsqueeze(1).to('cpu')

            cache_inputs = self.cache_inputs + inputs
            cache_labels = self.cache_labels + labels
            cache_questions = self.cache_questions + questions
            if self.cache_image_inputs == []:
                cache_image_inputs = image_inputs
            else:
                cache_image_inputs = torch.cat([self.cache_image_inputs, image_inputs], dim=0)
        elif not connector_mode:
            # Textual Edit
            text_inputs = batch['text_input']
            text_labels = batch['labels']
            text_questions = batch['prompt']

            if isinstance(text_labels, torch.Tensor):
                if self.config.model_name == "qwen-vl":
                    text_labels = detokenize(text_labels, self.qwenvl_tok)
                    text_labels = [s.strip() for s in text_labels]
                else:
                    text_labels = detokenize(text_labels, self.tok)
            
            cache_text_inputs = self.cache_text_inputs + text_inputs
            cache_text_labels = self.cache_text_labels + text_labels
            cache_text_questions = self.cache_text_questions + text_questions

        if self.config.use_lora: # finetune using LORA

            self.cache_inputs = cache_inputs
            self.cache_labels = cache_labels
            self.cache_image_inputs = cache_image_inputs
            self.cache_questions = cache_questions

            # if self.save_weight is not None:
            #     self.model.load_state_dict(self.save_weight, strict=False)
            self.model.train(True)

            if not self.config.inner_params:  # inner_params == []
                if connector_mode:
                    # Connector
                    weights = {
                        n: p
                        for n, p in self.model.named_parameters()
                        if ("connector" in n)
                    }
                elif batch['image'] is not None:
                    # Visual Edit
                    weights = {
                        n: p
                        for n, p in self.model.named_parameters()
                        if 'lora' in n and 'visual' in n
                    }
                else:
                    # Textual Edit
                    weights = {
                        n: p
                        for n, p in self.model.named_parameters()
                        if 'lora' in n and 'textual' in n
                    }
            else:
                names = set([n for n, p in self.nmodel.named_parameters()])
                pset = set(self.config.inner_params)
                for p in pset:
                    assert p in names, f"inner param {p} not in model"

                weights = {
                    n: p
                    for n, p in self.model.named_parameters()
                    if n in pset
                }
            # Save old weights for future restoration
            # self.save_weight = {k: v.detach().clone() for k, v in weights.items()}

            if connector_mode:
                edit_lr = self.config.lora_edit_lr/5
            else:
                edit_lr = self.config.lora_edit_lr

            opt = torch.optim.AdamW(
                [v for _, v in weights.items()],
                lr=edit_lr
            )
            for name, w in self.model.named_parameters():
                w.requires_grad = name in weights

            batch = batch,

            if connector_mode:
                image_level, text_level, _ = self.divide_image_text_level(*batch)
                cls_img_sims, cls_text_sims = None, None

                if image_level != None: # Image-Level Memory Matching
                    cls_img_sims, cls_img_idxs, cls_img_logits = self.run_image_classifier(*batch)
                    if cls_img_sims is not None:
                        rep_img_cls_inputs, rep_img_cls_texts, img_selected_ctxs, img_selected_qs, img_selected_labels = self.build_rep_input_tokens(batch[0], cls_img_idxs, edit_type="visual")

                if text_level != None: # Text-Level Memory Matching
                    if cls_img_sims is not None:
                        batch[0]["text_level"] = re.sub(r'\[.*?\]', img_selected_labels[0], text_level) # self.build_prompts(img_selected_ctxs, text_level) 
                    else:
                        batch[0]["text_level"] = text_level.replace('[', '').replace(']', '')
                    cls_text_sims, cls_text_idxs, cls_text_logits = self.run_text_classifier(*batch)
                    if cls_text_sims is not None:
                        rep_text_cls_inputs, rep_text_cls_texts, text_selected_ctxs, text_selected_qs, _ = self.build_rep_input_tokens(batch[0], cls_text_idxs, edit_type="textual")

                if cls_img_sims is not None and cls_text_sims is None: # Visual 
                    cls_sims = cls_img_sims
                    cls_logits = cls_img_logits 
                    rep_cls_texts = self.build_prompts(img_selected_ctxs, batch[0]["prompt"][0])
                    if self.config.use_lora:
                        rep_cls_texts = self.build_prompts(img_selected_ctxs, batch[0]["prompt"][0], use_lora=True, mode='vis')
                elif cls_img_sims is None and cls_text_sims is not None: # Textual
                    cls_sims = cls_text_sims
                    cls_logits = cls_text_logits
                    rep_cls_texts = self.build_prompts(text_selected_ctxs, batch[0]["prompt"][0])
                    if self.config.use_lora:
                        rep_cls_texts = self.build_prompts(text_selected_ctxs, batch[0]["prompt"][0], use_lora=True, mode='text')
                elif cls_img_sims is not None and cls_text_sims is not None: # Visual + Textual
                    cls_sims = (cls_img_sims + cls_text_sims) / 2 
                    cls_logits = cls_img_logits # FIXME: TEMP LOGITS
                    rep_cls_texts = ' '.join([self.build_prompts(img_selected_ctxs, text_selected_ctxs), batch[0]["prompt"][0]])
                    if self.config.use_lora:
                        rep_cls_texts = self.build_prompts([img_selected_ctxs[0], text_selected_ctxs[0]], batch[0]["prompt"][0], use_lora=True, mode='comp')
                else: 
                    cls_sims = torch.zeros(1)
                    cls_logits = torch.zeros(1)
                    rep_cls_texts = base_prompt.format( 
                        question = batch[0]["prompt"][0], 
                        answer = ''
                    )
            else:
                if self.config.base_prompt_edit:
                    rep_cls_texts = base_prompt.format(
                        question = batch[0]["prompt"][0],
                        answer = ''
                    )
                else:
                    rep_cls_texts = batch[0]["prompt"][0] + ' '

            # Prepare Inputs
            rep_inputs = {}
            if self.config.model_name == "llava":
                rep_inputs['labels'] = batch[0]['labels']
                rep_inputs['image'] = batch[0]['image']
                rep_inputs['prompts_len'] = [len(self.tok.encode(rep_cls_texts, add_special_tokens=False))]
                rep_inputs['text_input'] = ["".join([rep_cls_texts, batch[0]['answer'][0]])]
            elif self.config.model_name == "blip2":
                rep_inputs['labels'] = batch[0]['labels']
                rep_inputs['image'] = batch[0]['image']
                rep_inputs['prompts_len'] = [len(self.tok.encode(rep_cls_texts, add_special_tokens=False))]
                rep_inputs['text_input'] = ["".join([rep_cls_texts, batch[0]['answer'][0]])]
            elif self.config.model_name == 'minigpt4':
                rep_inputs['labels'] = batch[0]['labels']
                rep_inputs['image'] = batch[0]['image']
                rep_inputs['prompts_len'] = [len(self.tok.encode(rep_cls_texts, add_special_tokens=False))]
                rep_inputs['text_input'] = ["".join([rep_cls_texts, batch[0]['answer'][0]])]
            else: # TODO: add other models
                print(self.config.model_name + ' not supported!')
                pass

            if 'minigpt4' in self.config.model_name.lower() or 'blip' in self.config.model_name.lower() or 'llava' in self.config.model_name.lower():
                pbar = trange(self.config.num_steps, ncols=120)
                for it in pbar:
                    if batch[0]['image'] is not None and it == 2: # Visual
                        rep_inputs = {}
                        rep_inputs['labels'] = batch[0]['labels']
                        rep_inputs['image'] = batch[0]['image']
                        rep_cls_texts = "".join([rep_cls_texts, batch[0]['answer'][0], ' ', rep_cls_texts])
                        rep_inputs['prompts_len'] = [len(self.tok.encode(rep_cls_texts, add_special_tokens=False))]
                        rep_inputs['text_input'] = ["".join([rep_cls_texts, batch[0]['answer'][0]])]
                    elif batch[0]['image'] is None and it == 5: # Textual
                        rep_inputs = {}
                        rep_inputs['labels'] = batch[0]['labels']
                        rep_inputs['image'] = batch[0]['image']
                        rep_cls_texts = "".join([rep_cls_texts, batch[0]['answer'][0], ' ', rep_cls_texts])
                        rep_inputs['prompts_len'] = [len(self.tok.encode(rep_cls_texts, add_special_tokens=False))]
                        rep_inputs['text_input'] = ["".join([rep_cls_texts, batch[0]['answer'][0]])]

                    opt.zero_grad()
                    if self.config.use_lora:
                        if self.config.model_name == "blip2" or self.config.model_name == 'minigpt4':
                            outputs = self.model(rep_inputs)
                        if self.config.model_name == "llava":
                            outputs = self.model.model(rep_inputs) # PeftModelForCasualLM -> LlavaLlamaForCausalLM (LoRA: PeftModelForCasualLM) 
                    else:
                        outputs = self.model(rep_inputs)

                    if not isinstance(outputs, torch.Tensor):
                        outputs = outputs.logits
                    loss = self.edit_loss_fn(self.config, outputs, rep_inputs["labels"])["nll"]
                    pbar.set_postfix({"loss": loss.item()})
                    loss.backward()

                    opt.step()

                    for name, param in self.model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).sum() > 0:
                            print(f"NaN detected in gradients of {name}")

                        if torch.isnan(param).sum() > 0:
                            print(f"NaN detected in parameters of {name}")
                    
                    if connector_mode and it >= 2: # connector -> only 3 steps 
                        break
            else:
                raise not NotImplementedError("Model not supported")
    
        new_model = OURS(self.model, self.config, self.model_constructor, self.classifier, self.classifier_tok,
                        cache_inputs, cache_labels, cache_questions, cache_text_inputs, cache_text_labels, cache_text_questions,
                        self.scale, cache_image_inputs, self.classifier_image, self.classifier_image_processor, self.tok)        
        new_model.train(self.training)

        return new_model, {}

    def stats(self):
        return self.last_stats

    def embedding_logsim_matrix(self, cls_ctxs, test_input_text):
        cls_ctx_input = self.classifier_tok(cls_ctxs, return_tensors="pt", padding=True).to(self.config.device)
        cls_main_input = self.classifier_tok(test_input_text, return_tensors="pt", padding=True).to(self.config.device)
        if 'bert' in self.config.cls_name:
            # bert or distilbert
            ctx_embeds = self.classifier(**cls_ctx_input).last_hidden_state[:, 0].unsqueeze(1)
            main_embeds = self.classifier(**cls_main_input).last_hidden_state[:, 0].unsqueeze(1)
        else:
            # sentence-transformers model
            ctx_embeds = self.classifier(**cls_ctx_input).pooler_output.unsqueeze(1)
            main_embeds = self.classifier(**cls_main_input).pooler_output.unsqueeze(1)
        ctx_embeds = ctx_embeds.view(ctx_embeds.shape[0], self.config.dist_heads, -1)
        main_embeds = main_embeds.view(main_embeds.shape[0], self.config.dist_heads, -1)
        if self.config.bound_embeds:
            ctx_embeds = ctx_embeds.tanh()
            main_embeds = main_embeds.tanh()

        if self.config.cos:
            cos = (ctx_embeds[None] * main_embeds[:, None]).sum(-1) / (ctx_embeds[None].norm(2, -1) * main_embeds[:, None].norm(2, -1))
            dists = 1 - cos
        else:
            dists = (ctx_embeds[None] - main_embeds[:, None]).norm(2, -1)
            if self.config.square:
                dists = dists ** 2

        dists = dists.min(-1).values  # get rid of the dists head dimension

        assert dists.min() >= 0, "Shouldn't have negative distances!"
        cls_logsims = -dists * self.scale

        return cls_logsims
    
    def build_prompts(self, matched_qa, question, use_lora=False, mode=None):
        if use_lora:
            if mode == 'vis':
                if type(matched_qa) == list:
                    matched_qa = matched_qa[0]
                if type(question) == list:
                    question = question[0]
                output = ''.join([base_prompt.format(question = matched_qa), question])
                # output = vis_prompt_template.format(
                #     visual_info = matched_qa, 
                #     question = question
                # )
                
            elif mode == 'text':
                if type(matched_qa) == list:
                    matched_qa = matched_qa[0]
                if type(question) == list:
                    question = question[0]
                output = ''.join([base_prompt.format(question = matched_qa), question])
                # output = text_prompt_template.format(
                #     textual_info = matched_qa,
                #     question = question
                # )
            elif mode == 'comp':
                output = comp_prompt_template.format(
                    visual_info = matched_qa[0], 
                    textual_info = matched_qa[1], 
                    question = question
                ) # Visual, Textual
        else:
            if type(matched_qa) == list:
                matched_qa = matched_qa[0]
            if type(question) == list:
                question = question[0]
            output = ' '.join([str(matched_qa), str(question)])
        return output

    def encode_images(self, images):
        image_features = self.classifier_image(images)
        # print(self.get_model().get_vision_tower().dtype, image_features.dtype)
        # image_features = self.get_model().mm_projector(image_features)
        return image_features

    def embedding_logsim_matrix_images(self, cls_ctxs, test_input_image):
        
        #FIXME: add multiple batch code
        cls_ctxs = cls_ctxs
        test_input_image = test_input_image

        ctx_embeds = cls_ctxs.to(self.config.device)
        # # (1) cls_ctxs
        # if type(cls_ctxs) is list or cls_ctxs.ndim == 5:
        #     concat_images = torch.cat([image for image in cls_ctxs], dim=0)
        #     cls_image_features = self.encode_images(concat_images)
        #     # for Projector_layer
        #     # split_sizes = [image.shape[0] for image in cls_ctxs]
        #     # cls_image_features = torch.split(cls_image_features, split_sizes, dim=0)
        #     # cls_image_features = [x.flatten(0, 1).to(self.config.device) for x in cls_image_features]
        # else:
        #     cls_image_features = self.encode_images(cls_ctxs).to(self.config.device)
        # ctx_embeds = cls_image_features.last_hidden_state[:, 0].unsqueeze(1)

        # (2) test_input_image
        if type(test_input_image) is list or test_input_image.ndim == 5:
            concat_images = torch.cat([image for image in test_input_image], dim=0)
            main_image_features = self.encode_images(concat_images)
            # for Projector_layer
            # split_sizes = [image.shape[0] for image in cls_ctxs]
            # cls_image_features = torch.split(cls_image_features, split_sizes, dim=0)
            # cls_image_features = [x.flatten(0, 1).to(self.config.device) for x in cls_image_features]
        else:
            main_image_features = self.encode_images(cls_ctxs).to(self.config.device)
        main_embeds = main_image_features.last_hidden_state[:, 0].unsqueeze(1)

        # embedding logsim matrix
        ctx_embeds = ctx_embeds.view(ctx_embeds.shape[0], self.config.dist_heads, -1)
        main_embeds = main_embeds.view(main_embeds.shape[0], self.config.dist_heads, -1)
        if self.config.bound_embeds:
            ctx_embeds = ctx_embeds.tanh()
            main_embeds = main_embeds.tanh()

        if self.config.cos:
            cos = (ctx_embeds[None] * main_embeds[:, None]).sum(-1) / (ctx_embeds[None].norm(2, -1) * main_embeds[:, None].norm(2, -1))
            dists = 1 - cos
        else:
            dists = (ctx_embeds[None] - main_embeds[:, None]).norm(2, -1)
            if self.config.square:
                dists = dists ** 2

        dists = dists.min(-1).values  # get rid of the dists head dimension

        assert dists.min() >= 0, "Shouldn't have negative distances!"
        cls_logsims = -dists * self.scale

        return cls_logsims

    def crossattend_logsim_matrix(self, cls_ctxs, test_input_texts):
        batch = [ctx + self.classifier_tok.sep_token + test for test in test_input_texts for ctx in cls_ctxs]
        batch_toks = self.classifier_tok(batch, return_tensors="pt", padding=True).to(self.config.device)
        batch_logsims = self.classifier(**batch_toks).logits.log_softmax(-1)[:, 0]
        logsim_matrix = batch_logsims.view(len(test_input_texts), len(cls_ctxs))

        return logsim_matrix

    def build_rep_cache_contexts(self, edit_type):
        if edit_type == 'visual':
            sep = " "
            # if hasattr(self.model, "name_or_path") and "gpt" in self.model.name_or_path.lower():
                # The labels are include in the inputs for autoregressive models. Cut off the label for the classifier
            ctxs = [cin + sep for cin in self.cache_inputs]
            questions = [q for q in self.cache_questions]
            answers = [ans for ans in self.cache_labels]
            # else:
            #     ctxs = [cin + sep + clab + sep for cin, clab in zip(self.cache_inputs, self.cache_labels)]
        else:
            sep = " "
            ctxs = [cin + sep for cin in self.cache_text_inputs]
            questions = [q for q in self.cache_text_questions]
            answers = [ans for ans in self.cache_text_labels]
        return ctxs, questions, answers

    def build_cls_cache_inputs(self):
        # sep = self.classifier_tok.sep_token
        # if hasattr(self.model, "name_or_path") and "gpt" in self.model.name_or_path.lower():
            # The labels are include in the inputs for autoregressive models. Cut off the label for the classifier
        # inputs = [cin.rsplit(" ", 1)[0] + sep for cin in self.cache_inputs]
        # else:
        #     inputs = [cin + sep + clab + sep for cin, clab in zip(self.cache_inputs, self.cache_labels)]
        inputs = [ cin for cin, clabel in zip(self.cache_inputs, self.cache_labels)]
        return inputs

    def build_cls_cache_image_inputs(self):
        return self.cache_image_inputs
    
    def build_cls_text_cache_inputs(self):
        inputs = [ cin for cin, clabel in zip(self.cache_text_inputs, self.cache_text_labels)]
        return inputs

    def build_rep_input_tokens(self, kwargs, idxs, edit_type, generation=False):
        if "input_ids" in kwargs and kwargs["input_ids"] is not None:
            assert len(idxs) == len(kwargs["input_ids"]), "Need one cache idx for each test input"
        cache_contexts, cache_questions, cache_labels = self.build_rep_cache_contexts(edit_type)
        # print('context', cache_contexts)
        selected_contexts = [cache_contexts[idx.item()] for idx in idxs]
        selected_questions = [cache_questions[idx.item()] for idx in idxs]
        selected_labels = [cache_labels[idx.item()] for idx in idxs]
        # print('selected', selected_contexts)
        test_inputs = kwargs["text_input"]
        rep_texts = [self.build_prompts(ctx, inp) for ctx, inp in zip(selected_contexts, test_inputs)]
        if "qwen-vl" in self.config.model_name.lower() or "owl-2" in self.config.model_name.lower():
            rep_input_tokens = self.tok(rep_texts, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        else:
            rep_input_tokens = self.tok(rep_texts, return_tensors="pt", padding=True).to(self.config.device)

        rep_kwargs = {
            "input_ids": rep_input_tokens["input_ids"],
            "attention_mask": rep_input_tokens["attention_mask"],
        }

        if not generation:
            if 'labels' in kwargs.keys():
                rep_kwargs["labels"] = kwargs["labels"]

        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2" or self.config.model_name == "llava":
            # Add 'ignore' labels for the prepended cache inputs
            pre = torch.full((kwargs["labels"].shape[0], rep_kwargs["input_ids"].shape[-1] - kwargs["labels"].shape[-1]), -100,
                             device=kwargs["labels"].device)
            rep_kwargs["labels"] = torch.cat((pre, kwargs["labels"]), dim=-1)
        return rep_kwargs, rep_texts, selected_contexts, selected_questions, selected_labels

    def run_image_classifier(self, *inputs, **kwargs):
        cache_inputs = self.build_cls_cache_inputs()
        test_inputs = inputs[0]["image_level"]
        cache_image_inputs = self.build_cls_cache_image_inputs()
        test_image_inputs = inputs[0]["image_cls"]
        
        if cache_inputs != [] and cache_image_inputs != []:
            # Text Classifier
            if self.config.cross_attend: # cross-attention
                log_sim_matrix = self.crossattend_logsim_matrix(cache_inputs, test_inputs)
            else: # embedding log similarity (default)
                log_sim_matrix = self.embedding_logsim_matrix(cache_inputs, test_inputs)
            sims = log_sim_matrix.exp()
            assert sims.max() <= 1, "Similarities shouldn't exceed 1!"
            # print(sims.shape) # [1,1]
            cls_sims, cls_idxs = sims.max(-1)
            
            # Image Classifier
            if self.config.only_text == False:
                image_log_sim_matrix = self.embedding_logsim_matrix_images(cache_image_inputs, test_image_inputs)
                image_sims = image_log_sim_matrix.exp()
                assert image_sims.max() <= 1, "Similarities(image) shouldn't exceed 1!"
                if inputs[0]["text_level"][0] is None:
                    final_sims = (sims + image_sims) / 2
                else:
                    final_sims = (0.1 * sims + 0.9 * image_sims)
                cls_sims, cls_idxs = final_sims.max(-1)
        else:
            cls_sims, cls_idxs, log_sim_matrix = None, None, None

        return cls_sims, cls_idxs, log_sim_matrix
    
    def run_text_classifier(self, *inputs, **kwargs):
        cache_text_inputs = self.build_cls_text_cache_inputs()
        test_inputs = inputs[0]["text_level"]

        if cache_text_inputs != []:
            # Text Classifier
            if self.config.cross_attend: # cross-attention
                log_sim_matrix = self.crossattend_logsim_matrix(cache_text_inputs, test_inputs)
            else: # embedding log similarity (default)
                log_sim_matrix = self.embedding_logsim_matrix(cache_text_inputs, test_inputs)
            sims = log_sim_matrix.exp()
            assert sims.max() <= 1, "Similarities shouldn't exceed 1!"
            # print(sims.shape) # [1,1]
            cls_sims, cls_idxs = sims.max(-1)
        else:
            cls_sims, cls_idxs, log_sim_matrix = None, None, None
        
        return cls_sims, cls_idxs, log_sim_matrix

    def divide_image_text_level(self, *inputs, **kwargs):
        # Divide Image-Level / Text-Level
        [image_level, text_level] = inputs[0]['text_input_query'][0].split('\n', maxsplit=1)
        image_level = image_level[len('Image Level: '):].replace('[', '').replace(']', '').strip() if image_level.startswith('Image Level: ') else 'None'
        text_level = text_level[len('Text Level: '):].strip() if text_level.startswith('Text Level: ') else 'None'
        image_level = None if image_level == 'None' or image_level == "" else image_level
        text_level = None if text_level == 'None' or text_level == "" else text_level

        inputs[0]['image_level'] = [image_level]
        inputs[0]['text_level'] = [text_level]

        return image_level, text_level, inputs


    def generate(self, *args, **kwargs):
        input_text = self.replacement_tok.batch_decode(kwargs["input_ids"], skip_special_tokens=True)

        assert len(args) == 0, "Should only pass named arguments to generate()"
        if len(self.cache_inputs) > 0:
            cls_sims, cls_idxs, _ = self.run_image_classifier(*args, **kwargs)
            assert cls_sims.numel() == 1
            print(f"Cache score: {cls_sims.item()} " + ("[MISS]" if cls_sims.item() < 0.5 else "[HIT]"))
            if cls_sims.item() > 0.5:
                rep_input = self.build_rep_input_tokens(kwargs, cls_idxs, generation=True)
                kwargs["input_ids"] = rep_input["input_ids"]
                kwargs["attention_mask"] = rep_input["attention_mask"]
                rep_input_text = self.replacement_tok.decode(rep_input["input_ids"][0])
                print(f"Returning counterfactual model output for '{rep_input_text}'")
                if self.config.freeze_cntr:
                    return self.model.generate(*args, **kwargs)
                else:
                    return self.replacement.generate(*args, **kwargs)

        print(f"Returning base model output for '{input_text}'")
        return self.model.generate(*args, **kwargs)

    def forward(self, *inputs, return_logits_only=True, eps=torch.finfo(torch.float32).eps, pos_pairs=None, **kwargs):
        grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(self.training)

        # need to do soft mixing of logits if we're doing supervised training or we've specifically requested it
        soft = (not self.config.supervised) or self.config.soft_weighting
        with torch.no_grad():

            # Prepare Base Input
            base_inputs = {}
            if self.config.use_lora and self.config.base_prompt_test:
                base_texts = base_prompt.format(
                    question = inputs[0]['prompt'][0],
                    answer = ''
                )
            else:
                base_texts = inputs[0]["prompt"][0] + ' '

            if self.config.model_name == 'llava':
                base_inputs['image'] = inputs[0]['image']
                base_inputs['labels'] = inputs[0]['labels']
                base_inputs['prompts_len'] = [len(self.tok.encode(base_texts, add_special_tokens=False))]
                base_inputs['text_input'] = [base_texts + inputs[0]['answer'][0]]
            elif self.config.model_name == "blip2":
                base_inputs['image'] = inputs[0]['image']
                base_inputs['labels'] = inputs[0]['labels']
                base_inputs['prompts_len'] = [len(self.tok.encode(base_texts, add_special_tokens=False))]
                base_inputs['text_input'] = [base_texts + inputs[0]['answer'][0]]
            elif self.config.model_name == 'minigpt4':
                base_inputs['image'] = inputs[0]['image']
                base_inputs['labels'] = inputs[0]['labels']
                base_inputs['prompts_len'] = [len(self.tok.encode(base_texts, add_special_tokens=False))]
                base_inputs['text_input'] = [base_texts + inputs[0]['answer'][0]]
            else:
                raise NotImplementedError
            base_inputs = base_inputs,

            if len(self.cache_inputs) == 0 and len(self.cache_text_inputs) == 0:
                if self.config.model_name == "blip2" or self.config.model_name == "minigpt4" or self.config.model_name == "llava":
                    if self.config.use_lora:
                        if self.config.model_name == 'blip2':
                            self.model.opt_model.set_adapter([])
                        elif self.config.model_name == 'llava':
                            self.model.set_adapter([])
                        elif self.config.model_name == 'minigpt4':
                            self.model.llama_model.set_adapter([])
                    super_out = super().forward(*base_inputs, **kwargs).float()
                elif self.config.model_name == "qwen-vl": #TODO: Not implemented
                    super_out = self.model(inputs[0]['inputs'], **kwargs)
                elif "owl-2" in self.config.model_name.lower(): #TODO: Not implemented
                    _input = inputs[0]
                    super_out = self.model(_input['input_ids'].to(self.config.device), 
                                           images=_input['image'].to(self.config.device, dtype=torch.float16),
                                           **kwargs)
                else:
                    super_out = super().forward(*base_inputs, **kwargs).float()
                torch.set_grad_enabled(grad_enabled)
                return super_out
            else:
                if self.config.model_name == "blip2":
                    if "prompts_len" in kwargs:
                        prompts_len = kwargs.pop("prompts_len")
                    if self.config.use_lora:
                        self.model.opt_model.set_adapter([])
                    base_logits = super().forward(*base_inputs, **kwargs)
                    if not isinstance(base_logits, torch.Tensor):
                        base_logits = base_logits.logits
                    base_logits = base_logits.float()
                elif self.config.model_name == "minigpt4" or self.config.model_name == "llava" or self.config.model_name == "qwen-vl":
                    if self.config.use_lora:
                        if self.config.model_name == 'llava':
                            self.model.set_adapter([])
                        elif self.config.model_name == 'minigpt4':
                            self.model.llama_model.set_adapter([])
                    base_logits = super().forward(*base_inputs, **kwargs)
                    if not isinstance(base_logits, torch.Tensor):
                        base_logits = base_logits.logits
                    base_logits = base_logits.float()
                elif "owl-2" in self.config.model_name.lower():
                    self.model.train(False)
                    base_logits = super().forward(*base_inputs, **kwargs)
                    if not isinstance(base_logits, torch.Tensor):
                        base_logits = base_logits.logits
                    base_logits = base_logits.float()
                    self.model.train(self.training)
                else:
                    base_logits = super().forward(*base_inputs, **kwargs).float()
                if soft:
                    if base_logits.dim() == 3:
                        base_probs = base_logits.softmax(-1)
                    else:
                        base_probs = base_logits.sigmoid()
                    del base_logits

        image_level, text_level, _ = self.divide_image_text_level(*inputs, **kwargs)
        
        cls_img_sims, cls_text_sims = None, None

        if image_level != None: # Image-Level Memory Matching
            cls_img_sims, cls_img_idxs, cls_img_logits = self.run_image_classifier(*inputs, **kwargs)
            if cls_img_sims is not None:
                rep_img_cls_inputs, rep_img_cls_texts, img_selected_ctxs, img_selected_qs, img_selected_labels = self.build_rep_input_tokens(inputs[0], cls_img_idxs, edit_type="visual")

        if text_level != None: # Text-Level Memory Matching
            if cls_img_sims is not None:
                inputs[0]["text_level"] = re.sub(r'\[.*?\]', img_selected_labels[0], text_level) # self.build_prompts(img_selected_ctxs, text_level)
            else:
                inputs[0]["text_level"] = text_level.replace('[', '').replace(']', '')
            cls_text_sims, cls_text_idxs, cls_text_logits = self.run_text_classifier(*inputs, **kwargs)
            if cls_text_sims is not None:
                rep_text_cls_inputs, rep_text_cls_texts, text_selected_ctxs, text_selected_qs, text_selected_labels = self.build_rep_input_tokens(inputs[0], cls_text_idxs, edit_type="textual")

        if cls_img_sims is not None and cls_text_sims is None: # Visual 
            cls_sims = cls_img_sims
            cls_logits = cls_img_logits 
            rep_cls_texts = self.build_prompts(img_selected_ctxs, inputs[0]["prompt"][0]) + ' '
            if self.config.use_lora:
                if self.config.base_prompt_test:
                    img_selected_prompt = base_prompt.format(question = img_selected_qs[0], answer = img_selected_labels[0])
                    rep_cls_texts = ' '.join([img_selected_prompt, base_prompt.format(question = inputs[0]["prompt"][0], answer = '')])
                
                if self.config.model_name == 'blip2':
                    self.model.opt_model.set_adapter('visual')
                elif self.config.model_name == 'llava':
                    self.model.set_adapter('visual')
                elif self.config.model_name == 'minigpt4':
                    self.model.llama_model.set_adapter('visual')
                # rep_cls_texts = self.build_prompts(img_selected_ctxs, inputs[0]["prompt"][0], use_lora=True, mode='vis')
                # self.model.set_adapter("visual")
        elif cls_img_sims is None and cls_text_sims is not None: # Textual
            cls_sims = cls_text_sims
            cls_logits = cls_text_logits
            rep_cls_texts = self.build_prompts(text_selected_ctxs, inputs[0]["prompt"][0]) + ' '
            if self.config.use_lora:
                if self.config.base_prompt_test:
                    text_selected_prompt = base_prompt.format(question = text_selected_qs[0], answer=text_selected_labels[0])
                    rep_cls_texts = ' '.join([text_selected_prompt, base_prompt.format(question = inputs[0]["prompt"][0], answer = '')])
                if self.config.model_name == 'blip2':
                    self.model.opt_model.set_adapter('textual')
                elif self.config.model_name == 'llava':
                    self.model.set_adapter('textual')
                elif self.config.model_name == 'minigpt4':
                    self.model.llama_model.set_adapter('textual')
                # rep_cls_texts = self.build_prompts(text_selected_ctxs, inputs[0]["prompt"][0], use_lora=True, mode='text')
                # self.model.set_adapter("textual")
        elif cls_img_sims is not None and cls_text_sims is not None: # Visual + Textual
            cls_sims = (cls_img_sims + cls_text_sims) / 2 
            cls_logits = cls_img_logits # FIXME: TEMP LOGITS
            rep_cls_texts = ' '.join([self.build_prompts(img_selected_ctxs, text_selected_ctxs), inputs[0]["prompt"][0]])
            if self.config.use_lora:
                visual_info = f"{img_selected_qs[0]} -> {img_selected_labels[0]}"
                textual_info = f"{text_selected_qs[0]} -> {text_selected_labels[0]}"
                rep_cls_texts = self.build_prompts([visual_info, textual_info], inputs[0]["prompt"][0], use_lora=True, mode='comp')
                if self.config.model_name == 'blip2':
                    self.model.opt_model.set_adapter(["visual", "textual", "connector"])
                elif self.config.model_name == 'llava':
                    self.model.set_adapter(["visual", "textual", "connector"])
                elif self.config.model_name == 'minigpt4':
                    self.model.llama_model.set_adapter(['visual', 'textual', 'connector'])
        else: 
            cls_sims = torch.zeros(1)
            cls_logits = torch.zeros(1)
            rep_cls_texts = base_prompt.format(
                question = inputs[0]["prompt"][0],
                answer = ''
            )

        # Prepare Inputs
        rep_inputs = {}
        if self.config.model_name == "llava":
            rep_inputs['image'] = inputs[0]['image']
            rep_inputs['labels'] = inputs[0]['labels']
            rep_inputs['prompts_len'] = [len(self.tok.encode(rep_cls_texts, add_special_tokens=False))]
            rep_inputs['text_input'] = [''.join([rep_cls_texts, inputs[0]['answer'][0]])]
        elif self.config.model_name == 'blip2':
            rep_inputs['image'] = inputs[0]['image']
            rep_inputs['labels'] = inputs[0]['labels']
            rep_inputs['prompts_len'] = [len(self.tok.encode(rep_cls_texts, add_special_tokens=False))]
            rep_inputs['text_input'] = [''.join([rep_cls_texts, inputs[0]['answer'][0]])]
        elif self.config.model_name == 'minigpt4':
            rep_inputs['image'] = inputs[0]['image']
            rep_inputs['labels'] = inputs[0]['labels']
            rep_inputs['prompts_len'] = [len(self.tok.encode(rep_cls_texts, add_special_tokens=False))]
            rep_inputs['text_input'] = [''.join([rep_cls_texts, inputs[0]['answer'][0]])]
        else: 
            print(self.config.model_name + ' not supported')
            pass
        rep_inputs = rep_inputs,

        # Forward
        if self.config.model_name == "blip2":
            if "prompts_len" in kwargs:
                prompts_len = kwargs.pop("prompts_len")
            rep_cls_logits = super().forward(*rep_inputs, **kwargs)
            if not isinstance(rep_cls_logits, torch.Tensor):
                rep_cls_logits = rep_cls_logits.logits
            rep_cls_logits = rep_cls_logits.float()
        elif self.config.model_name == "minigpt4" or self.config.model_name == "llava" or self.config.model_name == "qwen-vl":
            rep_cls_logits = super().forward(*rep_inputs, **kwargs)
            if not isinstance(rep_cls_logits, torch.Tensor):
                rep_cls_logits = rep_cls_logits.logits
            rep_cls_logits = rep_cls_logits.float()
        elif "owl-2" in self.config.model_name.lower():
            self.model.train(False)
            rep_cls_logits = super().forward(*rep_inputs, **kwargs)
            if not isinstance(rep_cls_logits, torch.Tensor):
                rep_cls_logits = rep_cls_logits.logits
            rep_cls_logits = rep_cls_logits.float()
            self.model.train(self.training)
        else:
            rep_cls_logits = super().forward(*rep_inputs, **kwargs).float()

        if pos_pairs is not None:
            assert (pos_pairs[:, 0] == torch.arange(pos_pairs.shape[0], device=pos_pairs.device)).all()
            gold_idxs = pos_pairs[:, 1]
            rep_gold_inputs = self.build_rep_input_tokens(kwargs, gold_idxs)
            if self.config.freeze_cntr:
                rep_gold_logits = super().forward(**rep_gold_inputs)
            else:
                rep_gold_logits = _logits(self.replacement(**rep_gold_inputs))
        else:
            rep_gold_logits = rep_cls_logits

        cls_sims = cls_sims.view(-1, 1)  # For (binary) classification, predictions are (B x 1)
        if rep_cls_logits.dim() == 3:
            cls_sims.unsqueeze_(-1)  # For generation/seq2seq, predictions are (B x S x V)

        stats = {
            'sims/mean': cls_sims.mean().item(),
            'sims/pos': (cls_sims >= 0.5).float().mean().item(),
            'sims/neg': (cls_sims < 0.5).float().mean().item(),
            'params/scale': self.scale.item() if self.scale is not None else 0.0,
        }

        # if hasattr(self.model, "name_or_path") and "gpt" in self.model.name_or_path.lower():
        #     rep_cls_logits = rep_cls_logits[:, -kwargs["labels"].shape[-1]:, :]

        if soft:
            if base_probs.size(1) != rep_cls_logits.size(1):
                if base_probs.size(1) < rep_cls_logits.size(1):
                    rep_cls_logits = rep_cls_logits[:, -base_probs.size(1):, :]
                else: 
                    additional = torch.zeros(
                        base_probs.size(0),
                        base_probs.size(1) - rep_cls_logits.size(1),
                        base_probs.size(2)).to(rep_cls_logits.device)
                    rep_cls_logits = torch.cat([rep_cls_logits, additional], dim=1)
            rep_weight = cls_sims
            if rep_cls_logits.device != base_probs.device:
                rep_cls_logits = rep_cls_logits.to(base_probs.device)
            if rep_weight.device != base_probs.device:
                rep_weight = rep_weight.to(base_probs.device)
            if base_probs.dim() == 3:
                mixture_logits = ((1 - rep_weight) * base_probs + rep_weight * rep_cls_logits.softmax(-1) + eps).log()
            else:
                mixture_logits = ((1 - rep_weight) * base_probs + rep_weight * rep_cls_logits.sigmoid() + eps).log()
        else:
            if base_logits.size(1) != rep_cls_logits.size(1):
                rep_cls_logits = rep_cls_logits[:, -base_logits.size(1):, :]
            rep_idxs = torch.where(cls_sims > 0.5)[0]
            mixture_logits = base_logits
            if rep_idxs.numel() > 0:
                if rep_cls_logits.device != mixture_logits.device:
                    rep_cls_logits.to(mixture_logits.device)
                mixture_logits[rep_idxs] = rep_cls_logits[rep_idxs]

        torch.set_grad_enabled(grad_enabled)
        if return_logits_only:
            return mixture_logits
        else:
            return mixture_logits, cls_logits, rep_gold_logits, stats

if __name__ == '__main__':
    import types

    model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")

    config = types.SimpleNamespace()
    config.inner_params = [
        "transformer.h.9.mlp.c_fc.weight",
        "transformer.h.9.mlp.c_proj.weight",
        "transformer.h.10.mlp.c_fc.weight",
        "transformer.h.10.mlp.c_proj.weight",
        "transformer.h.11.mlp.c_fc.weight",
        "transformer.h.11.mlp.c_proj.weight",
    ]
    config.edit_lr = 0.0001

    config.gtn = types.SimpleNamespace()
    config.gtn.n_hidden = 1
    config.gtn = config.gtn.__dict__

    gtn = SERAC(model, config, lambda: copy.deepcopy(model)).cuda()
    # torch.save(gtn.state_dict(), "test_state.pt")
    import pdb; pdb.set_trace()
    gtn.load_state_dict(torch.load("test_state.pt"))
    x = torch.arange(20).view(1, 20).cuda() + 1000
    orig_logits = gtn(x)
    edited = gtn.edit(x, masks=torch.ones_like(x), labels=x)
    post_logits = gtn(x)

    assert torch.allclose(orig_logits, post_logits)

    orig_param = [p for (n, p) in gtn.model.named_parameters() if n == config.inner_params[-1]][0]
    edited_param = [p for (n, p) in edited.model.named_parameters() if n == config.inner_params[-1]][0]

    LOG.info((orig_param - edited_param).abs().max())
    edited.eval()
    LOG.info(gtn(x, labels=x).loss, edited(x, labels=x).loss, edited.edit_loss_fn(edited(x).logits, x)["nll"])
    edited2 = edited.edit(x, masks=torch.ones_like(x), labels=x)
    LOG.info(gtn(x, labels=x).loss, edited(x, labels=x).loss, edited2(x, labels=x).loss)