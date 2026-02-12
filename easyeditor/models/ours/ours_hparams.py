from dataclasses import dataclass, field
from ...util.hparams import HyperParams
from typing import Optional, Any, List
import yaml


@dataclass
class OURSMultimodalHparams(HyperParams):

    # Multimodal
    qformer_name_or_path: str
    state_dict_file: str
    
    # Image_dir
    coco_image: str
    rephrase_image: str

    # For OURS
    query_generate: str
    query_prompt_dir: str
    train_adapter: bool
    only_text: bool
    stage: int
    base_prompt_edit: bool
    base_prompt_test: bool
    
    # Model
    name: str
    model_name: str
    model_class: str
    tokenizer_class: str
    tokenizer_name: str
    cls_name: str
    cls_class: str
    cls_image_name: str
    cls_image_class: str
    cls_image_proc_class: str
    inner_params: List[str]

    archive: Any

    # Method
    alg: str
    alg_name: str
    num_steps: int
    lr: float
    edit_lr: float
    seed: int
    lr_lr: float
    cedit: float
    iedit: float
    cloc: float
    cbase: float
    cport: float
    dropout: float
    final_eval: bool
    supervised: bool
    train_base: bool
    no_grad_layers: Any
    soft_weighting: bool
    checkpoint_grad: bool
    cross_attend: bool
    cos: bool
    freeze: Any
    square: bool
    bound_embeds: bool
    use_all_negatives: bool
    dist_heads: int
    lora: Any

    # Output
    results_dir: str

    # Train
    device: str
    batch_size: int
    model_save_pt: int
    edit_bs: int
    silent: bool
    log_interval: int
    val_interval: int
    early_stop_patience: int
    early_stop_key: str
    eval_only: bool
    half: bool
    save: bool
    debug: bool
    log_errors: bool
    unlikelihood: bool

    val_batch_size: int
    accumulate_bs: int
    val_steps: int
    opt: str
    grad_clip: float

    max_length: int = 32
    max_epochs: Optional[int] = None
    max_iters: Optional[int] = None
    model_parallel: bool = False
    qformer_checkpoint: Optional[str] = None
    freeze_qformer: bool = True
    pretrained_ckpt: Optional[str] = None  

    # For LORA
    use_lora: bool = False
    for_eval: bool = False
    adapter_path: str = ''
    one_lora: bool = False  # Added: One LoRA baseline support (single adapter mode)
    lora_edit_lr: float = 1e-4
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = field(default_factory=list)

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)


        assert (config and (config['alg'] == 'OURS' or config['alg'] == 'LORA')) or print(f'OURSMultimodalHparams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg"]} ')
        return cls(**config)
