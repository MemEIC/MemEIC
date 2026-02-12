from dataclasses import dataclass, field
from ...util.hparams import HyperParams
from typing import Optional, Any, List
import yaml


@dataclass
class LORAMultimodalHparams(HyperParams):
    """
    Hyperparameters for LoRA Baseline (One-LoRA) multimodal editing.
    Uses FT algorithm with LoRA parameters.
    """
    
    # Multimodal
    qformer_name_or_path: Optional[str] = None
    state_dict_file: Optional[str] = None
    qformer_checkpoint: Optional[str] = None
    pretrained_ckpt: Optional[str] = None
    
    # Image_dir
    coco_image: str = None
    rephrase_image: str = None
    
    # Model
    name: str = None
    model_name: str = None
    model_class: str = None
    tokenizer_class: str = None
    tokenizer_name: str = None
    inner_params: List[str] = None

    archive: Any = None

    # LoRA parameters
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    lora_connector_type: Optional[str] = None
    for_eval: bool = False
    adapter_path: Optional[str] = None
    one_lora: bool = True  # One-LoRA baseline

    # Method
    alg: str = 'lora'
    alg_name: str = 'LORA'
    num_steps: int = 10
    lr: float = 1e-6
    edit_lr: float = 1e-4
    lr_lr: float = 1e-4
    lr_scale: float = 1.0
    weight_decay: float = 0
    seed: int = 42
    debug: bool = False
    cedit: float = 0.1
    iedit: float = 0.1
    cloc: float = 1.0
    cbase: float = 1.0
    dropout: float = 0.0
    train_base: bool = False
    no_grad_layers: Any = None
    one_sided: bool = False
    n_hidden: int = 1
    hidden_dim: Any = None
    init: str = 'id'
    norm: bool = True
    combine: bool = True
    x_only: bool = False
    delta_only: bool = False
    act: str = 'relu'
    rank: int = 1920
    mlp_class: str = 'IDMLP'
    shared: bool = True

    # Output
    results_dir: str = './results'

    # Train
    device: int = 0
    model_save_pt: int = 5000
    silent: bool = False
    log_interval: int = 100
    eval_log_interval: int = 1000
    final_eval: bool = True
    val_interval: int = 5000
    early_stop_patience: int = 5000
    early_stop_key: str = "loss/total_edit_val"
    eval_only: bool = True
    half: bool = False
    save: bool = False
    verbose: bool = True

    val_batch_size: int = 1
    accumulate_bs: int = 2
    val_steps: int = 500
    opt: str = 'Adam'
    grad_clip: float = 100.0
    
    batch_size: int = 1
    max_length: int = 30
    max_epochs: Optional[int] = None
    max_iters: Optional[int] = 50000
    model_parallel: bool = False
    freeze_qformer: bool = True

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg'] == 'lora') or print(
            f'LORAMultimodalHparams can not load from {hparams_name_or_path}, '
            f'alg_name is {config["alg"]} '
        )
        return cls(**config)
