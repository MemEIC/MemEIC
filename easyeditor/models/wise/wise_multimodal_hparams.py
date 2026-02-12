from dataclasses import dataclass
from typing import List, Union, Optional, Any
from ...util.hparams import HyperParams
import yaml


@dataclass
class WISEMultimodalHyperParams(HyperParams):

    device: int

    name: str
    model_name: str
    model_class: str
    tokenizer_class: str
    tokenizer_name: str
    inner_params: List[str]

    # Method
    alg: str
    alg_name: str
    objective_optimization: str
    mask_ratio: float
    # act_margin: List[float]
    alpha: float    # act_margin[0]
    beta: float  # act_margin[1]
    gamma: float  # act_margin[2]
    act_ratio: float
    merge_freq: int
    retrieve: bool
    replay: bool
    save_freq: Union[int, None]
    merge_alg: str
    norm_constraint: float
    weights: Union[float, None]
    densities: Union[float, None]

    dropout: float
    no_grad_layers: bool
    train_base: bool
    eval_only: bool
    archive: str
    debug: bool
    log_interval: int

    results_dir: str

    # Experiments
    edit_lr: float
    n_iter: int

    # Multimodal
    qformer_name_or_path: str
    qformer_checkpoint: str
    state_dict_file: str
    hidden_act: str

    # Image_dir
    coco_image: str
    rephrase_image: str

    # Defaults
    batch_size: int = 1
    val_batch_size: int = 1
    max_length: int = 30
    model_parallel: bool = False
    use_chat_template: bool = False
    freeze_qformer: bool = True
    pretrained_ckpt: Optional[str] = None 

    # Save and Load
    save_path: str = None
    load_path: str = None

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert config['merge_freq'] % config['save_freq'] == 0, 'merge_freq need to be divisible by save_freq (like 1000 / 500)'
        assert len(config['act_margin']) == 3
        config['alpha'], config['beta'], config['gamma'] = config['act_margin'][0], config['act_margin'][1], config['act_margin'][2]
        config.pop('act_margin')

        assert (config and config['alg_name'] == 'WISE'), \
            f'WISEHyperParams can not load from {hparams_name_or_path}. alg_name is {config["alg_name"]}'
        return cls(**config)