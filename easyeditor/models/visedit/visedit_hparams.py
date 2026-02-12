from dataclasses import dataclass, field
from ...util.hparams import HyperParams
from typing import Optional, Any, List
import yaml


@dataclass
class VisEditHparams(HyperParams):

    # Orig
    edit_model_name: str
    data_name: str
    batch_size: int

    data_n: int
    load_ckpt_path: str
    extra_devices: List[int]
    epochs: int
    train_name_prefix: str
    save_ckpt_per_i: int
    log_per_i: int
    ema_alpha: float
    random_seed: int
    dbs: int

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)


        assert (config and config['alg'] == 'VisEdit') or print(f'VisEditHparams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg"]} ')
        return cls(**config)