from ..trainer import MEND
from ..trainer import SERAC, SERAC_MULTI
from ..trainer import FT
from ..trainer import OURS
from ..trainer import WISEMultimodal


ALG_TRAIN_DICT = {
    'MEND': MEND,
    'SERAC': SERAC,
    'SERAC_MULTI': SERAC_MULTI,
    'FT': FT,
    'ft': FT,
    'lora': FT,
    'LORA': FT,
    'OURS': OURS,
    'WISE': WISEMultimodal
}
