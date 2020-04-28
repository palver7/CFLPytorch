__version__ = "0.6.3"
from .StdConvsCFL import StdConvsCFL 
from .EquiConvsCFL import EfficientNet as EquiConvs
from .utils import (
    GlobalParams,
    BlockArgs,
    BlockDecoder,
    efficientnet,
    get_model_params,
)

