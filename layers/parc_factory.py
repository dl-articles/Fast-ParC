import enum

from layers.fast_parc import FastParC, FastParCUnit
from layers.parc import ParCUnit


class ParcOperatorVariation(enum.IntEnum):
    FAST = 1
    BASIC = 0
class ParCOperator:
    def __new__(cls, *args, variation: ParcOperatorVariation = ParcOperatorVariation.FAST, **kwargs):
        if variation == ParcOperatorVariation:
            return FastParCUnit(*args, **kwargs)
        else:
            return ParCUnit(*args, **kwargs)