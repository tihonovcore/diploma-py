from enum import Enum


class ModelMode(Enum):
    UNTYPED = 1
    TYPED__INJECTION_VIA_NODES = 2
    TYPED__INJECTION_VIA_CONTEXT = 3

    def is_typed(self) -> bool:
        return self is ModelMode.TYPED__INJECTION_VIA_NODES or self is ModelMode.TYPED__INJECTION_VIA_CONTEXT

    def to_string(self) -> str:
        return str(self)[len('ModelMode.'):]
