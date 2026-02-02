
from . import registry

def load_all_modules() -> None:
    from . import models              
    from . import datasets              
    from . import evaluators              
    from . import metrics              
    from . import runners              
    from . import summarizers              

__all__ = ["registry", "load_all_modules"]
