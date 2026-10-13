
from . import registry

_module_profile = "full"
_loaded_profile = None


def load_all_modules(profile="full") -> None:
    """The optional value profile imports only the text-classification/evidence tasks."""
    global _module_profile, _loaded_profile
    if profile not in ("full", "value"):
        raise ValueError(f"Unknown module profile: {profile}")
    if _loaded_profile is not None and profile != _loaded_profile:
        raise RuntimeError("Run different module profiles in separate Python processes")
    _module_profile = profile
    from . import models              
    from . import datasets              
    from . import evaluators              
    from . import metrics              
    from . import runners              
    from . import summarizers              

    _loaded_profile = profile

__all__ = ["registry", "load_all_modules"]
