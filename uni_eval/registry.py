from typing import Any, Dict, Optional, Type

class Registry:

    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Type] = {}

    def register_module(self, name: Optional[str] = None, module: Optional[Type] = None):
        def _register(cls):
            key = name if name is not None else cls.__name__
            if key in self._module_dict:
                raise KeyError(f"{key} is already registered in {self._name}")
            self._module_dict[key] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register

    def build(self, cfg: Dict[str, Any]) -> Any:
        if not isinstance(cfg, dict):
            raise TypeError(f"Config must be a dict, but got {type(cfg)}")
        if 'type' not in cfg:
            raise KeyError(f"Config must contain 'type' key, but got {cfg}")
        
        args = cfg.copy()
        obj_type = args.pop('type')
        
        if obj_type not in self._module_dict:
            raise KeyError(f"'{obj_type}' is not registered in {self._name}. "
                           f"Available: {list(self._module_dict.keys())}")
        
        cls = self._module_dict[obj_type]
        return cls(**args)
MODELS = Registry('models')
DATASETS = Registry('datasets')
EVALUATORS = Registry('evaluators')
METRICS = Registry('metrics')
RUNNERS = Registry('runners')
SUMMARIZERS = Registry('summarizers')
ENVIRONMENTS = Registry('environments')

