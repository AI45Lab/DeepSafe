import argparse
import sys
import os
import importlib.util
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Run evaluation from config.')
    parser.add_argument('config', help='Path to configuration file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    return parser.parse_known_args()

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    spec = importlib.util.spec_from_file_location("config", config_path)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    for name, value in vars(cfg_module).items():
        if isinstance(value, dict) and 'model' in value and 'dataset' in value:
            return value
            
    raise ValueError("Could not find a valid task configuration dict in the provided file. "
                     "Ensure you define a dict with 'model', 'dataset', and 'evaluator' keys.")

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    import uni_eval              
    from uni_eval.registry import RUNNERS              
    from uni_eval.cli.parse_eval_config import _set_nested             
    uni_eval.load_all_modules()

    args, unknown_args = parse_args()
    
    print(f"Loading config from {args.config}...")
    cfg = load_config(args.config)

    if unknown_args:
        print(f"Applying CLI overrides: {unknown_args}")
        i = 0
        while i < len(unknown_args):
            arg = unknown_args[i]
            if not arg.startswith("--"):
                i += 1
                continue
            
            key = arg.lstrip("-")
            val = ""
            
            if "=" in key:
                key, val = key.split("=", 1)
                i += 1
            else:
                if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("-"):
                    val = unknown_args[i + 1]
                    i += 2
                else:
                    val = ""                     
                    i += 1
            
            _set_nested(cfg, key, val)

    runner_type = cfg.get('runner', {'type': 'LocalRunner'})
    if isinstance(runner_type, str):
        runner_type = {'type': runner_type}
    print(f"Initializing runner: {runner_type['type']}...")
    runner = RUNNERS.build(runner_type)
    runner.run(cfg)

if __name__ == '__main__':
    main()
