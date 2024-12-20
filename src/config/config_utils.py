import yaml
from pathlib import Path
from typing import Optional
# from .config_classes import Config, DataConfig, TrainingConfig, SchedulerConfig, NetworkConfig, LossConfig

# def deep_merge(dict1, dict2):
#     """Recursively merge two dictionaries."""
#     result = dict1.copy()
#     for key, value in dict2.items():
#         if key in result and isinstance(result[key], dict) and isinstance(value, dict):
#             result[key] = deep_merge(result[key], value)
#         else:
#             result[key] = value
#     return result

# def load_config(config_path: str, env: Optional[str] = None) -> Config:
#     """Load configuration from YAML file."""
#     config_dir = Path(config_path)
    
#     # Load environment specific config if specified
#     if not env:
#         config = yaml.safe_load((config_dir / 'default.yaml').read_text())
#     else:
#         config = yaml.safe_load((config_dir / f'{env}.yaml').read_text())
#         # Merge configs, with env config taking precedence
#         # default_config = deep_merge(default_config, env_config)
    
#     # Create config objects
#     return Config(
#         data=DataConfig(**config['data']),
#         training=TrainingConfig(**config['training']),
#         scheduler=SchedulerConfig(**config['scheduler']),
#         network=NetworkConfig(**config['network']),
#         loss=LossConfig(**config['loss'])
#     )

from omegaconf import OmegaConf

class DictToClass:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)
            
def load_config(config_path: str, env: Optional[str] = None, to_class: bool = True):
    config_path = Path(config_path)
    if not env:
        config = OmegaConf.load(config_path / 'default.yaml')
    else:
        config = OmegaConf.load(config_path / f'{env}.yaml')
    if not to_class:
        return config
    return DictToClass(config)
    

if __name__ == '__main__':
    config = load_config('default.yaml')
    print(config)