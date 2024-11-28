import yaml
from pathlib import Path
from typing import Optional
from .config_classes import Config, DataConfig, TrainingConfig, SchedulerConfig, NetworkConfig, LossConfig


def deep_merge(dict1, dict2):
    """Recursively merge two dictionaries."""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def load_config(config_path: str, env: Optional[str] = None) -> Config:
    """Load configuration from YAML file."""
    config_dir = Path(config_path)
    
    # Load default config first
    default_config = yaml.safe_load((config_dir / 'default.yaml').read_text())
    
    # Load environment specific config if specified
    if env:
        env_config = yaml.safe_load((config_dir / f'{env}.yaml').read_text())
        # Merge configs, with env config taking precedence
        default_config = deep_merge(default_config, env_config)
    
    # Create config objects
    return Config(
        data=DataConfig(**default_config['data']),
        training=TrainingConfig(**default_config['training']),
        scheduler=SchedulerConfig(**default_config['scheduler']),
        network=NetworkConfig(**default_config['network']),
        loss=LossConfig(**default_config['loss'])
    )