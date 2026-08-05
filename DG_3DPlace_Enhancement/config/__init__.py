"""
Configuration module
"""
import os
import yaml


def load_config(config_path=None):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default_config.yaml
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__),
            'default_config.yaml'
        )
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config, output_path):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save config file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)


def merge_configs(base_config, overrides):
    """
    Merge override configuration into base configuration.
    
    Args:
        base_config: Base configuration dictionary
        overrides: Override configuration dictionary
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in overrides.items():
        if isinstance(value, dict) and key in merged:
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged
