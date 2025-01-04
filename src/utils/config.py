import os
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = None) -> Dict[Any, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = 'config/data_transform.yaml'
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 处理环境变量
    if 'huggingface' in config and 'token' in config['huggingface']:
        token = config['huggingface']['token']
        if token.startswith('${') and token.endswith('}'):
            env_var = token[2:-1]
            config['huggingface']['token'] = os.getenv(env_var)
    
    return config
