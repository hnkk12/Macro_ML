import os
import yaml

def load_yaml(filepath: str) -> dict:
    """Load a YAML configuration file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_experiment_config(experiment_config_path: str) -> dict:
    """Load and merge experiment config with feature and model configs."""
    config = load_yaml(experiment_config_path)
    
    # Locate config folder relative to this file
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    features_path = os.path.join(base_dir, "configs", "features.yaml")
    features_config = load_yaml(features_path)
    config["feature_sets_definitions"] = features_config.get("feature_sets", {})
    
    models_path = os.path.join(base_dir, "configs", "models.yaml")
    models_config = load_yaml(models_path)
    config["models_definitions"] = models_config.get("models", {})
    
    return config
