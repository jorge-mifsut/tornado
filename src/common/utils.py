import yaml


def get_yaml(path):
    with open(path) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise exc
    return config

def save_yaml(config, path):
    from omegaconf import DictConfig, OmegaConf
    # Convert OmegaConf to plain dict to avoid serializing Python object tags
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    with open(path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
