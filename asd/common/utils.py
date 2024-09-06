import yaml


def load_config(config_file=None):
    assert config_file is not None, "Error: config file not found."
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

def parse_args(args_file):
    config = load_config(config_file=args_file)
    return config