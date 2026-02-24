import sys
from ruamel.yaml import YAML


yaml = YAML(pure="true", typ="safe")
yaml.default_flow_style = False


def is_notebook():
    """Checks to see if the python shell is notebook-based."""
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        return False
    except (NameError, ImportError):
        return False


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def load_yaml(fn):
    """Load a yaml file."""
    with open(fn, "r") as f:
        data = yaml.load(f)
    return data


def dump_yaml(fn, data):
    """Dump data to a yaml file."""
    with open(fn, "w") as f:
        yaml.dump(data, f)
