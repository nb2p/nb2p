import toml


def load(path: str):
    return toml.load(path)
