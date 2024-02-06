import os

datasets = {}


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    
    return decorator


def make(name, **kwargs):
    if kwargs.get('root_path') is None:
        DATA_PATH = os.getenv('DATA_PATH')
        kwargs['root_path'] = os.path.join(DATA_PATH, name)
    dataset = datasets[name](**kwargs)
    return dataset
