import yaml

def load_config(yaml_path):
    try:
        with open(yaml_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    except:
        print('try UTF-8 encoding')
        with open(yaml_path, 'r', encoding='UTF-8') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    return params