import yaml
import csv
import os

def load_config(yaml_path):
    try:
        with open(yaml_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    except:
        print('try UTF-8 encoding')
        with open(yaml_path, 'r', encoding='UTF-8') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    return params

def dele_sub_folders(folderPath):
    return 0

def make_csv(infos,directory,fileName):
    print(infos)
    keys = infos[0].keys()
    with open(os.path.join(directory,fileName)+'.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(infos)
    f.close()
    return infos
