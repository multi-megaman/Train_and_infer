import os
import cv2
import argparse
import torch
import json
from tqdm import tqdm

from utils import load_config, load_checkpoint
from infer.Backbone import Backbone
from dataset import Words

parser = argparse.ArgumentParser(description='Spatial channel attention')
parser.add_argument('--config', default='./checkpoints/model_1/config.yaml', type=str, help='配置文件路径')
#parser.add_argument('--image_path', default='data/DataBase/test/test_images', type=str, help='测试image路径')
#parser.add_argument('--image_path', default='data/new_test/images', type=str, help='测试image路径')
parser.add_argument('--image_path', default='data/Base_soma_subtracao/test/test_images', type=str, help='')
#parser.add_argument('--label_path', default='data/DataBase/test/test_labels.txt', type=str, help='测试label路径')
#parser.add_argument('--label_path', default='data/new_test/labels.txt', type=str, help='测试label路径')
parser.add_argument('--label_path', default='data/Base_soma_subtracao/test/test_nossas_labels_sem_oversets_enlinhados_e_vazaios.txt', type=str, help='')
args = parser.parse_args()

if not args.config:
    print('请提供config yaml路径！')
    exit(-1)

"""加载config文件"""
params = load_config(args.config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
params['device'] = device

words = Words(params['word_path'])
params['word_num'] = len(words)
params['struct_num'] = 7
params['words'] = words

model = Backbone(params)
model = model.to(device)

# load_checkpoint(model, None, params['checkpoint'])
state = torch.load(params['checkpoint'], map_location='cpu')

model.load_state_dict(state['model'])

model.eval()

word_right, node_right, exp_right, length, cal_num = 0, 0, 0, 0, 0

with open(args.label_path) as f:
    labels = f.readlines()

def convert(nodeid, gtd_list):
    isparent = False
    child_list = []
    for i in range(len(gtd_list)):
        if gtd_list[i][2] == nodeid:
            isparent = True
            child_list.append([gtd_list[i][0],gtd_list[i][1],gtd_list[i][3]])
    if not isparent:
        return [gtd_list[nodeid][0]]
    else:
        if gtd_list[nodeid][0] == '\\frac':
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] == 'Above':
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Below':
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Right':
                    return_string += convert(child_list[i][1], gtd_list)
            for i in range(len(child_list)):
                if child_list[i][2] not in ['Right','Above','Below']:
                    return_string += ['illegal']
                    
        #TESTE--------------------------
        elif gtd_list[nodeid][0] == '\\overset':
            
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] == 'Sup':
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Below':
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Right':
                    return_string += convert(child_list[i][1], gtd_list)
            for i in range(len(child_list)):
                if child_list[i][2] not in ['Right','Sup','Below']:
                    return_string += ['illegal']
        #-------------------------------
        else:
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] in ['l_sup']:
                    return_string += ['['] + convert(child_list[i][1], gtd_list) + [']']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Inside':
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Sub','Below']:
                    return_string += ['_','{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Sup','Above']:
                    return_string += ['^','{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Right']:
                    return_string += convert(child_list[i][1], gtd_list)
        return return_string


with torch.no_grad():
    bad_case = {}
    for item in tqdm(labels):
        name, *label = item.split()
        label = ' '.join(label)
        #if name.endswith('.jpg'):
        #    name = name.split('.')[0]
        img = cv2.imread(os.path.join(args.image_path, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dim = (150, 150)
        img = cv2.resize(img, (dim), interpolation=cv2.INTER_AREA)

        image = torch.Tensor(img) / 255
        image = image.unsqueeze(0).unsqueeze(0)

        image_mask = torch.ones(image.shape)
        image, image_mask = image.to(device), image_mask.to(device)

        prediction = model(image, image_mask)

        latex_list = convert(1, prediction)
        latex_string = ' '.join(latex_list)

        # print(latex_string)
        # cv2.imshow('image', img)
        #
        # cv2.waitKey()


        if latex_string == label.strip():
            exp_right += 1
        else:
            bad_case[name] = {
                'label': label,
                'predi': latex_string,
                'list': prediction
            }

    print(exp_right / len(labels))

with open('bad_case.json', 'w') as f:
    json.dump(bad_case, f, ensure_ascii=False)














