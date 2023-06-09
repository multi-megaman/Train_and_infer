import os
import cv2
from datetime import datetime
import numpy as np
# import argparse
import torch
from time import process_time
import json
from tqdm import tqdm
  

from .utils import load_config, load_checkpoint
from .infer.Backbone import Backbone
from .dataset import Words
from .utils import updata_lr, Meter, cal_score
from .dataset import get_dataset

from difflib import SequenceMatcher



def Make_inference(checkpointFolder,wordsPath,configPath,checkpointPath,deviceName,imagePath='data/Base_soma_subtracao/val/val_images',labelPath='data/Base_soma_subtracao/val/val_labels.txt',resize=None, date= "12/12/2012 12:12:12.121212"):
    #parser = argparse.ArgumentParser(description='Spatial channel attention')
    #parser.add_argument('--config', default='./checkpoints/model_1/config.yaml', type=str, help='配置文件路径')
    #parser.add_argument('--image_path', default='data/DataBase/test/test_images', type=str, help='测试image路径')
    #parser.add_argument('--image_path', default='data/new_test/images', type=str, help='测试image路径')
    #parser.add_argument('--image_path', default='data/Base_soma_subtracao/val/val_images', type=str, help='')
    #parser.add_argument('--label_path', default='data/DataBase/test/test_labels.txt', type=str, help='测试label路径')
    #parser.add_argument('--label_path', default='data/new_test/labels.txt', type=str, help='测试label路径')
    #parser.add_argument('--label_path', default='data/Base_soma_subtracao/val/val_labels.txt', type=str, help='')
    #args = parser.parse_args()

    

    if not configPath:
        print('请提供config yaml路径！')
        exit(-1)

    """加载config文件"""
    params = load_config(configPath)


    device = torch.device(deviceName)
    # device = 'cpu'
    params['device'] = device

    #words = Words(params['word_path'])
    words = Words(wordsPath)
    params['word_num'] = len(words)
    params['struct_num'] = 7
    params['words'] = words
    params['word_path'] = wordsPath

    params['checkpoint'] = checkpointPath

    model = Backbone(params)
    model = model.to(device)

    # load_checkpoint(model, None, params['checkpoint'])
    state = torch.load(params['checkpoint'], map_location='cpu')

    model.load_state_dict(state['model'])

    model.eval()

    # train_loader, eval_loader = get_dataset(params)

    # loss_meter_eval, word_right_eval, struct_right_eval, exp_right_eval = eval(params= params,model= model)


    word_right, node_right, exp_right, length, cal_num = 0, 0, 0, 0, 0

    with open(labelPath) as f:
        labels = f.readlines()

    def convert(nodeid, gtd_list):
        print(gtd_list)
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
        pred_times={}
        word_right={}
        inferences_awnser={}
        pred_time_mean = 0
        word_right_mean = 0
        pred_time_std = 0
        word_right_std = 0

            
        for item in tqdm(labels):
            name, *label = item.split()
            print("-----------------")
            print("Imagem: " + str(name))
            label = ' '.join(label)
            #if name.endswith('.jpg'):
            #    name = name.split('.')[0]
            img = cv2.imread(os.path.join(imagePath, name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if resize:
                dim = resize
                img = cv2.resize(img, (dim), interpolation=cv2.INTER_AREA)

            image = torch.Tensor(img) / 255
            image = image.unsqueeze(0).unsqueeze(0)

            image_mask = torch.ones(image.shape)
            image, image_mask = image.to(device), image_mask.to(device)

            #medir tempo
            pred_start = process_time()
            prediction = model(image, image_mask)
            pred_end = process_time()
            
            pred_time = pred_end-pred_start
            pred_times[name]=pred_time
            #medir tempo

            latex_list = convert(1, prediction)
            latex_string = ' '.join(latex_list)

            
            # print(latex_string)
            # cv2.imshow('image', img)
            #
            # cv2.waitKey()

            if latex_string == label.strip():
                print("ACERTOU!")
                exp_right += 1
                inferences_awnser[name]=(latex_string + " ---> V")
            else:
                print("ERROU!")
                inferences_awnser[name]=(latex_string + " ---> X")
                bad_case[name] = {
                    'label': label,
                    'predi': latex_string,
                    'list': prediction
                }
            #Word_right-------------------------

            latex_prediction_list = latex_string.split() 
            label_list = label.strip().split()

            print("latex_prediction_list: " + str(latex_prediction_list))
            print("label_list: " + str(label_list))

            word_right_ratio = SequenceMatcher(None,latex_string,label.strip(),autojunk=False).ratio()
            print("word_right_ratio: " + str(word_right_ratio))

            word_right[name]=word_right_ratio
            #-----------------------------------

        pred_time_mean = np.array(list(pred_times.values())).mean()
        word_right_mean = np.array(list(word_right.values())).mean()
        exp_rate = exp_right / len(labels)
        pred_time_std = np.array(list(pred_times.values())).std()
        word_right_std = np.array(list(word_right.values())).std()
            

        #CRIAR PASTA
        # inferences_directory = os.path.join(checkpointFolder,"inferences -" + str(date))
        # if not os.path.exists(inferences_directory):
        #     os.makedirs(inferences_directory)
        # #print(str(inferences_awnser))
        # with open(os.path.join(inferences_directory,"prediction times - mean "+str(pred_time_mean).replace(".",",")+"s.txt"),"w+", encoding='UTF8') as f:
        #     f.write(str(pred_times))
        # f.close()
        # with open(os.path.join(inferences_directory,"inferences - exp_rate- "+str(exp_rate).replace(".",",")+".txt"),"w+", encoding='UTF8') as g:
        #     g.write(str(inferences_awnser))
        # g.close()

    with open('bad_case.json', 'w') as f:
        json.dump(bad_case, f, ensure_ascii=False)

    return exp_rate, pred_time_mean, word_right_mean, pred_time_std, word_right_std, deviceName, params["experiment"]














