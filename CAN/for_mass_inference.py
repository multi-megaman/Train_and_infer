import os
import cv2
import argparse
import numpy as np
import torch
from time import process_time
import json
import pickle as pkl
from tqdm import tqdm
import time
import difflib

from .utils import load_config, load_checkpoint, compute_edit_distance
from .models.infer_model import Inference
from .dataset import Words
from .counting_utils import gen_counting_label

def Make_inference(checkpointFolder,wordsPath,configPath,checkpointPath,imagePath='data/Base_soma_subtracao/val/val_images',labelPath='data/Base_soma_subtracao/val/val_labels.txt', date= "12/12/2012 12:12:12.121212", device='cpu'):
    # parser = argparse.ArgumentParser(description='model testing')
    # parser.add_argument('--dataset', default='CROHME', type=str, help='数据集名称')
    # parser.add_argument('--image_path', default='datasets/CROHME/14_test_images.pkl', type=str, help='测试image路径')
    # parser.add_argument('--label_path', default='datasets/CROHME/14_test_labels.txt', type=str, help='测试label路径')
    # parser.add_argument('--word_path', default='datasets/CROHME/words_dict.txt', type=str, help='测试dict路径')

    # parser.add_argument('--draw_map', default=False)
    # args = parser.parse_args()

    if not configPath:
        print('请提供config yaml路径！')
        exit(-1)

    # if args.dataset == 'CROHME':
    #     config_file = 'config.yaml'

    """加载config文件"""
    params = load_config(configPath)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device
    # params['word_path'] = wordsPath
    print(wordsPath)
    words = Words(wordsPath)
    params['word_num'] = len(words)
    params['words'] = words
    params['word_path'] = wordsPath

    if 'use_label_mask' not in params:
        params['use_label_mask'] = False
    print(params['decoder']['net'])
    model = Inference(params, draw_map=False)
    model = model.to(device)

    params['checkpoint'] = checkpointPath
    load_checkpoint(model, None, params['checkpoint'])
    model.eval()

    with open(imagePath, 'rb') as f:
        images = pkl.load(f)

    with open(labelPath) as f:
        lines = f.readlines()

    line_right = 0
    e1, e2, e3 = 0, 0, 0
    bad_case = {}
    model_time = 0
    mae_sum, mse_sum = 0, 0

    with torch.no_grad():
        pred_times={}
        inferences_awnser={}
        pred_time_mean = 0

        #CRIAR PASTA
        inferences_directory = os.path.join(checkpointFolder,"inferences -" + str(date))
        if not os.path.exists(inferences_directory):
            os.makedirs(inferences_directory)
        for line in tqdm(lines):
            name, *labels = line.split()
            name = name.split('.')[0] if name.endswith('jpg') else name
            input_labels = labels
            labels = ' '.join(labels)
            img = images[name]
            #img = torch.Tensor(255-img) / 255
            img = torch.Tensor(img) / 255
            img = img.unsqueeze(0).unsqueeze(0)
            img = img.to(device)
            a = time.time()
            
            input_labels = words.encode(input_labels)
            input_labels = torch.LongTensor(input_labels)
            input_labels = input_labels.unsqueeze(0).to(device)

            #medir tempo
            pred_start = process_time()
            probs, _, mae, mse = model(img, input_labels, os.path.join(params['decoder']['net'], name))
            pred_end = process_time()
            
            pred_time = pred_end-pred_start
            pred_times[name]=pred_time
            #medir tempo

            mae_sum += mae
            mse_sum += mse
            model_time += (time.time() - a)


            prediction = words.decode(probs)
            prediction = prediction.replace(' eos', '')

            # print('{} => {}'.format(prediction,labels))  
            # for i,s in enumerate(difflib.ndiff(prediction, labels)):
            #     if s[0]==' ': continue
            #     elif s[0]=='-':
            #         print(u'Delete "{}" from position {}'.format(s[-1],i))
            #     elif s[0]=='+':
            #         print(u'Add "{}" to position {}'.format(s[-1],i))    
            # print()   

            if prediction == labels:
                line_right += 1
                inferences_awnser[name]=(prediction + " ---> V")
            else:
                inferences_awnser[name]=(prediction + " ---> X")
                bad_case[name] = {
                    'label': labels,
                    'predi': prediction
                }
                #print(name, prediction, labels)
            # print("PREDICAO:" + prediction)
            # print("LABEL:" + labels)
            # print("INREFENCE_AWNSER:" + str(inferences_awnser[name]))
            # print("line_right:" + str(line_right))

            distance = compute_edit_distance(prediction, labels)
            if distance <= 1:
                #print("\n\nACERTOU! " + name)
                e1 += 1
            if distance <= 2:
                e2 += 1
            if distance <= 3:
                e3 += 1

    pred_time_mean = np.array(list(pred_times.values())).mean()
    exp_rate = line_right / len(lines)
    with open(os.path.join(inferences_directory,"prediction times - mean "+str(pred_time_mean).replace(".",",")+"s.txt"),"w+", encoding='UTF8') as f:
        f.write(str(pred_times))
    f.close()
    with open(os.path.join(inferences_directory,"inferences - exp_rate- "+str(exp_rate).replace(".",",")+".txt"),"w+", encoding='UTF8') as g:
        g.write(str(inferences_awnser))
    g.close()

    print(f'model time: {model_time}')
    print(f'ExpRate: {line_right / len(lines)}')
    print(f'mae: {mae_sum / len(lines)}')
    print(f'mse: {mse_sum / len(lines)}')
    print(f'e1: {e1 / len(lines)}')
    print(f'e2: {e2 / len(lines)}')
    print(f'e3: {e3 / len(lines)}')

    with open(f'{params["decoder"]["net"]}_bad_case.json','w') as f:
        json.dump(bad_case,f,ensure_ascii=False)

    return exp_rate, pred_time_mean, params["experiment"]
