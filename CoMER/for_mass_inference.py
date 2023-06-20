from .comer.lit_comer import LitCoMER
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm
import os
from time import process_time
import numpy as np
from difflib import SequenceMatcher
import torch
from .comer.datamodule import vocab
import cv2

import timeit


def Make_inference(imagePath,labelPath,ckptPath,deviceName ="cpu",resize=None, printar=True):
    ckpt = ckptPath #carregar o modelo
    model = LitCoMER.load_from_checkpoint(ckpt,map_location=torch.device(deviceName))
    model = model.eval()
    # device = torch.device("cpu")
    # model = model.to(device)

    with open(labelPath) as f:
        labels = f.readlines()

    exp_rights= 0
    word_right=[]
    pred_time_mean = 0
    word_right_mean = 0
    pred_times =[]

    for item in tqdm(labels):
        name, *label = item.split()
        label = ' '.join(label)
        #if name.endswith('.jpg'):
        #    name = name.split('.')[0]
        # img_path = str(os.path.join(imagePath, name))

        # img = Image.open(img_path)

        # img = ToTensor()(img)

        # mask = torch.zeros_like(img, dtype=torch.bool)

        img = cv2.imread(os.path.join(imagePath, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if (resize):
            img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
            
        img = Image.fromarray(img)

        img = ToTensor()(img)

        #beam_size é um parâmetro que variamos com optuna: pode ser 10 (default) ou 5
        beam_size = 10

        #max_len e alpha a gente não mexe: usamos esses valores fixos
        max_len = 200
        alpha = 1.0

        mask = torch.zeros_like(img, dtype=torch.bool)#repare aqui que eu troquei para bool (comparei com o do CoMER e com a funcao collate_fn em datamodule.py)
        

        #medir tempo
        # pred_start = process_time()
        pred_start = timeit.default_timer()

        # hyp = model.beam_search(img)
        hyp = model.approximate_joint_search(img.unsqueeze(0), mask)[0]

        # pred_end = process_time()
        pred_end = timeit.default_timer()
        pred_time = pred_end-pred_start
        pred_times.append(pred_time)
         #medir tempo
        
        pred_latex = vocab.indices2label(hyp.seq)



        if pred_latex == label:
            exp_rights += 1

        word_right_ratio = SequenceMatcher(None,pred_latex,label,autojunk=False).ratio()
        word_right.append(word_right_ratio)

        if (printar):
            print("----------")
            print("Imagem: " + name)
            if pred_latex == label:
                print("Acertou!")
            else:
                print("Errou!")
            print("predi: "+pred_latex)
            print("label: "+label)
            print("word_right_ratio:" + str(word_right_ratio))
            print("Infer_time: " + str(pred_time))
    
    pred_time_mean = np.array(pred_times).mean()
    word_right_mean= np.array(word_right).mean()
    exp_right_mean = (exp_rights / len(labels))
    pred_time_std = np.array(pred_times).std()
    word_right_std = np.array(word_right).std()

    return exp_right_mean, pred_time_mean, word_right_mean, pred_time_std, word_right_std, deviceName, "CoMER"