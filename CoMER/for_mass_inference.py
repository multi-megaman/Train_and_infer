from comer.lit_comer import LitCoMER
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm
import os
from time import process_time
import numpy as np
from difflib import SequenceMatcher
import torch
from comer.datamodule import vocab

import timeit


def Make_inference(imagePath,labelPath):
    ckpt = '../lightning_logs/version_0/checkpoints/epoch=259-step=97759.ckpt' #carregar o modelo
    model = LitCoMER.load_from_checkpoint(ckpt)
    model = model.eval()
    device = torch.device("cpu")
    model = model.to(device)

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

        #img = cv2.imread(os.path.join(imagePath, name))
        #img_path = '18_em_1.bmp'  #carregar UMA imagem
        img_path = str(os.path.join(imagePath, name))

        img = Image.open(img_path)

        img = ToTensor()(img)

        mask = torch.zeros_like(img, dtype=torch.bool)

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

        print(pred_latex)
        print(label)

        if pred_latex == label:
            exp_rights += 1

        word_right_ratio = SequenceMatcher(None,pred_latex,label,autojunk=False).ratio()
        word_right.append(word_right_ratio)
    
    pred_time_mean = np.array(pred_times).mean()
    word_right_mean= np.array(word_right).mean()
    exp_right_mean = (exp_rights / len(labels))

    return exp_right_mean, pred_time_mean,word_right_mean, "CoMER"