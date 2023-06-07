import SAN.for_mass_inference as SAN_inference
import CAN.for_mass_inference as CAN_inference
import BTTR.for_mass_inference as BTTR_inference

import torch
import os
from datetime import datetime
from utils import load_config, dele_sub_folders, make_csv
import matplotlib.pyplot as plt
import numpy as np

from gen_plots import Make_plots
# from adjustText import adjust_text

#Variáveis para alterar----------------------------------------
checkpointsPath   = "./checkpoints" #Pasta que contém todas as pastas com os checkpoints e as configs (se houverem)
SanWords          = "./data/SAN/word.txt" #word.txt do SAN
CanWords          = "./data/CAN/word_can.txt" #word.txt do CAN
CanImagesPkl      = "data/CAN/val_image.pkl" #pkl de imagens do CAN (ele faz inferência com um pkl)
SanImagesPath     = "data/Base_soma_subtracao/val/val_images"
CanLabelPath      = 'data/Base_soma_subtracao/val/val_labels.txt'
SanLabelPath      = 'data/Base_soma_subtracao/val/val_labels.txt'

words             = SanWords
labelsPath        = SanLabelPath

actualDate        = datetime.now().strftime("%d-%m-%Y %H-%M-%S")                                          #ex: 10-05-2022 10-25-43
actualDevice      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                        #ex: "CUDA ou "CPU"
devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')

checkpointsFolder = [f.path for f in os.scandir(os.path.abspath(checkpointsPath)) if f.is_dir()]   #ex: ["C:/checkpoints/model_1","C:/checkpoints/model_2"]
checkpointsName   = [os.path.basename(x) for x in checkpointsFolder]                                 #ex: ["model_1","model_2"]
# checkpointsFile   = [os.path.join(x,(os.path.basename(x)+".pth"))  for x in checkpointsFolder]       #ex: ["C:/checkpoints/model_1/model_1.pth","C:/checkpoints/model_2/model_2.pth"]

checkpointsFile   = []                                                                              #ex: ["C:/checkpoints/model_1/model_1.pth","C:/checkpoints/model_2/model_2.pth"]

for i, folder in enumerate(checkpointsFolder):
    for file in os.listdir(folder):
        if file == os.path.basename(folder)+".pth" or file == os.path.basename(folder)+".ckpt":
            checkpointsFile.append(os.path.join(folder, file))

checkpointsConfig = [os.path.join(x,"config.yaml")  for x in checkpointsFolder]                    #ex: ["C:/checkpoints/model_1/config.yaml","C:/checkpoints/model_2/config.yaml"]

inferencesInfos   = {}
infosForCsv       = []

#FAZER A INFERÊNCIA DEPENDENDO DO ALGORITMO (SAN, CAN, ETC)
for x in range(len(checkpointsFolder)):
    # inferenceScript = None
    params = load_config(checkpointsConfig[x])
    experiment_name = params["experiment"]

    if experiment_name == "SAN":
        inferenceScript = SAN_inference
        words = SanWords
        labelsPath = SanLabelPath
        ImagesPath = SanImagesPath
    elif experiment_name == "CAN":
        inferenceScript = CAN_inference
        words = CanWords
        labelsPath = CanLabelPath
        ImagesPath = CanImagesPkl

    for device in devices: #para fazer inferências com GPU e sem GPU

        exp_rate, pred_time_mean, word_right,pred_time_std, word_right_std, device, experiment =  inferenceScript.Make_inference(checkpointFolder=checkpointsFolder[x],
                                                imagePath=ImagesPath,
                                                labelPath=labelsPath,
                                                configPath=checkpointsConfig[x],
                                                checkpointPath=checkpointsFile[x], 
                                                wordsPath=words,
                                                device=device,
                                                date=actualDate)
        #PARA MONTAR O GRÁFICO
        # if experiment in inferencesInfos:
        #     inferencesInfos[experiment]["exp_rate"].append(exp_rate)
        #     inferencesInfos[experiment]["word_rate_std"].append(word_right_std)
        #     inferencesInfos[experiment]["time_mean_std"].append(pred_time_std)
        #     inferencesInfos[experiment]["word_rate_mean"].append(word_right)
        #     inferencesInfos[experiment]["time_mean"].append(pred_time_mean)
        #     inferencesInfos[experiment]["device"].append(device)
        #     inferencesInfos[experiment]["model_name"].append(checkpointsName[x])
        # else:
        #     inferencesInfos[experiment] = {"exp_rate":[exp_rate],
        #                                 "word_rate_mean": [word_right],
        #                                 "word_rate_std": [word_right_std],
        #                                 "time_mean": [pred_time_mean],
        #                                 "time_std": [pred_time_std],
        #                                 "device": [device],
        #                                 "model_name": [checkpointsName[x]]}
        
        #PARA MONTAR O CSV
        infosForCsv.append({'experiment':experiment,'model_name':checkpointsName[x],'inference_time_mean_(seconds)':pred_time_mean,'inference_time_standard_deviation':pred_time_std,'expression_rate':exp_rate,'word_right_mean':word_right,'word_right_standard_deviation':word_right_std,'device': device})

#MOSTRA O GRÁFICO
# fig, ax = plt.subplots(figsize=(20,20))
# plt.xlabel("exp_rate", size=30)
# plt.ylabel("inference_time_mean", size=30)
# plt.xticks(np.arange(0, 1, 0.05))
# plt.title("Inferences", size=25)

#ANOTA O NOME DOS MODELOS NOS PONTOS DO GRÁFICO
# for n,experiment in enumerate(inferencesInfos):
#     plt.plot(inferencesInfos[experiment]["exp_rate"],inferencesInfos[experiment]["time_mean"],'o')
#     for i, modelName in enumerate(inferencesInfos[experiment]["model_name"]):
#         ax.annotate(str(experiment) + " " + str(modelName), (inferencesInfos[experiment]["exp_rate"][i], inferencesInfos[experiment]["time_mean"][i]),fontsize=13)


#SALVA O GRÁFICO E A TABELA
results_directory = os.path.join( "./inferResults/", actualDate)
if not os.path.exists(results_directory):
    os.makedirs(results_directory)
# plt.savefig(os.path.join(results_directory, actualDate) + '.png')
csvPath = make_csv(infosForCsv,results_directory,actualDate)

Make_plots(csvPath,results_directory)


