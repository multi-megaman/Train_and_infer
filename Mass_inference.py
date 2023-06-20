import SAN.for_mass_inference as SAN_inference
import CAN.for_mass_inference as CAN_inference
import BTTR.for_mass_inference as BTTR_inference
import CoMER.for_mass_inference as COMER_inference

import torch
import os
from datetime import datetime
from utils import load_config, dele_sub_folders, make_csv
import matplotlib.pyplot as plt
import numpy as np

from gen_plots import Make_plots
# from adjustText import adjust_text

#Variáveis para alterar----------------------------------------
checkpointsPath   = "./checkpoints"                                                                     #Pasta que contém todas as pastas com os checkpoints e as configs (se houverem)

SanWords          = "./data_HME100K_sum_sub_7k_150x150/SAN/word.txt"                                    #word.txt do SAN
SanImagesPath     = "data_HME100K_sum_sub_7k_150x150/HME100K_sum_sub_7k/test/test_images"               #Diretório de onde estão as imagens de validação do SAN
SanLabelPath      = 'data_HME100K_sum_sub_7k_150x150/HME100K_sum_sub_7k/test/test_labels.txt'           #txt das labels de validação do CAN

CanWords          = "./data_HME100K_sum_sub_7k_150x150/CAN/word_can.txt"                                #word.txt do CAN
CanImagesPkl      = "data_HME100K_sum_sub_7k_150x150/CAN/val_image.pkl"                                 #pkl de imagens do CAN (ele faz inferência com um pkl)
CanLabelPath      = 'data_HME100K_sum_sub_7k_150x150/HME100K_sum_sub_7k/test/test_labels.txt'           #txt das labels de validação do CAN

BttrImagesPath    = "./data_Base_soma_subtracao/Base_soma_subtracao/test/test_images"                     #Diretório de onde estão as imagens de validação do BTTR
BttrLabelPath     = './data_Base_soma_subtracao/Base_soma_subtracao/test/test_labels.txt'                 #txt das labels de validação do BTTR

ComerImagesPath    = "./data_Base_soma_subtracao/Base_soma_subtracao/test/test_images"                     #Diretório de onde estão as imagens de validação do CoMER
ComerLabelPath     = './data_Base_soma_subtracao/Base_soma_subtracao/test/test_labels.txt'                 #txt das labels de validação do CoMER


printar           = True                                                                               #para printar informações mais detalhadas de cada inferência em cada imagem.
ResizeImg         = (150,150)                                                                           #(altura, largura) de redimensionamento das imagens. O valor padrão é (150,150). OBS: marcar como None para não redimensionar
words             = SanWords
labelsPath        = SanLabelPath

actualDate        = datetime.now().strftime("%d-%m-%Y %H-%M-%S")                                        #ex: 10-05-2022 10-25-43           
devices = ['cpu']                                                                                       #ex: ["cpu"] ou ["cpu","cuda"]
if torch.cuda.is_available():
    devices.append('cuda')

checkpointsFolder = [f.path for f in os.scandir(os.path.abspath(checkpointsPath)) if f.is_dir()]        #ex: ["C:/checkpoints/model_1","C:/checkpoints/model_2"]
checkpointsName   = [os.path.basename(x) for x in checkpointsFolder]                                    #ex: ["model_1","model_2"]
checkpointsFile   = []                                                                                  #ex: ["C:/checkpoints/model_1/model_1.pth","C:/checkpoints/model_2/model_2.ckpt"]
for i, folder in enumerate(checkpointsFolder):
    for file in os.listdir(folder):
        if file == os.path.basename(folder)+".pth" or file == os.path.basename(folder)+".ckpt":
            checkpointsFile.append(os.path.join(folder, file))

checkpointsConfig = [os.path.join(x,"config.yaml")  for x in checkpointsFolder]                         #ex: ["C:/checkpoints/model_1/config.yaml","C:/checkpoints/model_2/config.yaml"]

inferencesInfos   = {}
infosForCsv       = []

#FAZER A INFERÊNCIA DEPENDENDO DO ALGORITMO (SAN, CAN, ETC)
for x in range(len(checkpointsFolder)):
    # inferenceScript = None
    params = load_config(checkpointsConfig[x])
    experiment_name = params["experiment"]

    #SAN
    if experiment_name == "SAN":
        inferenceScript = SAN_inference
        words = SanWords
        labelsPath = SanLabelPath
        ImagesPath = SanImagesPath

        for device in devices: #para fazer inferências com GPU e sem GPU
            exp_rate, pred_time_mean, word_right,pred_time_std, word_right_std, device, experiment =  inferenceScript.Make_inference(checkpointFolder=checkpointsFolder[x],
                                                    imagePath=ImagesPath,
                                                    labelPath=labelsPath,
                                                    configPath=checkpointsConfig[x],
                                                    checkpointPath=checkpointsFile[x], 
                                                    wordsPath=words,
                                                    resize=ResizeImg,
                                                    deviceName=device,
                                                    date=actualDate,
                                                    printar=printar)
            infosForCsv.append({'experiment':experiment,'model_name':checkpointsName[x],'inference_time_mean_(seconds)':pred_time_mean,'inference_time_standard_deviation':pred_time_std,'expression_rate':exp_rate,'word_right_mean':word_right,'word_right_standard_deviation':word_right_std,'device': device})

    #CAN  
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
                                                    date=actualDate,
                                                    printar=printar)
            infosForCsv.append({'experiment':experiment,'model_name':checkpointsName[x],'inference_time_mean_(seconds)':pred_time_mean,'inference_time_standard_deviation':pred_time_std,'expression_rate':exp_rate,'word_right_mean':word_right,'word_right_standard_deviation':word_right_std,'device': device})

    #BTTR
    elif experiment_name == "BTTR":
        inferenceScript = BTTR_inference
        labelsPath = BttrLabelPath
        ImagesPath = BttrImagesPath

        for device in devices:
            exp_rate, pred_time_mean,word_right, pred_time_std, word_right_std, device, experiment = inferenceScript.Make_inference(
                                                    imagePath=ImagesPath,
                                                    labelPath=labelsPath,
                                                    ckptPath=checkpointsFile[x],
                                                    deviceName=device,
                                                    resize=ResizeImg,
                                                    printar=printar)
            infosForCsv.append({'experiment':experiment,'model_name':checkpointsName[x],'inference_time_mean_(seconds)':pred_time_mean,'inference_time_standard_deviation':pred_time_std,'expression_rate':exp_rate,'word_right_mean':word_right,'word_right_standard_deviation':word_right_std,'device': device})
            
    #CoMER
    elif experiment_name == "COMER":
        inferenceScript = COMER_inference
        labelsPath = BttrLabelPath
        ImagesPath = BttrImagesPath

        for device in devices:
            exp_rate, pred_time_mean,word_right, pred_time_std, word_right_std, device, experiment = inferenceScript.Make_inference(
                                                    imagePath=ImagesPath,
                                                    labelPath=labelsPath,
                                                    ckptPath=checkpointsFile[x],
                                                    deviceName=device,
                                                    resize=ResizeImg,
                                                    printar=printar)
            infosForCsv.append({'experiment':experiment,'model_name':checkpointsName[x],'inference_time_mean_(seconds)':pred_time_mean,'inference_time_standard_deviation':pred_time_std,'expression_rate':exp_rate,'word_right_mean':word_right,'word_right_standard_deviation':word_right_std,'device': device})
            
#SALVA A TABELA
results_directory = os.path.join( "./inferResults/", actualDate)
if not os.path.exists(results_directory):
    os.makedirs(results_directory)
# plt.savefig(os.path.join(results_directory, actualDate) + '.png')
csvPath = make_csv(infosForCsv,results_directory,actualDate)

#GERA OS GRÁFICOS
Make_plots(csvPath,results_directory,show=True)


