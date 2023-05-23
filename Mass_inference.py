import SAN.for_mass_inference as SAN_inference
import CAN.for_mass_inference as CAN_inference
import torch
import os
from datetime import datetime
from utils import load_config
import matplotlib.pyplot as plt
import numpy as np

from adjustText import adjust_text

checkpointsPath = "./checkpoints" #Alterar essa variável
SanWords = "./data/word.txt"
CanWords = "./data/word_can.txt"
CanImagesPkl = "data/val_image.pkl"
SanImagesPath = "data/Base_soma_subtracao/val/val_images"
CanLabelPath = 'data/Base_soma_subtracao/val/val_labels_subset.txt'
SanLabelPath = 'data/Base_soma_subtracao/val/val_labels.txt'

words = SanWords
labelsPath = SanLabelPath

actualDate = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
actualDevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpointsFolder = [f.path for f in os.scandir(os.path.abspath(checkpointsPath)) if f.is_dir()]   #ex: ["C:/checkpoints/model_1","C:/checkpoints/model_2"]
checkpointsName = [os.path.basename(x) for x in checkpointsFolder]                                 #ex: ["model_1","model_2"]
checkpointsFile = [os.path.join(x,(os.path.basename(x)+".pth"))  for x in checkpointsFolder]       #ex: ["C:/checkpoints/model_1/model_1.pth","C:/checkpoints/model_2/model_2.pth"]
checkpointsConfig = [os.path.join(x,"config.yaml")  for x in checkpointsFolder]                    #ex: ["C:/checkpoints/model_1/config.yaml","C:/checkpoints/model_2/config.yaml"]
actualModelConfig = ""

inferencesInfos = {}

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


    exp_rate, pred_time_mean, experiment =  inferenceScript.Make_inference(checkpointFolder=checkpointsFolder[x],
                                            imagePath=ImagesPath,
                                            labelPath=labelsPath,
                                            configPath=checkpointsConfig[x],
                                            checkpointPath=checkpointsFile[x], 
                                            wordsPath=words,
                                            device=actualDevice,
                                            date=actualDate)
    
    if experiment in inferencesInfos:
        inferencesInfos[experiment]["exp_rate"].append(exp_rate)
        inferencesInfos[experiment]["time_mean"].append(pred_time_mean)
        inferencesInfos[experiment]["model_name"].append(checkpointsName[x])
    else:
        inferencesInfos[experiment] = {"exp_rate":[exp_rate],
                                       "time_mean": [pred_time_mean],
                                       "model_name": [checkpointsName[x]]}

#MOSTRA O GRÁFICO
fig, ax = plt.subplots(figsize=(20,20))
plt.xlabel("exp_rate", size=30)
plt.ylabel("inference_time_mean", size=30)
plt.xticks(np.arange(0, 1, 0.05))
plt.title("Inferences", size=25)

#ANOTA O NOME DOS MODELOS NOS PONTOS DO GRÁFICO
for n,experiment in enumerate(inferencesInfos):
    plt.plot(inferencesInfos[experiment]["exp_rate"],inferencesInfos[experiment]["time_mean"],'o')
    for i, modelName in enumerate(inferencesInfos[experiment]["model_name"]):
        ax.annotate(str(experiment) + " " + str(modelName), (inferencesInfos[experiment]["exp_rate"][i], inferencesInfos[experiment]["time_mean"][i]),fontsize=13)

# p1 = plt.plot(inferencesInfos[experiment]["exp_rate"],inferencesInfos[experiment]["time_mean"],color="black", alpha=0.5)
# anotacoes = []
# for n,experiment in enumerate(inferencesInfos):
#     for i, modelName in enumerate(inferencesInfos[experiment]["model_name"]):
#         anotacoes.append(plt.text(inferencesInfos[experiment]["exp_rate"],inferencesInfos[experiment]["time_mean"],(str(experiment) + " " + str(modelName), (inferencesInfos[experiment]["exp_rate"][i], inferencesInfos[experiment]["time_mean"][i]))))
# adjust_text(anotacoes, x=inferencesInfos[experiment]["exp_rate"], y=inferencesInfos[experiment]["time_mean"], autoalign='y',
#             only_move={'points':'y', 'text':'y'}, force_points=0.15,
#             arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

#SALVA O GRÁFICO
plt.savefig(str(actualDate) + '.png')
plt.show()


print(str(inferencesInfos))
