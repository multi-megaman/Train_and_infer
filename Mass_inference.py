import SAN.Mass_inference as SAN_inference
import torch
import os
from datetime import datetime
import matplotlib.pyplot as plt

checkpointsPath = "./checkpoints" #Alterar essa variável
words = "./data/word.txt" #Alterar essa variável

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
    exp_rate, pred_time_mean, experiment =  SAN_inference.Make_inference(checkpointFolder=checkpointsFolder[x],
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
fig, ax = plt.subplots(figsize=(12,8))
plt.xlabel("exp_rate", size=12)
plt.ylabel("inference_time_mean", size=12)
plt.title("Inferences", size=15)

#ANOTA O NOME DOS MODELOS NOS PONTOS DO GRÁFICO
for n,experiment in enumerate(inferencesInfos):
    plt.plot(inferencesInfos[experiment]["exp_rate"],inferencesInfos[experiment]["time_mean"],'o')
    for i, modelName in enumerate(inferencesInfos[experiment]["model_name"]):
        ax.annotate(str(experiment) + " " + str(modelName), (inferencesInfos[experiment]["exp_rate"][i], inferencesInfos[experiment]["time_mean"][i]))

#SALVA O GRÁFICO
plt.savefig(str(actualDate) + '.png')
plt.show()


print(str(inferencesInfos))

