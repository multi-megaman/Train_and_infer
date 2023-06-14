import pandas as pd
import matplotlib.pyplot as plt
import os

import numpy as np

def Cpu_x_Cuda_scatter(csv, savePath= None, show=True):

    experiment_rows = {} #Separa o CSV por experimentos, ex: {SAN:[lista_de_runs],CAN:[lista_de_runs]}
    for x in csv.iterrows():
        if x[1]['experiment'] in experiment_rows:
            experiment_rows[x[1]['experiment']].append(x[1])
        else:
            experiment_rows[x[1]['experiment']] = [x[1]]

    x = [] #Tempo de inferência em CPU
    y = [] #Tempo de inferência em CUDA
    labels = []
    for experiment in experiment_rows:                              #experimentos, ex: CAN, SAN
        for run in range(len(experiment_rows[experiment])):         #runs, ex: CAN_model_5, SAN_model_2
            # print(experiment_rows[experiment][run]['experiment'])
            if experiment_rows[experiment][run]['device'] == 'cpu':
                # print("cpu")
                x.append(experiment_rows[experiment][run]['inference_time_mean_(seconds)'])
            else:
                # print("cuda")
                y.append(experiment_rows[experiment][run]['inference_time_mean_(seconds)'])
            if experiment_rows[experiment][run]['model_name'] not in labels:
                labels.append(experiment_rows[experiment][run]['model_name'])
                #caso não tenham runs com Cuda, não será possivel fazer o plot
        
        if not y:
            return -1
        
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.scatter(x, y)
        ax.set_xlabel('CPU')
        ax.set_ylabel('CUDA')
        ax.set_title(str(experiment) + " CPU x CUDA inference time (seconds)")

        for i, txt in enumerate(labels):
            ax.annotate(txt, xy=(x[i], y[i]), xytext=(x[i], y[i]), ha='center', fontsize = 8, rotation = 15)
        
        
        if (savePath):
            file = os.path.join(savePath, (str(experiment)+'_CPUvsGPU.png'))
            if os.path.exists(file):
                os.remove(file)
            plt.savefig(file)
        if show:
            plt.show() 

def Cpu_x_Cuda_bar(csv, savePath= None,show=True):

    experiment_rows = {} #Separa o CSV por experimentos, ex: {SAN:[lista_de_runs],CAN:[lista_de_runs]}
    for x in csv.iterrows():
        if x[1]['experiment'] in experiment_rows:
            experiment_rows[x[1]['experiment']].append(x[1])
        else:
            experiment_rows[x[1]['experiment']] = [x[1]]


    for experiment in experiment_rows:                              #experimentos, ex: CAN, SAN
        print(experiment)
        x = [] #Tempo de inferência em CPU
        y = [] #Tempo de inferência em CUDA
        labels = []
        for run in range(len(experiment_rows[experiment])):         #runs, ex: CAN_model_5, SAN_model_2
            # print(experiment_rows[experiment][run]['experiment'])
            if experiment_rows[experiment][run]['device'] == 'cpu':
                # print("cpu")
                x.append(experiment_rows[experiment][run]['inference_time_mean_(seconds)'])
            else:
                # print("cuda")
                y.append(experiment_rows[experiment][run]['inference_time_mean_(seconds)'])
            if experiment_rows[experiment][run]['model_name'] not in labels:
                labels.append(experiment_rows[experiment][run]['model_name'])

        #caso não tenham runs com Cuda ou CPU, não será possivel fazer o plot
        if not y or not x:
            print("Não foi possivel fazer o plot CPU x CUDA do " + str(experiment) + " pois é necessário ter feito as inferências com CPU e GPU e está faltando um dos dois.")
            return -1

        #----PLOT--------------------------------------------
        # plt.rcParams['font.size'] = 6
        fig, ax = plt.subplots(layout='constrained',figsize=(15,10))
        width = 0.25  # largura da barra
        multiplier = 0
        graph_infos = {
            "CPU": x,
            "CUDA": y
        }

        for index in range(len(labels)):
            print(str(labels[index]) + " CPU: " + str(x[index]) + " CUDA: " + str(y[index]))

        x_coord = np.arange(len(labels))  # the label locations
        for name, values in graph_infos.items():
            offset = width * multiplier 
            rects = ax.bar(x_coord + offset, tuple(values), width, label=name)
            ax.bar_label(rects, padding=2)
            multiplier += 1

        # Adicionando as labels e customizando o gráfico.
        ax.set_ylabel('Seconds')
        ax.set_title(str(experiment) + " CPU x CUDA inference time (seconds)")
        ax.set_xticks(x_coord + width, labels)
        plt.xticks(rotation = 10)
        # ax.legend(loc='upper left', ncols=2)
        ax.legend(loc='best', ncols=2)
        # ax.set_ylim(0, 250)

        if (savePath):
            file = os.path.join(savePath, (str(experiment)+'_CPUvsGPU_time.png'))
            if os.path.exists(file):
                os.remove(file)
            plt.savefig(file)

        if show:
            plt.show() 
        #------------------------------------------------

def top_x_models(csv,bestQnt, savePath = None,show=True): #plot que pega os x melhores modelos e plota todos juntos comparando o expRate deles com o inference_time_mean (CPU e\ou GPU)
    device_rows = {} #Separa o CSV por CPU e CUDA, ex: {cpu:[lista_de_runs],cuda:[lista_de_runs]}
    for x in csv.iterrows():
        if x[1]['device'] in device_rows:
            device_rows[x[1]['device']].append(x[1])
        else:
            device_rows[x[1]['device']] = [x[1]]
    
    # print(device_rows)

    for device in device_rows: #devices, ex: cpu, cuda
        # print(device)
        # print(device)
        x = [] #Tempo de inferência em CPU
        y = [] #Tempo de inferência em CUDA
        labels = []

        experiment_rows = {} #Separa o CSV por experimentos dentro do device, ex: {SAN:[lista_de_runs],CAN:[lista_de_runs]}
        for device_run in device_rows[device]:

            if device_run['experiment'] in experiment_rows:
                experiment_rows[device_run['experiment']].append(device_run)
            else:
                experiment_rows[device_run['experiment']] = [device_run]


        for experiment in experiment_rows: #Serão capturados os x melhores modelos dentro dos experimentos, ex: CAN, SAN

                runs_dataframe = pd.DataFrame.from_dict(experiment_rows[experiment])
                runs_dataframe = runs_dataframe.sort_values(by=['expression_rate'],ascending=False).head(bestQnt) #pegando apenas as x primeiras colunas de cada experimento em ordem decrescente (ou seja, as x melhores inferências baseadas no exp_rate')

                # print(runs_dataframe)
                for run in runs_dataframe.iterrows():
                    x.append(run[1]['expression_rate'])
                    y.append(run[1]['inference_time_mean_(seconds)'])
                    labels.append(str(run[1]['experiment']) +" "+ str(run[1]['model_name']))

        
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.scatter(x, y)
        ax.set_xlabel('Expression_rate')
        ax.set_ylabel('inference_time_mean(seconds)')
        ax.set_title(str(device) + " inference Time vs Expression Rate")

        for i, txt in enumerate(labels):
            ax.annotate(txt, xy=(x[i], y[i]), xytext=(x[i], y[i]), ha='center', fontsize = 8, rotation = 15)
        
        
        if (savePath):
            file = os.path.join(savePath, (str(device)+'_inference_Time_vs_expRate.png'))
            if os.path.exists(file):
                os.remove(file)
            plt.savefig(file)
        if show:
            plt.show() 
