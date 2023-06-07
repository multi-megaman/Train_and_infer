import pandas as pd
import matplotlib.pyplot as plt

import plots

def Make_plots(csvPath,savePath=None):
    csv= pd.read_csv(csvPath) 


    #CUDA X CPU
    plots.Cpu_x_Cuda_bar(csv,savePath=savePath) #não vai fazer o plot caso não tenham experimentos com o Cuda, ou seja, caso o computador não tenha GPU dedicada, não tem motivo de fazer o plot.

#Exemplo de plot apontando manualmente para um CSV salvo, sem a necessidade de rodar a inferência novamente.
#Make_plots('.\\inferResults\\07-06-2023 20-17-35\\07-06-2023 20-17-35.csv','.\\inferResults\\07-06-2023 20-17-35')