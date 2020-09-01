#!/usr/bin/env python
# coding: utf-8
#Jupyter notebook

#Bibliotecas utilizadas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
from sklearn.datasets import load_breast_cancer

#Dataset
cancer = load_breast_cancer() #Dataset de câncer de mama de Wisconsin
df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])

#Normalização dos dados
def normaliza(dataframe): #Função que normaliza o dataframe
    return (dataframe - dataframe.describe()['mean'])/dataframe.describe()['std'] #Retorna o dado subtraído da média
    #e dividido pelo desvio padrão
df_norm = df.apply(normaliza) #Aplica a função no dataframe

#Matriz de correlação
df_corr = pd.DataFrame(np.dot(df_norm.T, df_norm)/(df.iloc[:,0].count() - 1)) #Cálculo da matriz de correlação

#Cálculo dos autovalores e autovetores
autovet = np.linalg.eig(df_corr)[1] #Autovetores da matriz de correlação
autoval = np.linalg.eig(df_corr)[0] #Autovalores da matriz de correlação
#print(sorted(autoval, reverse=True)) #Print dos autovalores

#Componentes principais
pcas = np.dot(df_norm, autovet)

#Plot das componentes principais e variância explicada
print("Variância explicada dos eixos 1 e 2 = ", autoval[0]/autoval.sum(),autoval[1]/autoval.sum() )
print("Autovalores componentes 1 e 2 = ", autoval[0], autoval[1])
#print("Autovalores =", autoval) #Descomentar essa linha para printar todos os autovalores
plt.scatter(-pcas[:,0], -pcas[:,1], c=cancer['target'], cmap='plasma', label='Benigno',
            edgecolor='black', s=25) #Plot das componentes principais com a variável target pintada
plt.scatter(0,0,c='yellow',label='Maligno', alpha=0.8)
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.legend()
#plt.savefig('PCA_linear.png', dpi=150)

#Plot das variáveis com alta correlação
plt.scatter(df['radius error'], df['perimeter error'], c=cancer['target'], cmap='plasma', label='Benigno',
            edgecolor='black', s=25, alpha=0.9) #Plot das colunas com 0.55 de correlação
plt.scatter(0,0,c='yellow',label='Maligno', alpha=0.8)
plt.xlabel('Radius error')
plt.ylabel('Perimeter error')
plt.legend()
#plt.savefig('Corr_0_5.png', dpi=150)

#pd.concat([df, pd.DataFrame(cancer['target'])], axis=1).corr()[0] #Cálculo da correlação das variáveis com a variável target

#Plot dos ruídos
print("Autovalores 19 e 20 = ", autoval[19], autoval[20]) #Print dos menores autovalores
print("Percentual = ", autoval[19]/autoval.sum(), autoval[20]/autoval.sum()) #Variância explicada para os
#respectivos autovalores
plt.scatter(-pcas[:,19], -pcas[:,20], c=cancer['target'], cmap='plasma', label='Benigno',
            edgecolor='black', s=25) #Plot das componentes principais com a variável target pintada
plt.scatter(0,0,c='yellow', alpha=0.9, label='Maligno')
plt.xlabel('Componente principal 29') #Nome do eixo x
plt.ylabel('Componente principal 30') #Nome do eixo y
plt.legend()
#plt.savefig("Eixos_ruido.png", dpi=150)

#Kernel-PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import KernelCenterer
gamma = 10 #Constante do kernel gaussiano
df_dist = euclidean_distances(df, df, squared=True) #Calcula a distância euclidiana quadrática entre os pares de dados
df_gauss = np.exp(-gamma*df_dist) #Aplica a função gaussiana
df_cent = KernelCenterer().fit_transform(df_gauss) #Centraliza a matriz kernel
autovet_k = np.linalg.eig(Kc)[1] #Calcula os autovetores da matriz
autoval_k = np.linalg.eig(Kc)[0] #Calcula os autovalores
pcas_kernel = np.dot(autovet_k, df_gauss) #Produto entre os autovetores e o kernel para gerar os PCAS
plt.scatter(pcas_kernel[0], pcas_kernel[1], c=cancer['target'], cmap='plasma', label='benigno') #Plot dos principais PCAs por kernel gaussiano
plt.scatter(0,0,label='maligno', c='yellow', alpha=0.9)
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()
#plt.savefig("Kernel.png", dpi=150)