import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # Faz a leitura do arquivo
    input_file = '0-Datasets/dataConjuntoClear.data'
    names = ['date','Temperature','Humidity','Light','CO2','HumidityRatio','Occupancy']
    medias = ['Temperature','Humidity','Light','CO2']
    modas = ['HumidityRatio']
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas     
    df['date'] = df['date'].astype('datetime64[ns]')                   
    ShowInformationDataFrame(df,"Dataframe original")
    
    

    #Medidas de tendencia central
    print("Media:")
    print(df['Temperature'].mean()) #Média
    print("Mediana:")
    print(df['Temperature'].median()) #Mediana
    print("Ponto medio")
    print((df['Temperature'].max() + df['Temperature'].min())/2)
    print("Moda:")
    print(df['Temperature'].mode()) #Moda

    #Medidas de dispersão
    print("Amplitude:")
    print(df['Temperature'].max() - df['Temperature'].min()) #Amplitude
    print("Desvio padrão:")
    print(df['Temperature'].std()) #Desvio padrão
    print("Variancia:")
    print(df['Temperature'].var()) #Variancia
    print("Cofienciente de variação")
    print((df['Temperature'].std()/df['Temperature'].mean())*100)


    print("\n\n--------Posição Relativa--------")
    for medida in medias + modas:
        coluna = df.loc[:, medida]
        print(medida)
        for Q in range(0, 100, 25):
            print("Q",int(Q/25),": ", coluna.quantile(Q/100))
        print("--------------------------------")

    #Medidas de Associação
    plt.figure()
    dfCorrelacao = df.loc[:,['Temperature','Humidity','Light','CO2','HumidityRatio','Occupancy']]
    sns.heatmap(dfCorrelacao.corr(), annot=True)
    plt.show()   

def ShowInformationDataFrame(df, message=""):
    print(message+"\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")

def ShowInformation(df, message=""):
    print(message+"\n")
    print(df.head(20))
    print("\n")

if __name__ == "__main__":
    main()