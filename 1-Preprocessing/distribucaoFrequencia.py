import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math as ma 

def main():
    # Faz a leitura do arquivo
    input_file = '0-Datasets/dataConjuntoClear.data'
    names = ['date','Temperature','Humidity','Light','CO2','HumidityRatio','Occupancy']
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas     
    df['date'] = df['date'].astype('datetime64[ns]')                   
    ShowInformationDataFrame(df,"Dataframe original")
    
    df['Temperature'] = df['Temperature'].astype(int)
    temperaturaFreq = df['Temperature'].value_counts()
    ShowFrequencia(temperaturaFreq,"Frequencia")


#LUZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ
    ampli = ma.ceil(df['Light'].max()-df['Light'].min())/5
    mini = df['Light'].min()
    #df = pd.DataFrame({'Light': [mini ]})
    valores=[mini, mini+ampli, mini+(ampli*2), mini+(ampli*3), mini+(ampli*4)]
    faixa=['0-339.6', '339.7-679.2', '679.3-1018.8', '1018.9-1358.4', '1358.5-1698.0']
    df['light_group'] = pd.cut(df['Light'], bins=[mini, mini+ampli, mini+(ampli*2), mini+(ampli*3), mini+(ampli*4), mini+(ampli*5)], include_lowest=True)
    ShowLightDataFrame(df, "Light")

    ShowLightDataFrame(df['light_group'].value_counts().sort_index(), "Light_group")
    plt.bar(faixa, df['light_group'].value_counts(), color="red")
    plt.xticks(faixa)
    plt.ylabel('Frequência')
    plt.xlabel('Niveis de Luz')
    plt.title('Destribuição de frequência para os niveis de Luz')
    plt.show()

#CO2222222222222222222222222222222222222222222222222222222
    mini = df['CO2'].min()
   
    ampli =(df['CO2'].max() - df['CO2'].min())/6
    #df = pd.DataFrame({'CO2':[mini, mini+ampli, mini+ampli*2, mini+ampli*3, mini+ampli*4, mini+ampli*5, mini+ampli*6]})
    df['CO2_group'] = pd.cut(df['CO2'], bins=[mini, mini+ampli, mini+ampli*2, mini+ampli*3, mini+ampli*4, mini+ampli*5, mini+ampli*6], include_lowest=True)
    ShowCO2Frequency(df, "CO2 tabela")


    freq = df['CO2_group'].value_counts().sort_index()
    ShowCO2Frequency(freq, "CO2 frequencia")

    intervalos = ['412.749-690.042', '690.042-967.333', '967.333-1244.625', '1244.625-1521.917', '1521.917-1799.208', '1799.208-2076.5']
    plt.bar(intervalos, freq, color="blue")
    plt.xticks(intervalos)
    plt.ylabel('Frequência')
    plt.xlabel('Niveis de CO2')
    plt.title('Destribuição de frequência para os niveis de C02')
    plt.show()

#TEMPERATURAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    mini = df['Temperature'].min()
   
    ampli =(df['Temperature'].max() - df['Temperature'].min())/6
    #df = pd.DataFrame({'CO2':[mini, mini+ampli, mini+ampli*2, mini+ampli*3, mini+ampli*4, mini+ampli*5, mini+ampli*6]})
    df['Temperature_group'] = pd.cut(df['Temperature'], bins=[mini, mini+ampli, mini+ampli*2, mini+ampli*3, mini+ampli*4, mini+ampli*5, mini+ampli*6], include_lowest=True)
    freTemp = df['Temperature_group'].value_counts().sort_index()
    ShowCO2Frequency(freTemp, "Temperature tabela")
    intervalos = ['18.999-19.901', '19.901-20.803', '20.803-21.704', '21.704-22.606', '22.606-23.507', '23.507-24.408']

    plt.bar(intervalos, freTemp, color="red")
    plt.xticks(intervalos)
    plt.ylabel('Frequência')
    plt.xlabel('Temperatura em Celsius')
    plt.title('Destribuição de frequência para temperatura')
    plt.show()  

def ShowInformationDataFrame(df, message=""):
    print(message+"\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")

def ShowFrequencia(df, message=""):
    print(message+"\n")
    print(df.head(20))
    print("\n")

def ShowLightDataFrame(df, message=""):
    print(message+"\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")

def ShowCO2Frequency(df, message = ""):
    print(message+"\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")


if __name__ == "__main__":
    main()