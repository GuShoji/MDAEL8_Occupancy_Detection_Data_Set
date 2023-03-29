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
    
    ampli = ma.ceil(df['Light'].max()-df['Light'].min())/5
    mini = df['Light'].min()
    #df = pd.DataFrame({'Light': [mini ]})
    valores=[mini, mini+ampli, mini+(ampli*2), mini+(ampli*3), mini+(ampli*4)]
    faixa=['0-339.6', '339.7-679.2', '679.3-1018.8', '1018.9-1358.4', '1358.5-1698.0']
    df['light_group'] = pd.cut(df['Light'], bins=[mini, mini+ampli, mini+(ampli*2), mini+(ampli*3), mini+(ampli*4), mini+(ampli*5)], include_lowest=True)
    ShowLightDataFrame(df, "Light")

    ShowLightDataFrame(df['light_group'].value_counts().sort_index(), "Light_group")
    plt.bar(faixa, df['light_group'].value_counts(), color="red")
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


if __name__ == "__main__":
    main()