import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

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
    
    df['Temperature'].hist()
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

if __name__ == "__main__":
    main()