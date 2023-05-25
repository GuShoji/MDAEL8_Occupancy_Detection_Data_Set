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
    df = pd.read_csv(input_file,   
                     names = names)  
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


    

    # calcular o escore z e quantis


    #Medida de posição relativa
    for medida in medias + modas:
        df[medida + '_zscore'] = (df[medida] - df[medida].mean()) / df[medida].std()


    for medida in medias + modas:
        q25, q50, q75 = np.percentile(df[medida], [25, 50, 75])
        print("Quantis para a coluna", medida)
        print("Q1 (25%):", q25)
        print("Q2 (50%):", q50)
        print("Q3 (75%):", q75)
        plt.boxplot(df[medida])
        plt.title("Diagrama de Caixa para a coluna " + medida)
        plt.show()

    '''
    print("\n\n--------Posição Relativa--------")
    for medida in medias + modas:
        coluna = df.loc[:, medida]
        print(medida)
        for Q in range(0, 100, 25):
            print("Q",int(Q/25),": ", coluna.quantile(Q/100))
        print("--------------------------------")

        plt.boxplot(coluna)
        plt.title("Exemplo de Boxplot")
        plt.xlabel("Dados")
        plt.ylabel("Valores")
        plt.show() 
    '''
    
  

    #Medidas de Associação
    plt.figure()
    dfCorrelacao = df.loc[:,['Temperature','Humidity','Light','CO2','HumidityRatio','Occupancy']]
    sns.heatmap(dfCorrelacao.corr(), annot=True)
    plt.show()   

    '''
    ### medida de correlação
    x = df['Light']
    y = df['Temperature']
    corr = x.corr(y)

    plt.scatter(x, y)
    plt.title('Correlação de Pearson: {:.2f}'.format(corr))
    plt.xlabel('Coluna X')
    plt.ylabel('Coluna Y')
    plt.show() 
    '''
    


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