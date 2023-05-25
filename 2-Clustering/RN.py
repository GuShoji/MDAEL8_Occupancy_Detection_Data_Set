from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

def main():
    input_fileTrain = '0-Datasets/original/datatrainingClear.data'
    input_fileTest = '0-Datasets/original/dataTesteClear.data'
    names = ['date','Temperature','Humidity','Light','CO2','HumidityRatio','Occupancy']
    features = ['Temperature','Humidity','Light','CO2','HumidityRatio']
    target = 'Occupancy'
    dfTrain = pd.read_csv(input_fileTrain,    # Nome do arquivo com dados
                     names = names) # Nome das colunas      
    dfTest = pd.read_csv(input_fileTest,    # Nome do arquivo com dados
                     names = names) # Nome das colunas                  
   
    target_names = ['0','1']

    #TESTE 
    # Separating out the features
    X_test = dfTest.loc[:, features].values
    print(X_test.shape)

    # Separating out the target
    y_test = dfTest.loc[:,[target]].values

    

    # Standardizing the features
    X_test = StandardScaler().fit_transform(X_test)
    normalizedDfTESTE = pd.DataFrame(data = X_test, columns = features)
    normalizedDfTESTE = pd.concat([normalizedDfTESTE, dfTest[[target]]], axis = 1)
    
    #TRAINING
    # Separating out the features
    X_train = dfTrain.loc[:, features].values
    print(X_train.shape)

    # Separating out the target
    y_train = dfTrain.loc[:,[target]].values

    #over = RandomOverSampler(sampling_strategy='minority')
    # fit and apply the transform
    #X_train, y_train = over.fit_resample(X_train, y_train)
    undersample = RandomUnderSampler(sampling_strategy='majority')
    #fit and apply the transform
    X_train, y_train = undersample.fit_resample(X_train, y_train)

    # Standardizing the features
    X_train = StandardScaler().fit_transform(X_train)

    # Criar uma instância do MLPClassifier
    #SOLVER='adam'
    #shape (500,25,25) = 0.7826367077393895
    #shape (10,10,10) = 0.9194652492550536
    #shpae (5, 5, 5, 5, 5, 5, 5, 5) = 0.879761617137795
    #shape (10,10,10,10) = 0.9177740194894097
    #shape (10,5,10,5) = 0.9059354111299025
    #solver = Sgd
    #Shape = (10,10,10,10) = 0.9200289925102682
    #shape = (10,10,10) = 0.9279213980832729
    #sgd>>adam>>lbfgs
    clf = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=10000, shuffle=False, random_state=1, solver='sgd', learning_rate= 'adaptive')

    # Treinar o modelo
    clf.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste
    predictions = clf.predict(X_test)
    print("PREDICT:", predictions)
    # Avaliar o modelo
    accuracy = clf.score(X_test, y_test)
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    main()