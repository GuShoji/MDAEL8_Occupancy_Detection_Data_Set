from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

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

    # Standardizing the features
    X_train = StandardScaler().fit_transform(X_train)
    normalizedDfTRAIN = pd.DataFrame(data = X_train, columns = features)
    normalizedDfTRAIN = pd.concat([normalizedDfTRAIN, dfTest[[target]]], axis = 1)
   
    undersample = RandomUnderSampler(sampling_strategy='majority')
    # fit and apply the transform
    X_train, y_train = undersample.fit_resample(X_train, y_train)

    

    clf = DecisionTreeClassifier(max_leaf_nodes=3)
    clf.fit(X_train, y_train)
    tree.plot_tree(clf)
    plt.show()
    
    predictions = clf.predict(X_test)
    print(predictions)
    
    result = clf.score(X_test, y_test)
    print('Acuraccy:')

    
    print(result)


if __name__ == "__main__":
    main()