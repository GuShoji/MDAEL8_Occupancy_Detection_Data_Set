# Initial imports
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    cm = np.round(cm, 2)
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    

"""def load_dataset(dataset='cancer'):        
    if dataset == 'iris':
        # Load iris data and store in dataframe
        iris = datasets.load_iris()
        names = iris.target_names
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['target'] = iris.target
    elif dataset == 'cancer':
        # Load cancer data and store in dataframe
        cancer = datasets.load_breast_cancer()
        names = cancer.target_names
        df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
        df['target'] = cancer.target
    
    print(df.head())
    return names, df
"""

def main():
     # Load iris data and store in dataframe
    input_fileTrain = '0-Datasets/original/datatrainingClear.data'
    input_fileTest = '0-Datasets/original/dataTesteClear.data'
    names = ['date','Temperature','Humidity','Light','CO2','HumidityRatio','Occupancy']
    features = ['Temperature','Humidity','Light','CO2','HumidityRatio']
    target = 'Occupancy'
    dfTrain = pd.read_csv(input_fileTrain,    # Nome do arquivo com dados
                     names = names) # Nome das colunas      
    dfTest = pd.read_csv(input_fileTest,    # Nome do arquivo com dados
                     names = names) # Nome das colunas                  
   
   
    
   #TRAINING
   
    # Separating out the features
    X_train = dfTrain.loc[:, features].values
    print(X_train.shape)

    # Separating out the target
    y_train = dfTrain.loc[:,[target]]


    over = RandomOverSampler(sampling_strategy='minority')
    # fit and apply the transform
    X_train, y_train = over.fit_resample(X_train, y_train)  

    #undersample = RandomUnderSampler(sampling_strategy='majority')
    # fit and apply the transform
    #X_train, y_train = undersample.fit_resample(X_train, y_train)
    

    # Standardizing the features
    norm = StandardScaler().fit(X_train)
    X_train = norm.transform(X_train)

    # Separating out the features
    X_test = dfTest.loc[:, features].values
    print(X_test.shape)

    # Separating out the target
    y_test = dfTest.loc[:,[target]].values

    # Standardizing the features
    X_test = norm.transform(X_test)
    
    
    
    # TESTS USING SVM classifier from sk-learn    
    svm = SVC(kernel='linear', C=10) # poly, rbf, linear
    # training using train dataset
    svm.fit(X_train, y_train)
    # get support vectors
    print(svm.support_vectors_)
    # get indices of support vectors
    print(svm.support_)
    # get number of support vectors for each class
    print("Qtd Support vectors: ")
    print(svm.n_support_)
    # predict using test dataset
    y_hat_test = svm.predict(X_test)
    y_hat_train = svm.predict(X_train)
     # Get test accuracy score
    accuracy = accuracy_score(y_test, y_hat_test)*100
    f1 = f1_score(y_test, y_hat_test,average='macro')
    print("Acurracy SVM from sk-learn: {:.5f}%".format(accuracy))
    print("F1 Score SVM from sk-learn: {:.5f}%".format(f1))

    # Get test confusion matrix    
    cm = confusion_matrix(y_test, y_hat_test) 
    cmt = confusion_matrix(y_train, y_hat_train)  
    plot_confusion_matrix(cmt, [0,1], False, "Confusion Matrix -TRAIN- SVM sklearn") 
    plot_confusion_matrix(cm, [0,1], False, "Confusion Matrix - SVM sklearn")      
    plot_confusion_matrix(cm, [0,1], True, "Confusion Matrix - SVM sklearn normalized" )  
    plt.show()


if __name__ == "__main__":
    main()