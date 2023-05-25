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
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

# Calculate distance between two points
def minkowski_distance(a, b, p=1):    
    # Store the number of dimensions
    dim = len(a)    
    # Set initial distance to 0
    distance = 0
    
    # Calculate minkowski distance using parameter p
    for d in range(dim):
        distance += abs(a[d] - b[d])**p
        
    distance = distance**(1/p)    
    return distance


def knn_predict(X_train, X_test, y_train, y_test, k, p):    
    # Make predictions on the test data
    # Need output of 1 prediction per test data point
    y_hat_test = []

    for test_point in X_test:
        distances = []

        for train_point in X_train:
            distance = minkowski_distance(test_point, train_point, p=p)
            distances.append(distance)
        
        # Store distances in a dataframe
        df_dists = pd.DataFrame(data=distances, columns=['dist'], 
                                index=y_train.index)
        
        # Sort distances, and only consider the k closest points
        df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]

        # Create counter object to track the labels of k closest neighbors
        counter = Counter(y_train[df_nn.index])

        # Get most common label of all the nearest neighbors
        prediction = counter.most_common()[0][0]
        
        # Append prediction to output list
        y_hat_test.append(prediction)
        
    return y_hat_test


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

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    


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
   
   
    # Separating out the features
    X_test = dfTest.loc[:, features].values
    print(X_test.shape)

    # Separating out the target
    y_test = dfTest.loc[:,[target]].values

    # Standardizing the features
    X_test = StandardScaler().fit_transform(X_test)
    
    #TRAINING
    # Separating out the features
    X_train = dfTrain.loc[:, features].values
    print(X_train.shape)

    # Separating out the target
    y_train = dfTrain.loc[:,[target]]

    # Standardizing the features
    X_train = StandardScaler().fit_transform(X_train)
    '''
    # STEP 1 - TESTS USING knn classifier write from scratch    
    # Make predictions on test dataset using knn classifier
    y_hat_test = knn_predict(X_train, X_test, y_train, y_test, k=5, p=2)

    # Get test accuracy score
    accuracy = accuracy_score(y_test, y_hat_test)*100
    f1 = f1_score(y_test, y_hat_test, average='macro')
    print("Acurracy K-NN from scratch: {:.2f}%".format(accuracy))
    print("F1 Score K-NN from scratch: {:.2f}%".format(f1))

    # Get test confusion matrix
    cm = confusion_matrix(y_test, y_hat_test)        
    plot_confusion_matrix(cm, dfTest.target, False, "Confusion Matrix - K-NN")      
    plot_confusion_matrix(cm, dfTest.target, True, "Confusion Matrix - K-NN normalized")  
    '''
    # STEP 2 - TESTS USING knn classifier from sk-learn
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_hat_test = knn.predict(X_test)

     # Get test accuracy score
    accuracy = accuracy_score(y_test, y_hat_test)*100
    f1 = f1_score(y_test, y_hat_test,average='macro')
    print("Acurracy K-NN from sk-learn: {:.2f}%".format(accuracy))
    print("F1 Score K-NN from sk-learn: {:.2f}%".format(f1))

    # Get test confusion matrix    
    cm = confusion_matrix(y_test, y_hat_test)        
    plot_confusion_matrix(cm, [0,1], False, "Confusion Matrix - K-NN sklearn")      
    plot_confusion_matrix(cm, [0,1], True, "Confusion Matrix - K-NN sklearn normalized" )  
    plt.show()


if __name__ == "__main__":
    main()