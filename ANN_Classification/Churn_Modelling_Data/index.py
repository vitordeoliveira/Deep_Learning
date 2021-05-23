# Artificial Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import os.path

data = pd.read_csv('Churn_Modelling.csv')
X = data.iloc[:,3:-1]
Y = data.iloc[:,-1].to_frame()

print(X["Geography"].value_counts())

# Label enconding
# from sklearn.preprocessing import LabelEncoder
# lb = LabelEncoder()
# new_label = lb.fit_transform(X["Geography"])

# Label enconding
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
X.iloc[:,2] = lb.fit_transform(X.iloc[:,2])


# One Hot enconding
new_columns = pd.get_dummies(X['Geography'])
X = X.drop( ["Geography"] , axis = 1)
X = pd.DataFrame.join(new_columns, X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score
if(os.path.isfile('saved_model.pb')):
    ann = tf.keras.models.load_model('./')
    y_pred = ann.predict(X_test)
    y_pred = (y_pred > 0.5)
    test = Y_test.to_numpy()
    y_pred_test = np.concatenate((y_pred, test), axis=1)
    # # Making the Confusion Matrix
    cm = confusion_matrix(Y_test.to_numpy(), y_pred)
    print(cm)
    accuracy_score(Y_test.to_numpy(), y_pred)
else:
    # # Part 2 - Building the ANN
    # # Initializing the ANN
    ann = tf.keras.models.Sequential()
    
    # # Adding the input layer and the first hidden layer
    ann.add(tf.keras.layers.Dense(8, activation='relu'))
    
    # # Adding the second hidden layer
    ann.add(tf.keras.layers.Dense(8, activation='relu'))
    
    # # Adding the output layer
    ann.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    # # Part 3 - Training the ANN
    # # Compiling the ANN
    ann.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # # Training the ANN on the Training set
    ann.fit(X_train, Y_train, batch_size = 32, epochs = 50)
    
    # # Part 4 - Making the predictions and evaluating the model
    
    # # Predicting the Test set results
    y_pred = ann.predict(X_test)
    y_pred = (y_pred > 0.5)
    
    test = Y_test.to_numpy()
    y_pred_test = np.concatenate((y_pred, test), axis=1)
    
    ann.save('./')
    











