
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from keras.layers import Dense, Input, concatenate, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop

def main(X_train, y_train, X_test, y_test):
    #Set hyperparameters
    num_hidden_neurons = 50 # number of neurons in hidden layer/s
    dropout = 0.25 # percentage of weights dropped out before softmax output (this prevents overfitting)
    epochs = 100 # number of epochs (complete training episodes over the training set) to run
    batch = 20 # mini batch size for better convergence
    #
    #Format data
    scaler = MinMaxScaler()
    one_hot = OneHotEncoder(categories="auto") # one hot encode the target classes
    X_train = scaler.fit_transform(X_train) #use one-hot encoding for categorical variables in the future
    X_test = scaler.fit_transform(X_test) #use one-hot encoding for categorical variables in the future
    y_train = one_hot.fit_transform(np.reshape(np.array(y_train), (-1,1)) ).toarray()
    y_test = one_hot.fit_transform(np.reshape(np.array(y_test), (-1,1)) ).toarray()
    #
    #Create network
    model = Sequential()
    model.add(Dense(num_hidden_neurons, input_shape = (X_train.shape[1], ), kernel_initializer="lecun_uniform", activation = "relu"))
    model.add(Dropout(dropout))
    #model.add(Dense(num_hidden_neurons, kernel_initializer="lecun_uniform", activation = "relu"))
    #model.add(Dropout(dropout))
    #output layer predicts probability of one of two outcomes (search/book)
    model.add(Dense(2, kernel_initializer="lecun_uniform", activation = "softmax"))
    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)
    #
    #Train the model
    model.fit(X_train, y_train, validation_data=[ [X_test], y_test],
              epochs=epochs, batch_size=batch)
    #
    #Predict for test set
    y_prob=model.predict_proba(X_test)
    #
    return([model, y_prob])

