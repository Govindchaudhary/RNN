# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling(using normalisation technique)
#since we will be using sigmoid layer in our model
#hence we want our values in the range of 0 and 1 and thus we are using normalisation
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1)) #paramter is the default so actually no need to specify
training_set_scaled = sc.fit_transform(training_set)


X_train = training_set_scaled[0:1257]  #stock price at time t
y_train = training_set_scaled[1:1258]  #stock price at time t+1


# Reshaping
#reshaping means changing the format of the input datset ie X_train
#here we are converting the X_train from 2D to 3D
#np.reshape will reshape our structure 
#(ist argument is the structure we want to reshape ,2nd argument is the structure of our new format)
#2nd argument consists of batch_size=no.of obseravtions=1257,timestep=1 day(in our case)
#3rd one is input_dim is the no. of input features in this case we have only on which is stock price at time t
X_train = np.reshape(X_train, (1257, 1, 1))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense  # to create hidden and output layers 
from keras.layers import LSTM # a type of RNN with long short term memory
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()  #initializing our RNN model

# Adding the first LSTM layer and some Dropout regularisation
#here units is the no. of memory units(commom practice to use 4 memory units)
#here input_shape  is =(timesteps,no.of features that our input has)
#None indicates that we can accept any timesteps
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))



# Adding the output layer (units is the no. of neurons in output layer)
#since our output is the stock price at time t+1 so we have only 1 output
regressor.add(Dense(units = 1))

# Compiling the RNN
#here you have two choices for optimizer either adam or rmsprop both have almost same results
#rmsprop is recommended in RNN ,in this case we choose adam
#since we are considdering regression problem we won't take binary_crossentropy
#but rather we choose  mean_squared_error(=sum of suared diff between real and predicted price)
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 200, batch_size = 32)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
#we will start with jan1 we will predict stock price of jan2 based on the stock price of jan1 and so on


X_test = real_stock_price   #we are taking real stock price as input to predict the output
X_test = sc.transform(X_test)  #scaling the X_test

X_test = np.reshape(X_test, (20, 1, 1)) #reshaping our X_test to get 3D format
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price) #rescaling to get the original scale

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
