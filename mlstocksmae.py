import numpy as np
import pandas as pd
import math as maths
import random as rand
import tensorflow as tf
import logging
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

###Pulls the stock data from the spreadsheet
data = pd.read_excel(r'/Users/myleskornfeld/Desktop/MachineLearning/RMTStonks/techstocks.xlsx', sheet_name='TechStocks')
df = pd.DataFrame(data, columns= ['MSFT','APPL','NLFX','GOOG','TSLA','FB','FDS','INTC','IBM'])
msft = list(df["MSFT"])
appl = list(df["APPL"])
nlfx = list(df["NLFX"])
goog = list(df["GOOG"])
tsla = list(df["TSLA"])
fb   = list(df["FB"])
fds  = list(df["FDS"])
intc = list(df["INTC"])
ibm  = list(df["IBM"])

companies = [msft,appl,nlfx,goog,tsla, fb, fds, intc, ibm]


##Function that organizes all the data from the spreadsheet
##Amount of data is window long, then predicts the next value
##The data doesn't overlap so (1,2,3,4,5) goes to (1,2,3),(3,4,5) for example
##Not (1,2,3),(2,3,4),(3,4,5)
##Also removes all data that contains a "Nan" value
def non_overlapping_data(window):
    cleaned_data = []
    index_data = [] #dummy variable that we loop over
    for company in companies:
        for stock in range(0,len(company)):
            if maths.isnan(company[stock]):
                break
            elif len(index_data)<(window-1):
                index_data.append(company[stock])
            elif stock+1 >= len(company):
                index_data = []
                break
            else:
                #print(stock)
                index_data.append(company[stock])
                index_data.append(company[stock+1])
                cleaned_data.append(index_data)
                index_data = []
    remove_nan_list = []
    for i in range(0,len(cleaned_data)):
        for data in cleaned_data[i]:
            if maths.isnan(data):
                remove_nan_list.append(i)
    remove_nan = []
    for i in range(0,len(cleaned_data)):
        if i not in remove_nan_list:
            remove_nan.append(cleaned_data[i])
    cleaned_data = remove_nan
    stock_prices = []
    up_or_down = []
    for data in cleaned_data:
        new_data = data[:len(data)-1]
        if data[len(data)-1] > new_data[len(new_data) - 1]:
            #stock price increases -- 1 for going up
            up_or_down.append(np.array(1.0))
        else:
            #stock price decreases -- 0 for going down
            up_or_down.append(np.array(0.0))
        stock_prices.append(np.array(new_data))
    direction = np.array(up_or_down)

    features = np.array(stock_prices).reshape(-1,window)
    labels = np.array(direction)

    return features,labels

##Same thing as above, except now (1,2,3),(2,3,4),(3,4,5)
def overlapping_data(window):
    cleaned_data = []
    index_data = [] #dummy variable that we loop over
    for company in companies:
        for stock in range(0,len(company)):
            index_data = company[stock:stock+window+1]
            if len(index_data) == window+1:
                cleaned_data.append(index_data)


    remove_nan_list = []
    for i in range(0,len(cleaned_data)):
        for data in cleaned_data[i]:
            if maths.isnan(data):
                remove_nan_list.append(i)

    remove_nan = []
    for i in range(0,len(cleaned_data)):
        if i not in remove_nan_list:
            remove_nan.append(cleaned_data[i])
    cleaned_data = remove_nan
    stock_prices = []
    up_or_down = []
    for data in cleaned_data:
        #print(data)
        new_data = data[:len(data)-1]
        if data[len(data)-1] > new_data[len(new_data) - 1]:
            #stock price increases -- 1 for going up
            #print(data,"  ",new_data," ",1)
            up_or_down.append(np.array(1.0))
        else:
            #stock price decreases -- 0 for going down
            #print(data,"  ",new_data," ",0)
            up_or_down.append(np.array(0.0))
        stock_prices.append(np.array(new_data))
    direction = np.array(up_or_down)

    features = np.array(stock_prices).reshape(-1,window)
    labels = np.array(direction)

    return features,labels

#Takes an array and returns each value as a percentage
def average_list(list):
    out_value = []
    for i in range(0,len(list)):
        out_value.append(list[i]/sum(list))
    return out_value

#Averages list over lists of lists, ie the features
def average_many_lists(list_of_lists):
    output = []
    for i in list_of_lists:
        output.append(average_list(i))
    return output

epoch_num = 100
window = 10
data = non_overlapping_data(window)
features = np.array(average_many_lists(data[0]))
labels = data[1]

tf.random.set_seed(42)
np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=42, test_size=0.2)


##Produces a polot to help us determine the learning rate
def determine_lr(model, name, guess):
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: guess * 10**(epoch/30))
    optimizer = tf.keras.optimizers.SGD(lr=guess, momentum=0.9) #Stochastic gradient descent optimizer
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])
    history = model.fit(X_train, y_train, epochs=epoch_num, callbacks = lr_schedule)
    plt.semilogx(history.history["lr"], history.history["loss"])
    plt.xlabel("Learning Rate (log)")
    plt.ylabel("Loss")
    plt.title("Learning Rate Determination "+name)
    plt.show()

#Produces a summary of the model and returns the history
def run_model(model, lrate):
    optimizer = tf.keras.optimizers.SGD(lr=lrate, momentum=0.9) #Stochastic gradient descent optimizer
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])
    history = model.fit(X_train, y_train, epochs=epoch_num, validation_data=(X_test, y_test))
    model.summary()
    return history

#Produces a plot of the Mae and the validation_Mae for the given model
def mae_plots(model, lrate, name):
    history = run_model(model, lrate)
    plt.plot(list(range(0,epoch_num)),history.history["mae"], label = 'mae')
    plt.plot(list(range(0,epoch_num)),history.history["val_mae"], label = 'val_mae')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Data")
    plt.title("Mae and Validation_Mae – "+name)
    plt.show()

    plt.plot(list(range(0,epoch_num)),history.history["loss"], label = 'loss')
    plt.plot(list(range(0,epoch_num)),history.history["val_loss"], label = 'val_loss')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Data")
    plt.title("Loss and Validation_Loss – "+name)
    plt.show()

#Produces a plot of the loss and the val_loss for hte given model


model_alphago_lite = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units = 1, input_shape = [window]),
    tf.keras.layers.Dense(units = 64),
    tf.keras.layers.Dense(units = 32),
    tf.keras.layers.Dense(units = 16),
    tf.keras.layers.Dense(units = 8),
    tf.keras.layers.Dense(units = 4),
    tf.keras.layers.Dense(units = 2, activation='relu') #2 because up or down
])

model2 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation = "relu", input_shape = [window]),
    tf.keras.layers.Dense(10, activation = "relu"),
    tf.keras.layers.Dense(1)
])

modelrnn1 = tf.keras.models.Sequential([
    #winow 10, lr = 5e-4
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                            input_shape = [None]),
    tf.keras.layers.SimpleRNN(100, return_sequences = True),
    tf.keras.layers.SimpleRNN(100),
    tf.keras.layers.Dense(1)
    #tf.keras.layers.Lambda(lambda x: x*200.0)
])

#determine_lr(modelrnn1, "First RNN", 1e-6)
#run_model(modelrnn1, 5e-4)
mae_plots(modelrnn1, 5e-4, "MAE Demonstration")
