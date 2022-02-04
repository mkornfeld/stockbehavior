import numpy as np
import pandas as pd
import math as maths
import random as rand
import tensorflow as tf
import logging
import matplotlib.pyplot as plt
import math
import time
from sklearn.model_selection import train_test_split
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

###Pulls the stock data from the spreadsheet
data = pd.read_excel(r'/Users/myleskornfeld/Desktop/MachineLearning/RMTStonks/techstocks.xlsx', sheet_name='TechStocks')
df = pd.DataFrame(data, columns= ['MSFT','APPL','NLFX','GOOG','TSLA','FB','FDS','INTC','IBM',
                                  'AMZN','SQ','ADBE','TWTR','NVDA'])
msft = list(df["MSFT"])
appl = list(df["APPL"])
nlfx = list(df["NLFX"])
goog = list(df["GOOG"])
tsla = list(df["TSLA"])
fb   = list(df["FB"])
fds  = list(df["FDS"])
intc = list(df["INTC"])
ibm  = list(df["IBM"])
amzn   = list(df["AMZN"])
sq  = list(df["SQ"])
adbe = list(df["ADBE"])
twtr  = list(df["TWTR"])
nvda  = list(df["NVDA"])

companies = [msft,appl,nlfx,goog,tsla, fb, fds, intc, ibm, amzn, sq, adbe, twtr, nvda]


##Function that organizes all the data from the spreadsheet
##Amount of data is window long, then predicts the next value
##The data doesn't overlap so (1,2,3,4,5) goes to (1,2,3),(3,4,5) for example
##Not (1,2,3),(2,3,4),(3,4,5)
##Also removes all data that contains a "Nan" value
##For accuracy, made 0.5 going up because values can only be between [0,1)
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
            #stock price increases -- 0.5 for going up
            up_or_down.append(np.array([1.0]))
        else:
            #stock price decreases -- 0 for going down
            up_or_down.append(np.array([0.0]))
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
            #stock price increases -- 0.5 for going up
            #print(data,"  ",new_data," ",1)
            up_or_down.append(np.array([1.0]))
        else:
            #stock price decreases -- 0 for going down
            #print(data,"  ",new_data," ",0)
            up_or_down.append(np.array([0.0]))
        stock_prices.append(np.array(new_data))
    direction = np.array(up_or_down)

    features = np.array(stock_prices).reshape(-1,window)
    labels = np.array(direction)

    return features,labels

#Averages list over lists of lists, ie the features
def average_many_lists(list_of_lists):
    output = []
    for i in list_of_lists:
        output.append(i/sum(i))
    return output

epoch_num = 100
window = 10
data = overlapping_data(window)
features = np.array(average_many_lists(data[0]))
labels = data[1]

tf.random.set_seed(42)
np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=42, test_size=0.2)

print("X_train",X_train.shape,X_train)
print("y_train",y_train.shape,y_train)

##Produces a polot to help us determine the learning rate
def determine_lr(model, name, guess):
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: guess * 10**(epoch/15))
    optimizer = tf.keras.optimizers.SGD(lr=guess, momentum=0.9) #Stochastic gradient descent optimizer
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epoch_num, callbacks = lr_schedule)
    plt.semilogx(history.history["lr"], history.history["loss"])
    plt.xlabel("Learning Rate (log)")
    plt.ylabel("Loss")
    plt.title("Learning Rate Determination "+name)
    plt.show()

#Produces a summary of the model and returns the history
def run_model(model, lrate):
    optimizer = tf.keras.optimizers.SGD(lr=lrate, momentum=0.9) #Stochastic gradient descent optimizer
    model.compile(optimizer = 'adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    print('data is compiled')
    history = model.fit(x=X_train, y=y_train, epochs=epoch_num, validation_data=(X_test, y_test))
    model.summary()
    return history

#Produces a plot of the Mae and the validation_Mae for the given model
def accuracy_plots(model, name, lrate):
    history = run_model(model, lrate)
    plt.plot(list(range(0,epoch_num)),history.history["accuracy"], label = 'accuracy')
    plt.plot(list(range(0,epoch_num)),history.history["val_accuracy"], label = 'val_accuracy')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Data")
    plt.title("Accuracy and Validation_Accuracy "+name)
    plt.show()

    plt.plot(list(range(0,epoch_num)),history.history["loss"], label = 'loss')
    plt.plot(list(range(0,epoch_num)),history.history["val_loss"], label = 'val_loss')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Data")
    plt.title("Loss and Validation_loss "+name)
    plt.show()

def export_model(model, lrate, name):
    optimizer = tf.keras.optimizers.SGD(lr=lrate, momentum=0.9) #Stochastic gradient descent optimizer
    model.compile(optimizer = 'adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    print('data is compiled')
    history = model.fit(x=X_train, y=y_train, epochs=epoch_num, validation_data=(X_test, y_test))
    t = time.time()
    title = name+str(int(t))
    export_path_keras = "./{}.h5".format(title)
    print(export_path_keras)
    model.save(export_path_keras)

model2 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation = "relu", input_shape = [window]),
    tf.keras.layers.Dense(128, activation = "relu"),
    tf.keras.layers.Dense(64, activation = "relu"),
    tf.keras.layers.Dense(1)
])

modelrnn1 = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                            input_shape = [None]),
    tf.keras.layers.SimpleRNN(100, return_sequences = True),
    tf.keras.layers.SimpleRNN(100),
    tf.keras.layers.Dense(1)
])

modelrnn2 = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                            input_shape = [None]),
    tf.keras.layers.SimpleRNN(100, return_sequences = True),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(1)
])

modelinitializer1 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation = "relu", input_shape = [window],
            kernel_initializer=tf.keras.initializers.RandomNormal(mean = 0.5, stddev = 0.5)),
    tf.keras.layers.Dense(128, activation = "relu"),
    tf.keras.layers.Dense(64, activation = "relu"),
    tf.keras.layers.Dense(1)
])

modelrnninitializer1 = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                            input_shape = [None]),
    tf.keras.layers.SimpleRNN(100, return_sequences = True,
            kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.5, stddev = 0.5)),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(1)
])


determine_lr(modelrnn1, "modelrnn1", 1e-6)
#run_model(modelrnninitializer1, 1e-3)
#export_model(modelrnn2, 5e-4, "modelrnn2_test")
#accuracy_plots(modelrnn2, " – ModelRNN2, Window = 15, LR: 2*10^-4", 2e-4)
