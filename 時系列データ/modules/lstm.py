from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dropout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

# show data_graph


def show(data):
    plt.figure()
    plt.plot(data.iloc[:, 0], data.iloc[:, 1], color="b", label="row_data")
    plt.show()

# get x,y
# グラフ描画のためのx,yを取得


def get_xy(data):
    print("data must have only figure data")
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    return x, y

# data.shape == >(n, 1)


def data_to_list(data):
    data_ = data.iloc[:, 0]
    data__ = data_.tolist()
    return data__

# down_sampling


def down_sampling(y, scale):  # y must series data =>pdの一次元データ (n,)
    yy = np.array(y)  # pandas.core.series.Series to
    yyy = yy.astype("float64")
    y_down = signal.decimate(yyy,10)
    return y_down
#plot graph 
def plot_time(y,height,width):
    plt.figure(figsize=(width,height))
    x = np.arange(0,4000,4)
    plt.plot(x, y, color="b", label="row_data")
    plt.xlabel("time[ms]")
    plt.ylabel("voltage[mV]")
    plt.show()


# data_shape => (sample,timesteps,feature_num)
# data_label => (label_num,)


def make_dataset(low_data):
    time_steps = int(input("timesteps:"))
    label_num = int(input("label_num:"))
    data, target = [], []
    maxlen = time_steps

    for i in range(len(low_data)-maxlen):
        data.append(low_data[i:i + maxlen])
        target.append(label_num)

    re_data = np.array(data).reshape(len(data), maxlen, 1)
    re_target = np.array(target).reshape(len(data), )

    return re_data, re_target

# you can change both train & label data


def merge_data(data1, data2):
    data = np.vstack((data1, data2))
    return data


def one_hot(label):
    one_hot_label = np_utils.to_categorical(label)
    return one_hot_label


def split(x_train, y_train):  # x_train=>data y_train=>labels
    choice = int(input("シャッフルする:1 シャッフルしない:2"))
    test_size = float(input("testサイズを選んでください"))
    cho = True
    while cho == True:
        if choice == 1:
            X_train, X_test, Y_train, Y_test = train_test_split(
                x_train, y_train, test_size=test_size, shuffle=True, random_state=42)
            cho = False
        elif choice == 2:
            X_train, X_test, Y_train, Y_test = train_test_split(
                x_train, y_train, test_size=test_size, shuffle=False)
            cho = False
        else:
            print("please select 1 or 2")
    return X_train, X_test, Y_train, Y_test

# エラーが起きる


def make_lstm(X_train, Y_train):  # X_train=>seq  Y_train=>label
    n_hidden = int(input("How hidden number you want to?"))
    # モデル構築
    optimizer = RMSprop()
    model = Sequential()
    model.add(LSTM(n_hidden, input_shape=(
        X_train.shape[1], X_train.shape[2]), return_sequences=False))  # trueはseq2seq
    model.add(Dropout(0.2))
    model.add(Dense(Y_train.shape[1]))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer, metrics=['accuracy'])
    model.summary()


def fit_model(X_train, Y_train, model):
    bach_size = int(input("select batch_size:"))
    epochs = int(input("select epochs"))
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)


def plot_history(history):
    # model loss graph

    def plot_history_loss(fit):
        axL.plot(fit.history['loss'], label="loss for training")
        axL.plot(fit.history['val_loss'],label="loss for validation")
        axL.set_title('model loss')
        axL.set_xlabel('epoch')
        axL.set_ylabel('loss')
        axL.legend(loc='upper right')

    # model accuracy graph

    def plot_history_accuracy(fit):
        axR.plot(fit.history['accuracy'],
                 label="accuracy for training")
        axR.plot(fit.history['val_accuracy'],label="accuracy for validation")
        axR.set_title('model accuracy')
        axR.set_xlabel('epoch')
        axR.set_ylabel('accuracy')
        axR.legend(loc='lower right')

    ig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 5))
    plt.subplots_adjust(wspace=0.5)
    plot_history_loss(history)
    plot_history_accuracy(history)

# model evaluete


def evaluate_model(model, x_test, y_test):
    history = model.evaluate(x_test, y_test, verbose=1)
    print("loss:", history[0], "accuracy:", history[1])
