from modules.load import *
from modules.lstm import *
from scipy.interpolate import make_interp_spline, UnivariateSpline
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, LSTM, TimeDistributed
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import plot_confusion_matrix
from sklearn.utils.multiclass import unique_labels
from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model
from sklearn.model_selection import KFold

from tcn import TCN, tcn_full_summary

# ① 不均一データのスムージング　20Hzのデータに変換
#minに関してはmin(x_data)とすることでデータの最小の秒数にすることが出来る。　maxも同様
def smoothings(x_data, y_data, min=0, max=50, rate=1000, font=True):
    # np.linspace(データの最小値, データの最大値, サンプリングレート)
    x_data_smooth = np.linspace(min, max, rate)

    fig, ax = plt.subplots(1, 1)

    # 平滑化スプラインの次数k =>今回は二次元であるため2
    spl = UnivariateSpline(x_data, y_data, s=0, k=2)
    y_data_smooth = spl(x_data_smooth)
    if font == True:
        ax.plot(x_data_smooth, y_data_smooth, 'b', label="resample")
        ax.plot(x_data, y_data, "red", label="original")
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
    elif font == False:
        ax.plot(x_data_smooth, y_data_smooth, 'b')
        ax.plot(x_data, y_data, "red")
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right')


    return x_data_smooth, y_data_smooth

# pandasのデータとして入れるとき 0列目に時間 1列目に電圧となっていることを確認する
# x : 秒数のデータ
# y : 電圧のデータ

#minに関してはmin(x_data)とすることでデータの最小の秒数にすることが出来る。　maxも同様
def smoothings_from_pd(df, min=0, max=50, rate=1000, font=True):
    x_data = df.iloc[:, 0]
    x_data  = x_data-x_data[0]
    y_data = df.iloc[:, 1]

    # np.linspace(データの最小値, データの最大値, サンプリングレート)
    x_data_smooth = np.linspace(min, max, rate)

    fig, ax = plt.subplots(1, 1)

    # 平滑化スプラインの次数k =>今回は二次元であるため2
    spl = UnivariateSpline(x_data, y_data, s=0, k=2)
    y_data_smooth = spl(x_data_smooth)

    if font == True:
        ax.plot(x_data_smooth, y_data_smooth, 'b', label="resample")
        ax.plot(x_data, y_data, "red", label="original")
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right')

    elif font == False:
        ax.plot(x_data_smooth, y_data_smooth, 'b')
        ax.plot(x_data, y_data, "red")
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right')

    flg = plt.xlim(10, 20)
    flg = plt.ylim(3.2, 3.8)
    flg = plt.show()

    #x_datasmoothに関しては(0,49)のデータでどれも等しいので無くても良い
    return x_data_smooth, y_data_smooth


# テーブル, 椅子, デニム, ダンボール, ペットボトル, 指, マジックテープ, スポンジ, ガムテープ, タオル の順番でセット
# データセットを作成する
def set_dict(df_list,
            data_col = ["table", "chair", "denim", "cardboard", "PET bottle", "finger", "magic tape", "sponge", "packing tape", "towel"]):
    datas = []
    logs = []
    means = []
    moves = []
    diffs = []
    for i in df_list:
        x, y = smoothings_from_pd(i)
        z = np.log(y)
        zz = pd.DataFrame(y).rolling(window=20).mean()
        zzz = pd.DataFrame(y)-zz
        zzzz = pd.DataFrame(y).diff()
        zz = np.array(zz).astype("float32")
        zzz = np.array(zzz).astype("float32")
        zzzz = np.array(zzzz).astype("float32")
        datas.append(y)
        logs.append(z)
        means.append(zz)
        moves.append(zzz)
        diffs.append(zzzz)

    #ディクショナリー型のデータセットを作製　key:value = label:wave
    df_dict = dict(zip(data_col, datas))
    df_dict_logs = dict(zip(data_col, logs))
    df_dict_means = dict(zip(data_col, means))
    df_dict_moves = dict(zip(data_col, moves))
    df_dict_diffs = dict(zip(data_col, diffs))
    return df_dict, df_dict_logs, df_dict_means, df_dict_moves, df_dict_diffs

# ②' データのグラフの確認 まとめて
# xには秒数データ(0,49)の20Hzのデータ
def show_concat(x, df_dict):
    plt.figure(figsize=(12, 4))
    for i in df_dict:
        plt.plot(x, df_dict[i],  label=i)
    plt.xlim(0, 10)
#     plt.ylim(2.4,2.8)
    plt.xlabel("time [s]")
    plt.ylabel("voltage [V]")
    plt.legend()
    plt.show()


# ②'　単一データの確認
# xには秒数データ(0,49)の20Hzのデータ yはdf_dict["col"]の形式
def show_one(x,y):
    plt.figure(figsize=(12, 4))
    plt.plot(x, y,  label="finger")
    plt.xlim(0, 10)
    # plt.ylim(2.4,2.8)
    plt.xlabel("time [s]")
    plt.ylabel("voltage [V]")
    plt.legend()
    plt.show()


# ②' p値の確認
def stats_p(df_dict):
    for i in df_dict:
        ct = sm.tsa.stattools.adfuller(df_dict[i], regression="ct")
        print("{}のp値:{}".format(i,ct[1]))


# ③データの標準化の一括
class _Standard:

    # dictをインスタンスとして定義
    def __init__(self, df_dict, df_dict_moves, df_dict_diffs):
        self.df_dict = df_dict
        self.df_dict_moves = df_dict_moves
        self.df_dict_diffs = df_dict_diffs

        self.data_col = ["table", "chair", "denim", "cardboard", "PET bottle",
                        "finger", "magic tape", "sponge", "packing tape", "towel"]

    # データの標準化の一括 1:一般 2:移動平均 3:階差
    def standard_one(self, select: int, created_list=[]):

        if select == 1:
            for i in self.df_dict:
                y = np.array(df_dict[i]).reshape(-1, 1)
                y = scale(y)
                a, y = np.array(y)
                y = np.array(y)
                created_list.append(y)

        elif select == 2:
            for i in self.df_dict_moves:
                y = np.array(df_dict[i]).reshape(-1, 1)
                y = scale(y)
                a, y = np.array(y)
                y = np.array(y)
                created_list.append(y)

        elif select == 3:
            for i in self.df_dict_diffs:
                y = np.array(df_dict[i]).reshape(-1, 1)
                y = scale(y)
                a, y = np.array(y)
                y = np.array(y)
                created_list.append(y)

        df_dict_std = dict(zip(self.data_col, created_list))
        return df_dict_std

    # 一括して標準化
    def standard_all(self, created_list=[]):
        df_con = [self.df_dict, self.df_dict_moves, self.df_dict_diffs]
        dic_list = []

        for i in df_con:
            for j in i:
                y = np.array(i[j]).reshape(-1, 1)
                y = scale(y)
                a, y = np.array(y)
                y = np.array(y)
                created_list.append(y)
            df_dict_std = dict(zip(self.data_col, created_list))
            dic_list.append(df_dict_std)
            #created_listを初期化
            created_list =[]

        return dic_list


# standardデータを渡す df_dict_stdとか
# standardデータを渡す df_dict_stdとか
class _CreateDataset(_Standard):
    #名前をラベルにしたDataSet

    def make_dataset_name(self, select:int, time_steps):
        x, y, z = [], [], []
        h = 0

        #selectの値によって分岐
        if select == 0:
            df_dict = self.df_dict
        elif select == 1:
            df_dict = self.df_dict_moves
        elif select == 2:
            df_dict = self.df_dict_diffs

        for i in df_dict:
            data, target = [], []
            maxlen = time_steps

            for j in range(len(df_dict[i])-maxlen):
                data.append(df_dict[i][j:j + maxlen])
                target.append(i)

            re_data = np.array(data).reshape(len(data), maxlen, 1)
            re_target = np.array(target).reshape(len(data), 1)
            csv_data = np.array(data).reshape(len(data), maxlen)
            csv_data = pd.DataFrame(csv_data)
            csv_data["LABEL"] = re_target
            h += 1
            x.append(re_data)
            y.append(re_target)
            z.append(csv_data)

        return x, y, z


    #ラベルを数値にしたDataSet

    def make_dataset(self, select: int, time_steps):
        x, y, z = [], [], []
        h = 0

        #selectの値によって分岐
        if select == 0:
            df_dict = self.df_dict
        elif select == 1:
            df_dict = self.df_dict_moves
        elif select == 2:
            df_dict = self.df_dict_diffs

        for i in df_dict:
            data, target = [], []
            maxlen = time_steps

            for j in range(len(df_dict[i])-maxlen):
                data.append(df_dict[i][j:j + maxlen])
                target.append(h)

            re_data = np.array(data).reshape(len(data), maxlen, 1)
            re_target = np.array(target).reshape(len(data), 1)
            csv_data = np.array(data).reshape(len(data), maxlen)
            csv_data = pd.DataFrame(csv_data)
            csv_data["LABEL"] = re_target
            h += 1
            x.append(re_data)
            y.append(re_target)
            z.append(csv_data)

        return x, y, z


    #ラベルを数値にしたDataSet ストライド付き

    def make_dataset_stride(self, select: int, time_steps, stride):
        x, y, z = [], [], []
        h = 0

        #selectの値によって分岐
        if select == 0:
            df_dict = self.df_dict
        elif select == 1:
            df_dict = self.df_dict_moves
        elif select == 2:
            df_dict = self.df_dict_diffs

        for i in df_dict:
            data, target = [], []
            maxlen = time_steps

            for j in range(int((len(df_dict[i])-maxlen)/int(stride))):
                data.append(df_dict[i][j*stride:j*stride + maxlen])
                target.append(h)

            re_data = np.array(data).reshape(len(data), maxlen, 1)
            re_target = np.array(target).reshape(len(data), 1)
            csv_data = np.array(data).reshape(len(data), maxlen)
            csv_data = pd.DataFrame(csv_data)
            csv_data["LABEL"] = re_target
            h += 1
            x.append(re_data)
            y.append(re_target)
            z.append(csv_data)

        return x, y, z

    def make_dataset_stride_std(self, select: int, time_steps,  stride):
        x, y, z = [], [], []
        h = 0

        #selectの値によって分岐
        if select == 0:
            df_dict = self.df_dict
        elif select == 1:
            df_dict = self.df_dict_moves
        elif select == 2:
            df_dict = self.df_dict_diffs

        for i in df_dict:
            data, target = [], []
            maxlen = time_steps

            for j in range(int((len(df_dict[i])-maxlen)/int(stride))):
                #時系列データ一つ分
                wave = df_dict[i][j*stride:j*stride + maxlen]
                scaler = StandardScaler()
                scaler.fit(wave)
                wave_std = scaler.transform(wave)
                data.append(wave_std)
                target.append(h)

            re_data = np.array(data).reshape(len(data), maxlen, 1)
            re_target = np.array(target).reshape(len(data), 1)
            csv_data = np.array(data).reshape(len(data), maxlen)
            csv_data = pd.DataFrame(csv_data)
            csv_data["LABEL"] = re_target
            h += 1
            x.append(re_data)
            y.append(re_target)
            z.append(csv_data)

        return x, y, z

    # pandasの形式にデータを変換
    def make_csv(self, name, dataset ,save_data=False):
        con = pd.DataFrame(index=[], columns=[])
        for i,j in enumerate(dataset):
            #save_dataをTrueにすると作製
            con=pd.concat([con,j])
            if save_data:
                j.to_csv("{}{}.csv".format(self.data_col[i],name))
        return con

    # make_csvを使わずにいきなりこっちをつかうとcontatまで行う
    def make_csv_con(self, name, dataset_list, save_data=False, drop=True):
        def make_csv(name, dataset ,save_data=False):
            con = pd.DataFrame(index=[], columns=[])
            for i,j in enumerate(dataset):
                #save_dataをTrueにすると作製
                con=pd.concat([con,j])
                if save_data:
                    j.to_csv("{}{}.csv".format(self.data_col[i],name))
            return con
        con1 = make_csv(name, dataset_list[0], save_data)
        con2 = make_csv(name, dataset_list[1], save_data)
        con3 = make_csv(name, dataset_list[2], save_data)
        # データの統合 con3に関しては最後にLABELのカラムを残すためilocしない
        cons = pd.concat(
            [con1.iloc[:, :-1], con2.iloc[:, :-1], con3], axis=1)

        if drop == True:
            drop_n = sum(
                [True for idx, row in cons.iterrows() if any(row.isnull())])
            cons = cons.dropna()
            print("{}件Nanデータが有ったため削除しました".format(drop_n))
        else:
            pass
        return cons

    # データセットの形状変更 -> (sample, step, feature)
    def make_data(self,concat):
        colms = concat.shape[1]-1
        step = int(colms/3)
        ex1 = concat.iloc[:,0:step]
        ex2 = concat.iloc[:,step:step*2]
        ex3 = concat.iloc[:,step*2:step*3]
        ex1 = np.array(ex1).reshape(-1,step,1)
        ex2 = np.array(ex2).reshape(-1,step,1)
        ex3 = np.array(ex3).reshape(-1,step,1)
        fin = np.concatenate([ex1,ex2,ex3], axis=2)
        label = concat.iloc[:,-1]
        label = np.array(label)
        label = label.reshape(-1,1)
        label =one_hot(label)

        return fin,ex1,ex2,ex3,label

    def _check(self,concat,check_rand = 5,check_col="finger"):
        plt.figure(figsize=(12, 4))
        # *0.05 samplingrate=20Hzなので
        plt.plot((np.arange(0, 20))*0.05, concat[200],  label="finger")
        plt.xlim(0, 2)
        plt.ylim(-4, 4)
        plt.xlabel("time [s]")
        plt.ylabel("voltage [V]")
        plt.legend()
        plt.show()




class Deeps:

    def __init__(self, batchsize=128, unit=128, label_num=10, dropout=0.2, epochs=30):
        self.batchsize = batchsize
        self.unit = unit
        self.label_num = label_num
        self.dropout = dropout
        self.es = EarlyStopping(
            monitor='val_loss', patience=10, verbose=0, mode='auto')
        self.epochs = epochs

    def lstm(self, X_train):
        #         optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        optimizer = RMSprop(learning_rate=0.01)
        model = Sequential()
        model.add(LSTM(self.unit, input_shape=(
            X_train.shape[1], X_train.shape[2]), return_sequences=False))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.label_num))
        model.add(Activation("softmax"))
        model.summary()
        return model

    def lstm2(self, X_train):
                # optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        optimizer = RMSprop(learning_rate=0.01)
        model = Sequential()
        model.add(LSTM(self.unit, input_shape=(
            X_train.shape[1], X_train.shape[2]), return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.unit, return_sequences=False))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.label_num))
        model.add(Activation("softmax"))
#         model.compile(optimizer=optimizer,
#                     loss='categorical_crossentropy',
#                     metrics=['accuracy'])
        model.summary()
        return model

    def tcn(self, X_train):
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                        epsilon=None, decay=0.0, amsgrad=False)
        i = Input(shape=(X_train.shape[1], X_train.shape[2]))
        # The TCN layers are here.
        o = TCN(self.unit, nb_stacks=2, dilations=[
            1, 2, 4, 8, 16, 32], return_sequences=False)(i)
        o = Dense(10, activation='softmax')(o)

        model = Model(inputs=[i], outputs=[o])
#         model.compile(optimizer=optimizer,
#                     loss='categorical_crossentropy',
#                     metrics=['accuracy'])
        model.summary()
        return model

    # tcnのとき => optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999,epsilon = None, decay = 0.0, amsgrad = False)
    # lstmのとき => optimizer = RMSprop(learning_rate=0.01)
    def model_compile(self, model, optimizer):
        model.compile(loss="categorical_crossentropy",
                    optimizer=optimizer, metrics=['accuracy'])
        model.summary()
        return model

    def learn(self, model, X_train, Y_train, validation_split):
        history = model.fit(X_train, Y_train, batch_size=self.batchsize,
                            epochs=self.epochs, validation_split = validation_split,  callbacks=self.es)
        return history

    # ここのモデルはcompile前のモデルを入れる
#     # classの中だと使えない？
    def cross_val_learn(self,model_init, x_train, y_train, n_split: int):
        kf = KFold(n_splits=n_split, random_state=1234)
        _history =[]
        for train_index, val_index in kf.split(x_train, y_train):
            # モデルを更新する
            model = model_init
            model.compile(optimizer="adam",
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
            model_init.fit(x=x_train[train_index], y=y_train[train_index], batch_size = self.batchsize, epochs=self.epochs, verbose=1)
            _history.append(model.evaluate(x=x_train[val_index], y=y_train[val_index], batch_size=self.batchsize))
        _history = np.asarray(_history)
        loss = np.mean(_history[:, 0])
        acc = np.mean(_history[:, 1])
        print(f'loss: {loss} ± {np.std(_history[:, 0])} | acc: {acc} ± {np.std(_history[:, 1])}')
        return model_init, _history

    def plot_history(self, history):
        # model loss graph

        def plot_history_loss(fit):
            axL.plot(fit.history['loss'], label="loss for training")
            axL.plot(fit.history['val_loss'], label="loss for validation")
            axL.set_title('model loss')
            axL.set_xlabel('epoch')
            axL.set_ylabel('loss')
            axL.legend(loc='upper right')

        # model accuracy graph

        def plot_history_accuracy(fit):
            axR.plot(fit.history['accuracy'],
                    label="accuracy for training")
            axR.plot(fit.history['val_accuracy'],
                    label="accuracy for validation")
            axR.set_title('model accuracy')
            axR.set_xlabel('epoch')
            axR.set_ylabel('accuracy')
            axR.legend(loc='lower right')

        ig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 5))
        plt.subplots_adjust(wspace=0.5)
        plot_history_loss(history)
        plot_history_accuracy(history)

    def model_save(self, model, model_name):
        model.save("{}.h5".format(model_name))
        model.save_weights("{}_weight.h5".format(model_name))

    def test(self, model, X_test, Y_test):
        evaluate_model(model, X_test, Y_test)



# classの外に出してつかう
def cross_val_learn(x_train, y_train, n_split: int):
    kf = KFold(n_splits=n_split, random_state=1234)
    _history = []
    for train_index, val_index in kf.split(x_train, y_train):
        # モデルを更新する  ←　ここに使いたいモデルを直接入力　そうじゃないとモデルの状態が更新されない
        model = dee.lstm(X_train)
        model.compile(optimizer="adam",
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        model.fit(x=x_train[train_index], y=y_train[train_index],
                batch_size=30, epochs=10, verbose=1)
        _history.append(model.evaluate(
            x=x_train[val_index], y=y_train[val_index], batch_size=30))
    _history = np.asarray(_history)
    loss = np.mean(_history[:, 0])
    acc = np.mean(_history[:, 1])
    print(
        f'loss: {loss} ± {np.std(_history[:, 0])} | acc: {acc} ± {np.std(_history[:, 1])}')
    return model, _history
