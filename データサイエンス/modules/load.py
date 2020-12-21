import pandas as pd
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sns.set()  # これでグラフ描画にseabornが使われるようになる

# 複数のcsvファイルを纏める場合
# .csvファイルを取り出してdataframeに変換

# df.rename(columns={変更前のカラム名: 変更後のカラム名}, inplace=True) カラム名の変更

# データを結合してconcatファイルとして保存する


def file_concat(file_pass):
    file_name = input("統合したいfileをcsvの前までつけてください")
    defe = []
    for i in range(len(glob.glob(file_pass+"/*.csv"))):
        df = pd.read_csv(glob.glob(file_pass+"/*.csv")[i])
        defe.append(df)
    frame = pd.concat(defe, join='inner')  # joinをinnerに指定
    frame.to_csv(file_pass+"/"+file_name+".csv", encoding="utf-8-sig")


# 単体ファイルの取り出し


def load(files):
    df = pd.read_csv(files)
    return df


def show(df):
    print('dara shape==>\n', df.shape)
    print("------------------------------")
    print('index ==>\n', df.index)
    print("------------------------------")
    print('column ==>\n', df.columns)
    print("------------------------------")
    print('dtype==>\n', df.dtypes)
    print("------------------------------")
    print(df.head())
    print("------------------------------")
    print("describe==>\n", df.describe())
    df = df.drop_duplicates()  # delete duplicate data about index

    return df


def train_test_csv(df):
    target = input("please choice target column:")
    df_target = df[target]
    df_feature = df.drop(target, axis=1)
    df_target.to_csv("./data/target.csv", encoding="utf-8-sig")
    df_feature.to_csv("./data/feature.csv", encoding="utf-8-sig")

    return df_feature, df_target


# 量的データとカテゴリデータに分割


def sepalate(df):
    df_ob = []  # object型格納ボックス
    df_num = []  # int,float型格納ボックス
    # object型とint,float型をそれぞれ別のlist型に格納
    for i in range(len(df.columns)):
        print(df.iloc[:, i].dtype)
        if df.iloc[:, i].dtype == "object":
            df_ob.append(df.iloc[:, i])
        else:
            df_num.append(df.iloc[:, i])

    # object型のlist=>dataframeに変換
    # オブジェクトのリストをdataframeに変更する
    # オブジェクトのリストをdataframeに変更する
    df_obob = pd.DataFrame(df_ob[0])
    for i in range(len(df_ob)-1):
        df_o = pd.DataFrame(df_ob[i+1])
        print(df_o.columns)
        df_obob = pd.concat([df_obob, df_o], axis=1)

    # オブジェクトのリストをdataframeに変更する
    df_numnum = pd.DataFrame(df_num[0])
    for i in range(len(df_num)-1):
        df_n = pd.DataFrame(df_num[i+1])
        print(df_n.columns)
        df_numnum = pd.concat([df_numnum, df_n], axis=1)

        print(df_obob)
        print(df_numnum)

    return df_obob, df_numnum


# カテゴリ変数の中身を確認

def show_object(df_obob):
    for i in range(len(df_obob.columns.values)):
        print(df_obob.columns.values[i])
        print("↓")
        print(df_obob[df_obob.columns.values[i]].unique())
        print("total:", len(df_obob[df_obob.columns.values[i]].unique()))


# 指定のカラムの消去

def delete(df):
    print("colmuns")
    print("↓")
    print("")
    for i, col in enumerate(df.columns.values, 1):
        print(i, ":", col)

    print("----------------------------------------------------------")
    choice_num = int(input("What colmundo you want to kill ?:"))
    df_ob = df.drop(df.columns[choice_num-1], axis=1)
    print("----------------------------------------------------------")
    print(df.columns[choice_num-1], "is deleted")
    return df_ob

# ---------------objectに関して------------------------


# str => int or float　変換
# 手書き↓
# dfdf["Embarked"][dfdf["Embarked"] == "S"] = 0
# dfdf["Embarked"][dfdf["Embarked"] == "C"] = 1
# dfdf["Embarked"][dfdf["Embarked"] == "Q"] = 2

# ラクするよう
# LabelEncoderのインスタンスを生成
def label_encode(df):
    print("choice label encode colmuns !")
    print(df.columns.values)
    col = input().split()
    le = LabelEncoder()
    # ラベルを覚えさせる
    le = le.fit(df[col])
    # ラベルを整数に変換
    df[col] = le.transform(df[col])
    return df

# 数値データに戻したobjectとnumデータの合成


def concat_num_ob(df_ob, df_num):
    df_con = pd.concat([df_ob, df_num], axis=1)
    return df_con


# 欠損値データの確認

def null_table(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum()/len(df)
    null_table = pd.concat([null_val, percent], axis=1)
    null_table_ren_columns = null_table.rename(
        columns={0: '欠損数', 1: '%'})
    return null_table_ren_columns

# 欠損データの穴埋め
# dfには補完するカラムを選択する
# data['Age'].fillna(20)                   # 列Ageの欠損値を20で穴埋め
# data['Age'].fillna(data['Age'].mean())   # 列Ageの欠損値をAgeの平均値で穴埋め
# data['Age'].fillna(data['Age'].median()) # 列Ageの欠損値をAgeの中央値で穴埋め
# data['Age'].fillna(data['Age'].mode())   # 列Ageの欠損値をAgeの最頻値で穴埋め

# これするとバグる


class complement:
    def __init__(self, df):
        self.df = df

    # 欠損値の除去
    def dropna(self):
        df = self.df.dropna()
        return df

    def fill(self):
        print(self.df.columns.values)
        col = input("補完するデータを選んで下さい")
        sel = int(input("補完方法を選択してください\n1: 自ら選択, 2: mean, 3: median, 4: mode "))
        if sel == 1:
            col = input("カラム名:")
            choice = int(input("補完するデータは? 1:数値,2:オブジェクト"))
            if choice == 1:
                num = float(input("数値は:"))
                df = self.df[col].fillna(num)
            elif choice == 2:
                st = input("データは:")
                df = self.df[col].fillna(st)

        elif sel == 2:
            df = self.df.fillna(self.df[col].mean())
            return df

        elif sel == 3:
            df = self.df.fillna(self.df[col].median())
            return df
        elif sel == 4:
            df = self.df.fillna(self.df[col].mode())
            return df

# グラフ描画


def graph_plot(df):
    choice = int(input("グラフの描画タイプはどうしますか?\n1:全て 2:ターゲットを決める"))
    if choice == 1:
        show = sns.pairplot(df)
        print(show)
    elif choice == 2:
        target = input("ターゲットはどうしますか？")
        show = sns.pairplot(df, hue=target)
        print(show)


# ターゲットと特徴データに分割する

def train_test_csv(df):
    target = input("please choice target column:")
    df_target = df[target]
    df_feature = df.drop(target, axis=1)
    df_target.to_csv("./data/target.csv", encoding="utf-8-sig")
    df_feature.to_csv("./data/feature.csv", encoding="utf-8-sig")

    return df_feature, df_target

# 統計量の確認


def stats(data, target):
    result = sm.OLS(target.astype(float),
                    sm.add_constant(data.astype(float))).fit()
    print("result.summary()")
    return result

# 標準化


def scale(train_data):
    scaler = StandardScaler()
    scaler.fit(train_data)
    x_scaled = scaler.transform(train_data)
    x_sca = pd.DataFrame(x_scaled)
    return scaler, x_sca

# one-hotラベル化する


def one_hot_encode(df):
    one_df = np_utils.to_categorical(df)
    return one_df


# トレーニングデータとテストデータに分割

def split(X_data, y_data, test_size):
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=test_size, random_state=43)

    return X_train, X_test, y_train, y_test
