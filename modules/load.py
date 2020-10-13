import pandas as pd
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 複数のcsvファイルを纏める場合
# .csvファイルを取り出してdataframeに変換

# df.rename(columns={変更前のカラム名: 変更後のカラム名}, inplace=True) カラム名の変更


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
        print(df[df_obob.columns.values[i]].unique())
        print("total:", len(df[df_obob.columns.values[i]].unique()))


# カラムの消去

def delete(df):
    print("colmuns")
    print("↓")
    print(df.columns.values)
    choice_num = int(input("How do you want to kill columns?:"))
    i = 0
    while i <= choice_num:
        i += 1
