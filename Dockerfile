FROM ubuntu:16.04

# apt-getはdockerを構築する：定番   \で改行
RUN apt-get update && apt-get install -y \
    # root以外のユーザーがrootの権限を使うため
    sudo \
    # internetからツールを取得
    wget \
    # vim をエディタとして使用
    vim 

# root以外のユーザーが使いやすいようにする　よく使う 
# WORKDIRで最初の作業場所を宣言できる dirがなければ作成する
WORKDIR /opt

COPY . .

# anacondaインストーラのインストール
RUN wget https://repo.continuum.io/archive/Anaconda3-2019.10-Linux-x86_64.sh && \
    # anacondaインストーラからanaconda3をopt/にインストール
    sh /opt/Anaconda3-2019.10-Linux-x86_64.sh -b -p /opt/anaconda3 && \
    # 不要なshファイルを消す -fで強制消去
    rm -f Anaconda3-2019.10-Linux-x86_64.sh
# pathを通す ENVはexport PATH=と同じ
ENV PATH /opt/anaconda3/bin:$PATH

ADD note/requirement.txt /
# pipを活用できるようにする
RUN pip install --upgrade pip 
    #     # requirement.txtはpip -r install でインストールする
#RUN pip install -r /requirement.txt
# jupyter lab --ip=0.0.0.0 --allow-root --LabApp.token=''
#  --LabApp.token= はpasswordをなくす設定
# CMD ["jupyter","lab","--ip=0.0.0.0","--allow-root","--LabApp.token='' "]
