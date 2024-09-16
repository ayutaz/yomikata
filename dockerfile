# CUDA 12.1 ベースイメージを使用
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# 環境変数を設定
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# 必要なシステムパッケージをインストール
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python3 をデフォルトの python にする
RUN ln -s /usr/bin/python3 /usr/bin/python

# 作業ディレクトリを設定
WORKDIR /app

# requirements.txtファイルをコピー
COPY requirements/requirements-inference.txt .

# 必要なディレクトリを作成
RUN mkdir -p /app/stores/dbert

# PyTorch と CUDA 12.1 互換のバージョンをインストール
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# NumPy, および他の依存関係をインストール
RUN pip3 install --no-cache-dir \
    numpy==1.24.3 \
    plac \
    streamlit \
    spacy \
    pandas \
    altair==4.2.2

# 他の依存関係をインストール
RUN pip3 install --no-cache-dir -r requirements-inference.txt

# Yomikataをインストール
RUN pip3 install yomikata

# モデルアーティファクトを直接ダウンロードして展開
RUN mkdir -p /app/yomikata && \
    wget https://github.com/passaglia/yomikata/releases/download/0.0.1-alpha/dbert-artifacts.tar.gz -O /tmp/dbert-artifacts.tar.gz && \
    tar -xzvf /tmp/dbert-artifacts.tar.gz -C /app/yomikata && \
    rm /tmp/dbert-artifacts.tar.gz && \
    ls -R /app/yomikata

# 環境変数の設定
ENV YOMIKATA_MODEL_DIR=/app/yomikata/dbert-artifacts

# アプリケーションのソースコードをコピー
COPY . .

# ポート8501を開放（Streamlitのデフォルトポート）
EXPOSE 8501

# デバッグ用：環境変数とモデルディレクトリの内容を表示
RUN echo $YOMIKATA_MODEL_DIR && ls -l $YOMIKATA_MODEL_DIR

RUN ls -la /app/yomikata/dbert-artifacts

# Streamlitアプリケーションを実行するコマンド
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]