# Python 3.10のベースイメージを使用
FROM python:3.10-slim-buster

# 作業ディレクトリを設定
WORKDIR /app

# 必要なシステムパッケージをインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# requirements.txtファイルをコピー
COPY requirements/requirements-inference.txt .

# 必要なディレクトリを作成
RUN mkdir -p /app/stores/dbert

# NumPy, PyTorch, および他の依存関係をインストール
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    torch==1.13.1+cpu \
    torchvision==0.14.1+cpu \
    torchaudio==0.13.1 \
    plac \
    streamlit \
    spacy \
    pandas \
    altair==4.2.2 \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 他の依存関係をインストール（torch関連とnumpyを除く）
RUN pip install --no-cache-dir -r requirements-inference.txt

# Yomikataをインストール
RUN pip install yomikata

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