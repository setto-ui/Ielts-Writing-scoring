# Base image Python 3.10 slim
FROM python:3.10-slim

# Cài Java (nếu language-tool cần) + các công cụ cơ bản
RUN apt-get update && apt-get install -y \
    openjdk-21-jdk-headless \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Copy requirements và cài thư viện
# ⚡ Dùng mirror Tsinghua để tránh timeout + fix numpy < 2
RUN pip install --no-cache-dir --default-timeout=1000 -i https://pypi.tuna.tsinghua.edu.cn/simple \
    flask \
    transformers \
    scikit-learn \
    pandas \
    numpy==1.26.4 \
    language-tool-python \
    torch==2.2.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Copy toàn bộ code vào container
COPY . .

# Mở port Flask
EXPOSE 5000

# Chạy API
CMD ["python", "app.py"]
