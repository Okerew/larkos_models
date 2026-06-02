FROM pytorch/pytorch:2.12.0-cuda12.6-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    bash \
    build-essential \
    libjson-c-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

RUN chmod +x /app/compile.sh \
    /app/modules/fusion_mechanism/compile.sh

ENTRYPOINT ["/bin/bash", "-c", "\
    chmod +x /app/compile.sh /app/modules/fusion_mechanism/compile.sh && \
    /app/compile.sh && \
    /app/modules/fusion_mechanism/compile.sh && \
    exec python main.py"]
