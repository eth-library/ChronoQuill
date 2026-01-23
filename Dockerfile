FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends wget unzip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
RUN uv pip install --system -r requirements.txt

CMD ["tail", "-f", "/dev/null"]

# docker build -t myimage . && docker run -it --rm -v .:/app myimage