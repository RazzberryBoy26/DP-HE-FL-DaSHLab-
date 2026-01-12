FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
COPY global_node.py .
RUN python -c "import torchvision; torchvision.datasets.MNIST('./data', train=True, download=True)"
CMD ["python", "global_node.py"]