FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV SHGNN_CHECKPOINT=/app/checkpoints/best.pt
ENV SHGNN_ONNX_PATH=/app/model.onnx

EXPOSE 8000

CMD ["sh-gnn", "serve", "--port", "8000"]