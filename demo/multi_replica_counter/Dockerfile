FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements-multi-replica.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-multi-replica.txt cryptography
COPY gradio/ ./gradio/
COPY client/python/ ./client/python/
RUN pip install --no-cache-dir -e ./client/python -e .
COPY demo/multi_replica_counter/ ./demo/multi_replica_counter/
EXPOSE 7860
CMD ["python", "demo/multi_replica_counter/run.py"]
