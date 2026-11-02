FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt trực tiếp danh sách thư viện không cần qua file requirements.txt
RUN pip install --no-cache-dir zerotts fastapi "uvicorn[standard]" soundfile numpy

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
