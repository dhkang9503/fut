FROM python:3.10-slim

WORKDIR /app

COPY . .

# 필수 패키지 설치
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-m", "main.py"]
