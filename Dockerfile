FROM python:3.13-slim

WORKDIR /app

# 필수 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main1.py"]