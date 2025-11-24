FROM python:3.10-slim

WORKDIR /app

# 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# 소스 복사
COPY . .

# 파이썬 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# (선택) 컨테이너에서 대시보드 포트 노출
EXPOSE 8000

# 봇 + 대시보드 서버 둘 다 실행
# - main.py : 자동매매 봇
# - dashboard_server.py : FastAPI/uvicorn 서버 (포트 8000)
CMD ["sh", "-c", "python main.py & uvicorn dashboard_server:app --host 0.0.0.0 --port 8000"]
