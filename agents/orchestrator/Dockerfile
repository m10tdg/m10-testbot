FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Playwright needs browser binaries — only in the execution-agent image
RUN playwright install --with-deps chromium
COPY . .
CMD ["python", "main.py"]