FROM python:3.12-slim

WORKDIR /app

# Install all system dependencies needed for Playwright/Chromium
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu fonts-liberation libglib2.0-0 libnspr4 libnss3 libx11-6 libxext6 libxfixes3 libxrender1 libxss1 xdg-utils ca-certificates libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libatspi2.0-0 libexpat1 libxcomposite1 libxdamage1 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 libcairo2 libasound2 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright without --with-deps (we provide system deps manually)
RUN playwright install chromium

COPY . .
CMD ["python", "main.py"]