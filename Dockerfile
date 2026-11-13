FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY forecastforge.py /app/forecastforge.py
RUN useradd --create-home --uid 10001 appuser
USER appuser
ENTRYPOINT ["python", "/app/forecastforge.py"]
