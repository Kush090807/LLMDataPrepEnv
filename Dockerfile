FROM python:3.10-slim

WORKDIR /app

# Upgrade pip and install standard requirements
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir pydantic openai openenv fastapi uvicorn

# Copy environment files
COPY openenv.yaml .
COPY data.json .
COPY models.py .
COPY env.py .
COPY graders.py .
COPY inference.py .
COPY app.py .

# Expose standard HF Space port (if applicable)
EXPOSE 7860

# Entrypoint for running the baseline inference script
CMD ["python", "app.py"]
