FROM python:3.12-slim

WORKDIR /app

# Install shared requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy only llm_service code and .env
COPY llm_service/ .

EXPOSE 8001

CMD ["uvicorn", "llm_api:app", "--host", "0.0.0.0", "--port", "8001"]
