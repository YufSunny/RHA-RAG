FROM python:3.12-slim

WORKDIR /app

# Install deps first (cache this layer unless requirements.txt changes).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project.
COPY . .

EXPOSE 8000
CMD ["python", "server.py"]
