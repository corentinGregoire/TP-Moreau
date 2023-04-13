# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10-slim-bullseye

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

ENV PATH $PATH:/root/.local/bin

# Install pip requirements
ADD requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY . /app/

CMD ["python", "main.py"]
