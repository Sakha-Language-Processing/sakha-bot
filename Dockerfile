FROM python:3.11-slim
WORKDIR work
COPY bot.py prepare.py requirements.txt ./
RUN apt-get update && \
    apt-get install -y build-essential && \
    pip install -r requirements.txt && \
    python prepare.py
CMD python bot.py
