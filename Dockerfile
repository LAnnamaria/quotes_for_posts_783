FROM python:3.8.12-buster

COPY quotes_for_posts_783 /quotes_for_posts_783
COPY api.py /api.py
COPY requirements.txt /requirements.txt
COPY quantum-potion-337818-b287521fbd19.json /key.json

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

CMD uvicorn api:app --host 0.0.0.0 --port $PORT
