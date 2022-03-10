FROM python:3.8.12-buster

COPY quotes_for_posts_783 /quotes_for_posts_783
COPY api /api
COPY requirements.txt /requirements.txt
COPY /home/alibor/code/LAnnamaria/GCPcreds/quantum-potion-337818-b287521fbd19.json /key.json

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api:app --host 0.0.0.0 --port $PORT
