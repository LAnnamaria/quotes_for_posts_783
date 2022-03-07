FROM python:3.8.12-buster

COPY quotes_for_posts_783 /quotes_for_posts_783
COPY nn_euc.joblib /nn_euc.joblib
COPY nn_min.joblib /nn_min.joblib
COPY top5.joblib /top5.joblib
COPY model.joblib /model.joblib
COPY requirements.txt /requirements.txt


RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api:app --host 0.0.0.0 --port $PORT