from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime
import pytz
import joblib
from quotes_for_posts_783.trainer_get_caption import ImageCaption
from quotes_for_posts_783.trainer_get_quotes import GetData , GetQuote


PATH_TO_LOCAL_IMAGETOPIC_MODEL = 'model.joblib'
PATH_TO_LOCAL_CAPTION_MODEL = '''Mohana´s joblib model'''
PATH_TO_LOCAL_TOP5_MODEL = 'top5.joblib'
PATH_TO_LOCAL_NN_MIN_MODEL = 'nn_min.joblib'
PATH_TO_LOCAL_NN_EUC_MODEL = 'nn_euc.joblib'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Quotes for your posts! Get fitting quotes for your picture with one click!"}

@app.get("/top5")
def top5():
    q = GetData()
    cap_tr = ImageCaption()
    cap = cap_tr.nlp()
    quotes = q.clean_data(cap)
    trainer = GetQuote(quotes)
    quotess = trainer.run()
    top5 = trainer.top5_fuc(quotess)
    #most_suitable_quote = trainer.most_suitable(quotess,tags)
    return {'top5' : list(top5) }

@app.get("/final")
def most_suitable(tags):
    tags = """here comes the input from the user"""
    q = GetData()
    cap_tr = ImageCaption()
    cap = cap_tr.nlp()
    quotes = q.clean_data(cap)
    trainer = GetQuote(quotes)
    quotess = trainer.run()
    most_suitable_quote = trainer.most_suitable(quotess,tags)
    return most_suitable_quote


    # localized_pickup_datetime = pytz.timezone("US/Eastern").localize(datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S"), is_dst=None).astimezone(pytz.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    # params = {"key":[pickup_datetime],\
    #     "pickup_datetime": [localized_pickup_datetime],\
    #     "pickup_longitude": [float(pickup_longitude)],\
    #     "pickup_latitude": [float(pickup_latitude)],\
    #     "dropoff_longitude": [float(dropoff_longitude)],\
    #     "dropoff_latitude": [float(dropoff_latitude)],\
    #     "passenger_count": [int(passenger_count)]}

    # X_pred = pd.DataFrame(data = params)
    # pipeline = joblib.load(PATH_TO_LOCAL_MODEL)
    # y_pred = pipeline.predict(X_pred)
    # return {"fare": y_pred[0]}
