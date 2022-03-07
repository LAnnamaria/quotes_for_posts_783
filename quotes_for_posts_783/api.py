from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime
import pytz
import joblib

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

@app.get("/predict")
def predict(image):
    image = """here comes the path to the picture?"""
    topic_image_model = joblib.load(PATH_TO_LOCAL_IMAGETOPIC_MODEL)
    image_topic = topic_image_model.predict(image)
    caption_model = joblib.load(PATH_TO_LOCAL_CAPTION_MODEL)
    image_caption = caption_model.predict(image_topic)
    top5_model = joblib.load(PATH_TO_LOCAL_TOP5_MODEL)
    top5 = top5_model.predict(image_caption)
    return top5

@app.get("/predict")
def most_suitable(tags):
    tags = """here comes the input from the user"""
    most_suitable_model = joblib.load(PATH_TO_LOCAL_MOST_SUITABLE_MODEL)
    most_suitable_quote = most_suitable_model.predict(tags)
    return most_suitable_quote