from urllib.parse import quote_plus
import pandas as pd
import string
import os


BUCKET_NAME = 'quotes_for_posts_783'

BUCKET_TRAIN_DATA_PATH = 'raw_data/Quotes_dataset/quotes.csv'

MODEL_NAME = 'quotes_classification'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

def get_quotes_data():
    '''returns a DataFrame with 500k quotes, their authors and categories'''
#<<<<<<< HEAD:quotes_for_posts_783/quotesdata.py
    #path = os. getcwd()
    pass
#=======
 #   path = os. getcwd()
#<<<<<<< HEAD:quotes_for_posts_783/data/quotesdata.py
 #   quotes = pd.read_csv(f"{path}/raw_data/quotes - reduced.csv")
#=======
  #  quotes = pd.read_csv(f"{path}/raw_data/quotes.csv")
#>>>>>>> bf2c607e60a808e57d49dff4f8b46f8f57eb58c3:quotes_for_posts_783/data/quotesdata.py
#>>>>>>> e7e4abab19bae76d6e829280f968bfb13b804c9a:quotes_for_posts_783/quotesdata.py


def remove_punctuations(text):
    '''removes punctuation from a given text'''
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def clean_data(test=False):

    quotes =  pd.read_csv(f'gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}')
    quotes = quotes.head(1000)
    '''setting everything to lowercase, replacing - with , and deleting duplicate tags in category
    (creating list_tags for further use) and adding an extra column called count_tags. Returning the quotes dataframe'''
    for index, row in quotes.iterrows():
        quotes.loc[index, "category"] = str(row['category']).lower().replace('-',', ')
    quotes['list_tags'] = quotes['category'].copy()
    for index,row in quotes.iterrows():
        quotes.loc[index, 'count_tags'] = len(str(row['list_tags']).split(','))
    for index,row in quotes.iterrows():
        quotes.at[index, 'list_tags'] = str(row['list_tags']).split(',')
    for index,row in quotes.iterrows():
        quotes.at[index, 'list_tags'] = str(set(row['list_tags']))
    quotes['list_tags'] = quotes['list_tags'].apply(remove_punctuations)
    return quotes

if __name__ == '__main__':
    quotes_clean = clean_data()
