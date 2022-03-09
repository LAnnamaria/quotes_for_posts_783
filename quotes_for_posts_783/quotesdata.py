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
    #path = os. getcwd()
    quotes = pd.read_csv(f'gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}')
    
    return quotes

def remove_punctuations(text):
    '''removes punctuation from a given text'''
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def clean_data(quotes, test=False):
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
    quotes = get_quotes_data()
    quotes = clean_data(quotes)
    print(quotes.info())
