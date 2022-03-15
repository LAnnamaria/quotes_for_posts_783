import os
import pandas as pd
import string

def get_quotes_data():
    '''returns a DataFrame with 500k quotes, their authors and categories'''
    path = os. getcwd()
    quotes = pd.read_csv(f"{path}/raw_data/quotes.csv")
    return quotes

def remove_punctuations(text):
    '''removes punctuation from a given text'''
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def clean_data(test=False):
    quotes = get_quotes_data()
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
    quotes.to_csv('raw_data/cleaned_quotes.csv', index=False)
    print('Saved the cleaned dataframe as cleaned_quotes')
    return quotes

if __name__ == '__main__':
    quotes_clean = clean_data()
