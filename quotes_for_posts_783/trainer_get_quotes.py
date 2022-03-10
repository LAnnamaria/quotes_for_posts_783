from quotes_for_posts_783.trainer_get_caption import ImageCaption
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import string
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation
#import quotes_for_posts_783.quotesdata as qd
#import quotes_for_posts_783.utils as u



def remove_punctuations(text):
    '''removes punctuation from a given text'''
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def clean_data(image_caption , test=False):
    path = os.getcwd()
    path = '/home/morad/code/LAnnamaria/quotes_for_posts_783'
    quotes =  pd.read_csv("/home/morad/code/LAnnamaria/quotes_for_posts_783/raw_data/quotes - reduced.csv")
   # quotes = quotes.head(10000)
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
    quotes.iloc[-1] = [image_caption, 'image','image',remove_punctuations(image_caption),'1']
    return quotes

def set_pipline(quotes):
    lda_model = LatentDirichletAllocation(learning_decay=1, n_components=5)
    vectorizer = TfidfVectorizer(max_df=0.75, stop_words="english",ngram_range=(1,2),norm='l1')
    topic_pipeline = Pipeline([('tfidf',vectorizer),('lda', lda_model)])
    trained = topic_pipeline
    trained.fit(quotes.list_tags)
    joblib.dump(trained,'top55.joblib')
    print('saved')


class GetQuote():
    def __init__(self,quotes):
        loaded_model1 = joblib.load('t5.joblib')
        self.top5 = loaded_model1
        loaded_model2 = joblib.load('n_min.joblib')
        self.nn_min = loaded_model2
        loaded_model3 = joblib.load('n_euc.joblib')
        self.nn_euc = loaded_model3
        self.tfidf_weight = None
        self.quotes = quotes
        self.image_topic = None
        self.own_tags = None

    def run(self):
        trained_topics = self.top5.transform(self.quotes.list_tags)
        for index,row in self.quotes.iterrows():
            self.quotes['topic'] = self.quotes.quote.copy()
        for index,row in self.quotes.iterrows():
            self.quotes.at[index, 'topic'] = np.where(trained_topics[index] == max(trained_topics[index]))[0][0]
        return self.quotes

    def top5_fuc(self,quotes):
        self.image_topic = int(quotes.iloc[-1, [-1]])
        only_topic = quotes[quotes.topic == self.image_topic]
        self.vectorizer = TfidfVectorizer(max_df=0.75, stop_words="english",ngram_range=(1,2),norm='l1')
        self.tfidf_weight = self.vectorizer.fit_transform(only_topic['list_tags'].values.astype('U'))
        image_index = -1
        min, indices = self.nn_min.kneighbors(self.tfidf_weight[image_index], n_neighbors = 100)
        neighbors_min = pd.DataFrame({'min': min.flatten(), 'id': indices.flatten()})
        top5_results = (only_topic.merge(neighbors_min, right_on = 'id', left_index = True).sort_values('min')[['quote', 'author']]).head()
        return top5_results

    def most_suitable(self,quotes , own_tags):
        print(quotes.topic)
        most_suiting = quotes.loc[quotes.topic != self.image_topic]
        most_suiting.iloc[-1] = [own_tags,'image','image',own_tags,'1',self.image_topic]
        image_index = -1
        euc, indices = self.nn_euc.kneighbors(self.tfidf_weight[image_index], n_neighbors = 100)
        neighbors_euc = pd.DataFrame({'euc': euc.flatten(), 'id': indices.flatten()})
        result_most_s = (most_suiting.merge(neighbors_euc, right_on = 'id', left_index = True).
                        sort_values('euc')[['quote', 'author']]).head(1)
        return result_most_s
if __name__ == "__main__":
    #cap_tr = ImageCaption()
    #cap = cap_tr.nlp
    cap = 'It will be added from Mohana´s'
    quotes = clean_data(cap)
    #set_pipline(quotes)
    getquoteclass = GetQuote
    trainer = GetQuote(quotes)
    #print(trainer.top5.transform(cap))
    quotess = trainer.run()
    '''print(trainer.top5_fuc(quotess))
    print(trainer.most_suitable(quotess,cap))
'''
