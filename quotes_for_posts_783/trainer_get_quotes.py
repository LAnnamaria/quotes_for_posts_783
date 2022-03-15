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

class GetData():
    def __init__(self):
        pass

    def remove_punctuations(self,text):
        '''removes punctuation from a given text'''
        for punctuation in string.punctuation:
            text = text.replace(punctuation, '')
        return text

    def clean_data(self, image_caption , test=False):
        path = os.getcwd()
        #path = '/home/morad/code/LAnnamaria/quotes_for_posts_783'
        quotes =  pd.read_csv(f"{path}/raw_data/cleaned_quotes.csv")
        #quotes = quotes.head(10000)
        print(quotes.head())
        quotes.iloc[-1] = [image_caption, 'image','image',self.remove_punctuations(image_caption),'1']
        return quotes

def set_pipline(quotes):
    lda_model = LatentDirichletAllocation(learning_decay=1, n_components=5)
    vectorizer = TfidfVectorizer(max_df=0.75, stop_words="english",ngram_range=(1,2),norm='l1')
    topic_pipeline = Pipeline([('tfidf',vectorizer),('lda', lda_model)])
    topic_pipeline.fit(quotes.list_tags)
    joblib.dump(topic_pipeline,'top55.joblib')
    print('saved')


class GetQuote():
    def __init__(self,quotes):
        loaded_model1 = joblib.load('quotes_for_posts_783_top5.joblib')
        self.top5 = loaded_model1
        loaded_vec_top5 = joblib.load('quotes_for_posts_783_vectorizer_top5.joblib')
        self.vec_top5 = loaded_vec_top5
        loaded_vec_ms = joblib.load('quotes_for_posts_783_vectorizer_ms.joblib')
        self.vec_ms = loaded_vec_ms
        loaded_model2 = joblib.load('quotes_for_posts_783_nn_min.joblib')
        self.nn_min = loaded_model2
        loaded_model3 = joblib.load('quotes_for_posts_783_nn_euc.joblib')
        self.nn_euc = loaded_model3
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

    def top5_fuc(self):
        self.image_topic = int(self.quotes.iloc[-1, [-1]])
        only_topic = self.quotes[self.quotes.topic == self.image_topic]
        tfidf_weight = self.vec_top5.transform(only_topic['list_tags'].values.astype('U'))
        image_index = -1
        min, indices = self.nn_min.kneighbors(tfidf_weight[image_index], n_neighbors = 100)
        neighbors_min = pd.DataFrame({'min': min.flatten(), 'id': indices.flatten()})
        top5_results = (only_topic.merge(neighbors_min, right_on = 'id', left_index = True).sort_values('min')[['quote', 'author']]).head()
        return top5_results

    def most_suitable(self, own_tags):
        self.image_topic = int(self.quotes.iloc[-1, [-1]])
        most_suiting = self.quotes.loc[self.quotes.topic != self.image_topic]
        most_suiting.iloc[-1] = [own_tags,'image','image',own_tags,'1',self.image_topic]
        tfidf_weight = self.vec_ms.transform(most_suiting['list_tags'].values.astype('U'))
        image_index = -1
        euc, indices = self.nn_euc.kneighbors(tfidf_weight[image_index], n_neighbors = 100)
        neighbors_euc = pd.DataFrame({'euc': euc.flatten(), 'id': indices.flatten()}).set_index('id')
        result_most_s = (most_suiting.join(neighbors_euc).sort_values('euc')[['quote', 'author']]).head(1)
        return result_most_s

if __name__ == "__main__":
    #cap_tr = ImageCaption()
    #cap = cap_tr.nlp
    cap = 'Morad is skydiving in Prague'
    q = GetData()
    quotes = q.clean_data(cap)
    #set_pipline(quotes)
    trainer = GetQuote(quotes)
    trainer.run()
    print(trainer.top5_fuc())
    print(trainer.most_suitable(cap))
