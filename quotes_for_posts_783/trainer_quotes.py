from cProfile import run
from lzma import MODE_FAST
import pandas as pd
import quotes_for_posts_783.quotesdata as qd
import quotes_for_posts_783.utils as u
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import joblib
from google.cloud import storage


BUCKET_NAME = 'quotes_for_posts_783'

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -
STORAGE_LOCATION_1 = 'quotes_for_posts_783/top5.joblib'
STORAGE_LOCATION_2 = 'quotes_for_posts_783/nn_min.joblib'
STORAGE_LOCATION_3 = 'quotes_for_posts_783/nn_euc.joblib'
STORAGE_LOCATION_4 = 'quotes_for_posts_783/vectorizer_top5.joblib'
STORAGE_LOCATION_5 = 'quotes_for_posts_783/vectorizer_ms.joblib'

def upload_model_to_gcp_1():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STORAGE_LOCATION_1)
    blob.upload_from_filename('top5.joblib')
def upload_model_to_gcp_2():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STORAGE_LOCATION_2)
    blob.upload_from_filename('nn_min.joblib')
def upload_model_to_gcp_3():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STORAGE_LOCATION_3)
    blob.upload_from_filename('nn_euc.joblib')
def upload_vectorizertop5_to_gcp_2():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STORAGE_LOCATION_4)
    blob.upload_from_filename('vectorizer_top5.joblib')
def upload_vectorizerms_to_gcp_3():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STORAGE_LOCATION_5)
    blob.upload_from_filename('vectorizer_ms.joblib')


class QuotesTrainer():
    def __init__(self, quotes):
        """
            quotes is a dataframe with the image_caption already incorporated at index -1

        """
        self.vectorizer = None
        self.quotes = quotes
        self.image_topic = None
        self.own_tags = None


    '''def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        lda_model = LatentDirichletAllocation(learning_decay=1, n_components=5)
        topic_pipeline = Pipeline([('tfidf', self.vectorizer),('lda', lda_model)])
        joblib.dump(topic_pipeline,'top5.joblib')
        upload_model_to_gcp_1()
        print(f"uploaded top5.joblib to gcp cloud storage under \n => {STORAGE_LOCATION_1}")'''

    def run(self):
        """set and train the pipeline """

        lda_model = LatentDirichletAllocation(learning_decay=1, n_components=5)
        self.vectorizer = TfidfVectorizer(max_df=0.75, stop_words="english",ngram_range=(1,2),norm='l1')
        topic_pipeline = Pipeline([('tfidf', self.vectorizer),('lda', lda_model)])
        trained = topic_pipeline
        trained_topics = trained.fit(self.quotes.list_tags)
        joblib.dump(trained_topics,'top5.joblib')
        upload_model_to_gcp_1()
        #print(f"uploaded top5.joblib to gcp cloud storage under \n => {STORAGE_LOCATION_1}")
        trained_topics = trained.transform(self.quotes.list_tags)
        for index,row in self.quotes.iterrows():
            self.quotes['topic'] = self.quotes.quote.copy()
        for index,row in self.quotes.iterrows():
            self.quotes.at[index, 'topic'] = np.where(trained_topics[index] == max(trained_topics[index]))[0][0]
        return self.quotes

    def top5(self, quotes):
        self.quotes = quotes
        """evaluates the pipeline on df_test and return the RMSE"""
        vectorizer_top5 = TfidfVectorizer(max_df=0.75, stop_words="english",ngram_range=(1,2),norm='l1')
        self.image_topic = int(self.quotes.iloc[-1, [-1]])
        only_topic = self.quotes[self.quotes.topic == self.image_topic]
        tfidf_weight = vectorizer_top5.fit_transform(only_topic['list_tags'].values.astype('U'))
        joblib.dump(vectorizer_top5,'vectorizer_top5.joblib')
        upload_vectorizertop5_to_gcp_2()
        #print(f"uploaded vectorizer_top5.joblib to gcp cloud storage under \n => {STORAGE_LOCATION_4}")
        nn_min = NearestNeighbors(metric = 'minkowski')
        nn_min.fit(tfidf_weight)
        joblib.dump(nn_min,'nn_min.joblib')
        upload_model_to_gcp_2()
        #print(f"uploaded nn_min.joblib to gcp cloud storage under \n => {STORAGE_LOCATION_2}")
        image_index = -1
        min, indices = nn_min.kneighbors(tfidf_weight[image_index], n_neighbors = 100)
        neighbors_min = pd.DataFrame({'min': min.flatten(), 'id': indices.flatten()})
        top5 = (only_topic.merge(neighbors_min, right_on = 'id', left_index = True).sort_values('min')[['quote', 'author']]).head()
        return top5

    def most_suitable(self,quotes , own_tags):
        most_suiting = quotes.loc[quotes.topic != self.image_topic]
        most_suiting.iloc[-1] = [own_tags,'image','image',own_tags,'1',self.image_topic]
        vectorizer_ms = TfidfVectorizer(max_df=0.75, stop_words="english",ngram_range=(1,2),norm='l1')
        tfidf_weight = vectorizer_ms.fit_transform(most_suiting['list_tags'].values.astype('U'))
        joblib.dump(vectorizer_ms,'vectorizer_ms.joblib')
        upload_vectorizerms_to_gcp_3()
        #print(f"uploaded vectorizer_ms.joblib to gcp cloud storage under \n => {STORAGE_LOCATION_5}")
        nn_euc = NearestNeighbors(metric = 'euclidean')
        nn_euc.fit(tfidf_weight)
        joblib.dump(nn_euc,'nn_euc.joblib')
        upload_model_to_gcp_3()
        #print(f"uploaded nn_euc.joblib to gcp cloud storage under \n => {STORAGE_LOCATION_3}")
        image_index = -1
        euc, indices = nn_euc.kneighbors(tfidf_weight[image_index], n_neighbors = 100)
        neighbors_euc = pd.DataFrame({'euc': euc.flatten(), 'id': indices.flatten()})
        result_most_s = (most_suiting.merge(neighbors_euc, right_on = 'id', left_index = True).
                        sort_values('euc')[['quote', 'author']]).head(1)
        return result_most_s

if __name__ == "__main__":
    image_caption = 'It will be added from Mohana´s'
    quotes = qd.clean_data()
    quotes = u.image_cap_to_quotes(quotes,image_caption)
    trainer = QuotesTrainer(quotes)
    quotess = trainer.run()
    trainer.top5(quotess)
    #trainer.most_suitable(image_caption)
    trainer.most_suitable(quotess,image_caption)

   # satisfied = input('Are you satisfied with any of these quotes? Y/N')
    #if satisfied == 'Y':
    #   print('See you next time!')
    #else:
    #   own_tags = input("Please give me 5 words that are descriptive of your picture:")
     #  print(trainer.most_suitable(image_caption))
