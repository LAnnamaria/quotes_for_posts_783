from lzma import MODE_FAST
import pandas as pd
import quotes_for_posts_783.data.quotesdata as qd
import quotes_for_posts_783.utils as u
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import joblib

class QuotesTrainer():
    def __init__(self, quotes):
        """
            quotes is a dataframe with the image_caption already incorporated at index -1
            
        """
        self.vectorizer = TfidfVectorizer(max_df=0.75, min_df=0.1, stop_words="english",ngram_range=(1,2),norm='l1')
        self.image_caption = image_caption
        self.quotes = quotes
        self.image_topic = None
        self.own_tags = None

   
    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        lda_model = LatentDirichletAllocation(learning_decay=1, n_components=5)
        topic_pipeline = Pipeline([('tfidf', self.vectorizer),('lda', lda_model)])
        joblib.dump(topic_pipeline,'top5.joblib')
        

    def run(self):
        """set and train the pipeline"""
        trained = joblib.load('top5.joblib')
        trained_topics = trained.fit_transform(self.quotes.list_tags)
        for index,row in self.quotes.iterrows():
            self.quotes['topic'] = self.quotes.quote.copy()
        for index,row in self.quotes.iterrows():
            self.quotes.at[index, 'topic'] = np.where(trained_topics[index] == max(trained_topics[index]))[0][0]
        return self.quotes

    def top5(self):
        """evaluates the pipeline on df_test and return the RMSE"""
        self.image_topic = int(self.quotes.iloc[-1, [-1]])
        only_topic = self.quotes[self.quotes.topic == self.image_topic]
        tfidf_weight = self.vectorizer.fit_transform(only_topic['list_tags'].values.astype('U'))
        nn_min = NearestNeighbors(metric = 'minkowski')
        joblib.dump(nn_min,'nn_min.joblib')
        nn_min = joblib.load('nn_min.joblib')
        nn_min.fit(tfidf_weight)
        image_index = -1
        min, indices = nn_min.kneighbors(tfidf_weight[image_index], n_neighbors = 100)
        neighbors_min = pd.DataFrame({'min': min.flatten(), 'id': indices.flatten()})
        top5 = (only_topic.merge(neighbors_min, right_on = 'id', left_index = True).sort_values('min')[['quote', 'author']]).head()
        return top5

    def most_suitable(self,own_tags):
        most_suiting = self.quotes[self.quotes.topic != self.image_topic]
        own_tags = own_tags
        most_suiting.iloc[-1] = [own_tags,'image','image',own_tags,'1',self.image_topic]
        tfidf_weight = self.vectorizer.fit_transform(most_suiting['list_tags'].values.astype('U'))
        nn_euc = NearestNeighbors(metric = 'euclidean')
        joblib.dump(nn_euc,'nn_euc.joblib')
        nn_euc = joblib.load('nn_euc.joblib')
        nn_euc.fit(tfidf_weight)
        image_index = -1
        euc, indices = nn_euc.kneighbors(tfidf_weight[image_index], n_neighbors = 100)
        neighbors_euc = pd.DataFrame({'euc': euc.flatten(), 'id': indices.flatten()})
        result_most_s = (most_suiting.merge(neighbors_euc, right_on = 'id', left_index = True).
                        sort_values('euc')[['quote', 'author']]).head(1)
        return result_most_s

if __name__ == "__main__":
    image_caption = 'It will be added from Mohana´s model'
    quotes = qd.get_quotes_data()
    quotes = qd.clean_data(quotes)
    quotes = u.image_cap_to_quotes(quotes,image_caption)
    
    trainer = QuotesTrainer(quotes)
    #trainer.set_pipeline()
    trainer.run()
    print(trainer.top5())
    satisfied = input('Are you satisfied with any of these quotes? Y/N')
    if satisfied == 'Y':
        print('See you next time!')
    else:
        own_tags = input("Please give me 5 words that are descriptive of your picture:")
        print(trainer.most_suitable(own_tags))
    

    
