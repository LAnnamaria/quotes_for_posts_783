from quotes_for_posts_783.trainer_get_class import ImageGroup
import joblib
import numpy as np
import matplotlib.image as mpimg
import os
import cv2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
from rake_nltk import Rake
from keras.models import Model
import string
import pandas as pd




class ImageCaption():
    def __init__(self):
        path = os.getcwd()
        '''self.model_0 = joblib.load('model.joblib')
        self.model_1 = joblib.load('model.joblib')
        self.model_2 = joblib.load('model.joblib')
        self.model_3 = joblib.load('model.joblib')
        self.model_4 = joblib.load('model.joblib')'''
        model = InceptionV3(weights='imagenet')
        self.model_new = Model(model.input, model.layers[-2].output)
        self.im = mpimg.imread(f"{path}/tempDir")
        tr = ImageGroup(path)
        self.group_lists = tr.image_group()
        self.df0 = pd.read_csv(f"{path}/raw_data/df_images_0.csv")
        self.df1 = pd.read_csv(f"{path}/raw_data/df_images_1.csv")
        self.df2 = pd.read_csv(f"{path}/raw_data/df_images_2.csv")
        self.df3 = pd.read_csv(f"{path}/raw_data/df_images_3.csv")
        self.df4 = pd.read_csv(f"{path}/raw_data/df_images_4.csv")
    def get_params(self,df):
        #df['image_name'] = df['image_name'].map(str)
        train = list(df['image_name'].map(str))
        descriptions = df.groupby('image_name')['comments'].apply(list).to_dict()
        table = str.maketrans('', '', string.punctuation)
        for key, desc_list in descriptions.items():
            for i in range(len(desc_list)):
                desc = desc_list[i]
                desc = desc.split()
                desc = [word.lower() for word in desc]
                desc = [w.translate(table) for w in desc]
                desc_list[i] =  ' '.join(desc)

        vocabulary = set()
        for key in descriptions.keys():
                [vocabulary.update(d.split()) for d in descriptions[key]]

        lines = list()
        for key, desc_list in descriptions.items():
            for desc in desc_list:
                lines.append(f'{key} {desc}')
        new_descriptions = '\n'.join(lines)

        train_descriptions = dict()
        for line in new_descriptions.split('\n'):
            tokens = line.split()
            image_id, image_desc = tokens[0], tokens[1:]
            if image_id in train:
                if image_id not in train_descriptions:
                    train_descriptions[image_id] = list()
                desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
                train_descriptions[image_id].append(desc)
        all_train_captions = []
        for key, val in train_descriptions.items():
            for cap in val:
                all_train_captions.append(cap)

        word_count_threshold = 10
        word_counts = {}
        nsents = 0
        for sent in all_train_captions:
            nsents += 1
            for w in sent.split(' '):
                word_counts[w] = word_counts.get(w, 0) + 1
        vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

        ixtoword = {}
        wordtoix = {}
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1
        vocab_size = len(ixtoword) + 1

        all_desc = list()
        for key in train_descriptions.keys():
            [all_desc.append(d) for d in train_descriptions[key]]
        lines = all_desc
        max_length = max([len(d.split()) for d in lines], default=0)
        return wordtoix , ixtoword , max_length
    def greedySearch(self,photo,model , wordtoix ,ixtoword, max_length):
        in_text = 'startseq'
        for i in range(max_length):
            sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
            sequence = pad_sequences([sequence], maxlen=max_length)
            yhat = model.predict([photo,sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = ixtoword[yhat]
            in_text += ' ' + word
            if word == 'endseq':
                break

        final = in_text.split()
        final = final[1:-1]
        final = ' '.join(final)
        return final


    def beam_search_predictions(self,image,model, wordtoix ,ixtoword, max_length ,beam_index = 3):
        start = [wordtoix["startseq"]]
        start_word = [[start, 0.0]]
        while len(start_word[0][0]) < max_length:
            temp = []
            for s in start_word:
                par_caps = pad_sequences([s[0]], maxlen= max_length, padding='post')
                preds = model.predict([image,par_caps], verbose=0)
                word_preds = np.argsort(preds[0])[-beam_index:]
                # Getting the top <beam_index>(n) predictions and creating a
                # new list so as to put them via the model again
                for w in word_preds:
                    next_cap, prob = s[0][:], s[1]
                    next_cap.append(w)
                    prob += preds[0][w]
                    temp.append([next_cap, prob])

            start_word = temp
            # Sorting according to the probabilities
            start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
            # Getting the top words
            start_word = start_word[-beam_index:]

        start_word = start_word[-1][0]
        intermediate_caption = [ixtoword[i] for i in start_word]
        final_caption = []

        for i in intermediate_caption:
            if i != 'endseq':
                final_caption.append(i)
            else:
                break

        final_caption = ' '.join(final_caption[1:])
        return final_caption
    def run(self):
        cap_list = []
        for i in self.group_lists:
            if i == 0:
                wordtoix ,ixtoword, max_length = self.get_params(self.df0)
                im = self.im
                im = cv2.resize(im, (299,299))
                im = np.expand_dims(im, 0)
                im = preprocess_input(im)
                fea_vec = self.model_new.predict(im)
                fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
                img = fea_vec.reshape((1,2048))
                cap_list.append(self.greedySearch(img,self.model_0,wordtoix ,ixtoword, max_length),
                            self.beam_search_predictions(img,self.model_0,wordtoix ,ixtoword, max_length, beam_index = 3),
                            self.beam_search_predictions(img,self.model_0,wordtoix ,ixtoword, max_length, beam_index = 5),
                            self.beam_search_predictions(img,self.model_0,wordtoix ,ixtoword, max_length, beam_index = 7),
                            self.beam_search_predictions(img,self.model_0,wordtoix ,ixtoword, max_length, beam_index = 10)
                            )
            if i == 1:
                wordtoix ,ixtoword, max_length = self.get_params(self.df1)
                im = self.im
                im = cv2.resize(im, (299,299))
                im = np.expand_dims(im, 0)
                im = preprocess_input(im)
                fea_vec = self.model_new.predict(im)
                fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
                img = fea_vec.reshape((1,2048))
                cap_list.append(self.greedySearch(img,self.model_1 ,wordtoix ,ixtoword, max_length),
                            self.beam_search_predictions(img,self.model_1,wordtoix ,ixtoword, max_length, beam_index = 3),
                            self.beam_search_predictions(img,self.model_1,wordtoix ,ixtoword, max_length, beam_index = 5),
                            self.beam_search_predictions(img,self.model_1,wordtoix ,ixtoword, max_length, beam_index = 7),
                            self.beam_search_predictions(img,self.model_1,wordtoix ,ixtoword, max_length, beam_index = 10)
                            )
            if i == 2:
                wordtoix ,ixtoword, max_length = self.get_params(self.df2)
                im = self.im
                im = cv2.resize(im, (299,299))
                im = np.expand_dims(im, 0)
                im = preprocess_input(im)
                fea_vec = self.model_new.predict(im)
                fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
                img = fea_vec.reshape((1,2048))
                cap_list.append(self.greedySearch(img,self.model_2,wordtoix ,ixtoword, max_length),
                            self.beam_search_predictions(img,self.model_2,wordtoix ,ixtoword, max_length, beam_index = 3),
                            self.beam_search_predictions(img,self.model_2,wordtoix ,ixtoword, max_length, beam_index = 5),
                            self.beam_search_predictions(img,self.model_2,wordtoix ,ixtoword, max_length, beam_index = 7),
                            self.beam_search_predictions(img,self.model_2,wordtoix ,ixtoword, max_length, beam_index = 10))
            if i == 3:
                wordtoix ,ixtoword, max_length = self.get_params(self.df3)
                im = self.im
                im = cv2.resize(im, (299,299))
                im = np.expand_dims(im, 0)
                im = preprocess_input(im)
                fea_vec = self.model_new.predict(im)
                fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
                img = fea_vec.reshape((1,2048))
                cap_list.append(self.greedySearch(img,self.model_3,wordtoix ,ixtoword, max_length),
                            self.beam_search_predictions(img,self.model_3,wordtoix ,ixtoword, max_length, beam_index = 3),
                            self.beam_search_predictions(img,self.model_3,wordtoix ,ixtoword, max_length, beam_index = 5),
                            self.beam_search_predictions(img,self.model_3,wordtoix ,ixtoword, max_length, beam_index = 7),
                            self.beam_search_predictions(img,self.model_3,wordtoix ,ixtoword, max_length, beam_index = 10)
                            )
            if i == 4:
                wordtoix ,ixtoword, max_length = self.get_params(self.df4)
                im = self.im
                im = cv2.resize(im, (299,299))
                im = np.expand_dims(im, 0)
                im = preprocess_input(im)
                fea_vec = self.model_new.predict(im)
                fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
                img = fea_vec.reshape((1,2048))
                cap_list.append(self.greedySearch(img,self.model_4,wordtoix ,ixtoword, max_length),
                            self.beam_search_predictions(img,self.model_4,wordtoix ,ixtoword, max_length, beam_index = 3),
                            self.beam_search_predictions(img,self.model_4,wordtoix ,ixtoword, max_length, beam_index = 5),
                            self.beam_search_predictions(img,self.model_4,wordtoix ,ixtoword, max_length, beam_index = 7),
                            self.beam_search_predictions(img,self.model_4,wordtoix ,ixtoword, max_length, beam_index = 10)
                            )
        return cap_list

    def nlp(self):
        cap_list = self.run()
        cap_srt = ','.join(cap_list)
        rake_nltk_var = Rake()
        rake_nltk_var.extract_keywords_from_text(cap_srt)
        keyword_extracted = rake_nltk_var.get_ranked_phrases()
        keyword_extracted = list(dict.fromkeys(keyword_extracted))
        keyword_extracted = ','.join(keyword_extracted)
        return keyword_extracted

if __name__ == "__main__":
    tr = ImageCaption()
    tr.nlp()
