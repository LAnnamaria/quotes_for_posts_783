from quotes_for_posts_783.trainer_get_class import ImageGroup
import joblib
import numpy as np
import matplotlib.image as mpimg
import os
import cv2
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
from rake_nltk import Rake




class ImageCaption():
    def __init__(self):
        path = os.getcwd()
        '''self.model_0 = joblib.load('model.joblib')
        self.model_1 = joblib.load('model.joblib')
        self.model_2 = joblib.load('model.joblib')
        self.model_3 = joblib.load('model.joblib')
        self.model_4 = joblib.load('model.joblib')'''
        self.im = mpimg.imread(f"{path}/tempDir")
        tr = ImageGroup(path)
        self.group_lists = tr.image_group()
        self.max_length = None
        self.wordtoix = None
        self.ixtoword = None
    def greedySearch(self,photo,model):
        in_text = 'startseq'
        for i in range(self.max_length):
            sequence = [self.wordtoix[w] for w in in_text.split() if w in self.wordtoix]
            sequence = pad_sequences([sequence], maxlen=self.max_length)
            yhat = model.predict([photo,sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = self.ixtoword[yhat]
            in_text += ' ' + word
            if word == 'endseq':
                break

        final = in_text.split()
        final = final[1:-1]
        final = ' '.join(final)
        return final
    def beam_search_predictions(self,image,model, beam_index = 3):
        start = [self.wordtoix["startseq"]]
        start_word = [[start, 0.0]]
        while len(start_word[0][0]) < self.max_length:
            temp = []
            for s in start_word:
                par_caps = pad_sequences([s[0]], maxlen=self.max_length, padding='post')
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
        intermediate_caption = [self.ixtoword[i] for i in start_word]
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
                #model_new = joblib.load('model.joblib')
                im = self.im
                im = cv2.resize(im, (299,299))
                im = np.expand_dims(im, 0)
                im = preprocess_input(im)
                #fea_vec = model_new.predict(im)
                fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
                img = fea_vec.reshape((1,2048))
                cap_list.append(self.greedySearch(img,self.model_0),
                            self.beam_search_predictions(img,self.model_0, beam_index = 3),
                            self.beam_search_predictions(img,self.model_0, beam_index = 5),
                            self.beam_search_predictions(img,self.model_0, beam_index = 7),
                            self.beam_search_predictions(img,self.model_0, beam_index = 10)
                            )
            if i == 1:
                #model_new = joblib.load('model.joblib')
                im = self.im
                im = cv2.resize(im, (299,299))
                im = np.expand_dims(im, 0)
                im = preprocess_input(im)
                #fea_vec = model_new.predict(im)
                fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
                img = fea_vec.reshape((1,2048))
                cap_list.append(self.greedySearch(img,self.model_1),
                            self.beam_search_predictions(img,self.model_1, beam_index = 3),
                            self.beam_search_predictions(img,self.model_1, beam_index = 5),
                            self.beam_search_predictions(img,self.model_1, beam_index = 7),
                            self.beam_search_predictions(img,self.model_1, beam_index = 10)
                            )
            if i == 2:
                #model_new = joblib.load('model.joblib')
                im = self.im
                im = cv2.resize(im, (299,299))
                im = np.expand_dims(im, 0)
                im = preprocess_input(im)
                #fea_vec = model_new.predict(im)
                fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
                img = fea_vec.reshape((1,2048))
                cap_list.append(self.greedySearch(img,self.model_2),
                            self.beam_search_predictions(img,self.model_2, beam_index = 3),
                            self.beam_search_predictions(img,self.model_2, beam_index = 5),
                            self.beam_search_predictions(img,self.model_2, beam_index = 7),
                            self.beam_search_predictions(img,self.model_2, beam_index = 10))
            if i == 3:
                #model_new = joblib.load('model.joblib')
                im = self.im
                im = cv2.resize(im, (299,299))
                im = np.expand_dims(im, 0)
                im = preprocess_input(im)
                #fea_vec = model_new.predict(im)
                fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
                img = fea_vec.reshape((1,2048))
                cap_list.append(self.greedySearch(img,self.model_3),
                            self.beam_search_predictions(img,self.model_3, beam_index = 3),
                            self.beam_search_predictions(img,self.model_3, beam_index = 5),
                            self.beam_search_predictions(img,self.model_3, beam_index = 7),
                            self.beam_search_predictions(img,self.model_3, beam_index = 10)
                            )
            if i == 4:
                #model_new = joblib.load('model.joblib')
                im = self.im
                im = cv2.resize(im, (299,299))
                im = np.expand_dims(im, 0)
                im = preprocess_input(im)
                #fea_vec = model_new.predict(im)
                fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
                img = fea_vec.reshape((1,2048))
                cap_list.append(self.greedySearch(img,self.model_4),
                            self.beam_search_predictions(img,self.model_4, beam_index = 3),
                            self.beam_search_predictions(img,self.model_4, beam_index = 5),
                            self.beam_search_predictions(img,self.model_4, beam_index = 7),
                            self.beam_search_predictions(img,self.model_4, beam_index = 10)
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
