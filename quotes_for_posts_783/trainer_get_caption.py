from quotes_for_posts_783.trainer_get_class import ImageGroup
import joblib
import numpy as np
import matplotlib.image as mpimg
import os
import cv2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input




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
    def run(self):
        cap_list = []
        for i in self.group_lists:
            if i == 0:
                #model_0 = joblib.load('model.joblib')
                im = self.im
                im = cv2.resize(im, (299,299))
                im = np.expand_dims(im, 0)
                im = preprocess_input(im)
                #fea_vec = model_0.predict(im)
                fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
                img = fea_vec.reshape((1,2048))
                '''cap_list.append(greedySearch(img),
                            beam_search_predictions(image3, beam_index = 3),
                            beam_search_predictions(image3, beam_index = 5),
                            beam_search_predictions(image3, beam_index = 7),
                            beam_search_predictions(image3, beam_index = 10)
                            )'''
            if i == 1:
                #model_1 = joblib.load('model.joblib')
                im = self.im
                im = cv2.resize(im, (299,299))
                im = np.expand_dims(im, 0)
                im = preprocess_input(im)
                #fea_vec = model_1.predict(im)
                fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
                img = fea_vec.reshape((1,2048))
                '''cap_list.append(greedySearch(img),
                            beam_search_predictions(image3, beam_index = 3),
                            beam_search_predictions(image3, beam_index = 5),
                            beam_search_predictions(image3, beam_index = 7),
                            beam_search_predictions(image3, beam_index = 10)
                            )'''
            if i == 2:
                #model_2 = joblib.load('model.joblib')
                im = self.im
                im = cv2.resize(im, (299,299))
                im = np.expand_dims(im, 0)
                im = preprocess_input(im)
                #fea_vec = model_2.predict(im)
                fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
                img = fea_vec.reshape((1,2048))
                '''cap_list.append(greedySearch(img),
                            beam_search_predictions(image3, beam_index = 3),
                            beam_search_predictions(image3, beam_index = 5),
                            beam_search_predictions(image3, beam_index = 7),
                            beam_search_predictions(image3, beam_index = 10)
                            )'''
            if i == 3:
                #model_3 = joblib.load('model.joblib')
                im = self.im
                im = cv2.resize(im, (299,299))
                im = np.expand_dims(im, 0)
                im = preprocess_input(im)
                #fea_vec = model_3.predict(im)
                fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
                img = fea_vec.reshape((1,2048))
                '''cap_list.append(greedySearch(img),
                            beam_search_predictions(image3, beam_index = 3),
                            beam_search_predictions(image3, beam_index = 5),
                            beam_search_predictions(image3, beam_index = 7),
                            beam_search_predictions(image3, beam_index = 10)
                            )'''
            if i == 4:
                #model_4 = joblib.load('model.joblib')
                im = self.im
                im = cv2.resize(im, (299,299))
                im = np.expand_dims(im, 0)
                im = preprocess_input(im)
                #fea_vec = model_4.predict(im)
                fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
                img = fea_vec.reshape((1,2048))
                '''cap_list.append(greedySearch(img),
                            beam_search_predictions(image3, beam_index = 3),
                            beam_search_predictions(image3, beam_index = 5),
                            beam_search_predictions(image3, beam_index = 7),
                            beam_search_predictions(image3, beam_index = 10)
                            )'''
        return cap_list

    def nlp(self):
        cap_list = self.run()
