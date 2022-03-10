import pandas as pd
import matplotlib.image as mpimg
import os
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2
import numpy as np
import joblib
class ImageGroup():
    def __init__(self,path):
        loaded_model = joblib.load('model.joblib')
        self.model = loaded_model
        self.path = path
    def image_class(self):
        path = os.getcwd()
        im = mpimg.imread(f"{self.path}/tempDir")
        im = cv2.resize(im, (224, 224))
        ## Expanding image dims so this represents 1 sample
        im = np.expand_dims(im, 0)
        im = preprocess_input(im)
        classes_xx=np.argmax(self.model.predict(im),axis=1)[0]
        #print(classes_xx)
        return classes_xx
    def image_group(self):
        grouped_classes_df = pd.read_csv(f"{self.path}/raw_data/grouped_classes_df.csv")
        classes_xx = self.image_class()
        pr_im = grouped_classes_df.loc[grouped_classes_df['classnumber'] == classes_xx]
        group_lists = list(pr_im['clusternumber'])[0]
        print(group_lists)
        return group_lists
if __name__ == "__main__":
    path = os.getcwd()
    tr = ImageGroup(path)
    tr.image_group()
