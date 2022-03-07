import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50 , resnet50
from sklearn.cluster import AgglomerativeClustering
import joblib

'''def store_data():
    data = {
    'flattenPhoto' : [],
    'photoclass' : [],
    'photoclasskmeans' : [],
    'image_name' : []
    }
    return data'''
def classefier_model():
    MyModel = Sequential()
    MyModel.add(ResNet50(
    include_top = False, weights='imagenet',    pooling='avg',
    ))
    MyModel.layers[0].trainable = False
    joblib.dump(MyModel, 'model.joblib')

def LoadDataAndDoEssentials(path, h, w):
    loaded_model = joblib.load('model.joblib')
    img = mpimg.imread(path)
    img = cv2.resize(img, (h, w))
    ## Expanding image dims so this represents 1 sample
    img = np.expand_dims(img, 0)
    img = resnet50.preprocess_input(img)
    extractedFeatures = loaded_model.predict(img)
    classes_x=np.argmax(extractedFeatures,axis=1)[0]
    extractedFeatures = np.array(extractedFeatures)
    store_data['photoclass'].append(classes_x)
    store_data['flattenPhoto'].append(extractedFeatures.flatten())

def ReadAndStoreMyImages(path):
    df = pd.read_csv('/home/morad/code/LAnnamaria/quotes_for_posts_783/raw_data/images_name.csv')
    df_im = df.tail(10)
    images = list(df_im['image_name'])
    for i in images:
        imagePath = f"{path}/{i}.jpg"
        store_data['image_name'].append(i)
        LoadDataAndDoEssentials(imagePath, 224, 224)

def culster_model():
    ReadAndStoreMyImages('/home/morad/code/LAnnamaria/quotes_for_posts_783/raw_data/10_images')
    Training_Feature_vector = np.array(store_data['flattenPhoto'], dtype = 'float64')
    kmeans = AgglomerativeClustering(n_clusters = 5)
    k = kmeans.fit_predict(Training_Feature_vector)
    store_data['photoclasskmeans'] = k

def display_data():
    data = pd.DataFrame(store_data)
    data = data.sort_values('photoclass').reset_index()
    g_im = data.groupby('photoclasskmeans')
    g_im = pd.DataFrame(g_im)
    for i in range(5):
        if i == 0:
            img_0 = list(g_im[1][i]['image_name'])
        if i == 1:
            img_1 = list(g_im[1][i]['image_name'])
        if i == 2:
            img_2 = list(g_im[1][i]['image_name'])
        if i == 3:
            img_3 = list(g_im[1][i]['image_name'])
        if i == 4:
            img_4 = list(g_im[1][i]['image_name'])
    print(img_0)
    print(img_1)
    print(img_2)
    print(img_3)
    print(img_4)

    return img_0, img_1, img_2, img_3, img_4

if __name__ == "__main__":
    store_data = {
    'flattenPhoto' : [],
    'photoclass' : [],
    'photoclasskmeans' : [],
    'image_name' : []
    }
    culster_model()
    display_data()
