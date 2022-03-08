import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import cv2
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50 , resnet50
from sklearn.cluster import AgglomerativeClustering
import joblib
from google.cloud import storage
#import os
### GCP configuration - - - - - - - - - - - - - - - - - - -

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'quotes_for_posts_783'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
# /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
# or if you want to use the full dataset (you need need to upload it first of course)
BUCKET_TRAIN_DATA_PATH = 'raw_data/image_dataset'
BUCKET_TRAIN_DATA_PATH_1= 'raw_data/image_dataset/images_name.csv'
BUCKET_TRAIN_DATA_PATH_2 ='raw_data/image_dataset/images_comments.csv'
BUCKET_TRAIN_DATA_PATH_3 ='raw_data/image_dataset/image_dataset'
##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'image_classification'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# not required here

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -
STORAGE_LOCATION = 'quotes_for_posts_783/trainer_images_lists_model.joblib'

def get_data():
    """method to get the training data (or a portion of it) from google cloud bucket"""
    df_im = pd.read_csv(f'gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH_1}')

    df_comments = pd.read_csv(f'gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH_2}')
    df_comments
    return df_im, df_comments

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )

def store_data():
    data = {
    'flattenPhoto' : [],
    'photoclass' : [],
    'photoclasskmeans' : [],
    'image_name' : []
    }
    return data
def upload_model_to_gcp():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STORAGE_LOCATION)
    blob.upload_from_filename('trainer_images_lists_model.joblib')

def classefier_model():
    MyModel = Sequential()
    MyModel.add(ResNet50(
    include_top = False, weights='imagenet',    pooling='avg',
    ))
    MyModel.layers[0].trainable = False
    joblib.dump(MyModel, 'trainer_images_lists_model.joblib')
    upload_model_to_gcp()
    print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")


def LoadDataAndDoEssentials(img_blob, h, w):
    loaded_model = joblib.load('trainer_images_lists_model.joblib')
    img = mpimg.imread(img_blob)
    img = cv2.resize(img, (h, w))
    ## Expanding image dims so this represents 1 sample
    img = np.expand_dims(img, 0)
    img = resnet50.preprocess_input(img)
    extractedFeatures = loaded_model.predict(img)
    classes_x=np.argmax(extractedFeatures,axis=1)[0]
    extractedFeatures = np.array(extractedFeatures)
    store_data['photoclass'].append(classes_x)
    store_data['flattenPhoto'].append(extractedFeatures.flatten())

def ReadAndStoreMyImages(df_im):
    #pathi = os. getcwd()
    images = list(df_im['image_name'])
    for i in images:
        imagePath = f'{BUCKET_TRAIN_DATA_PATH_3}/{i}.jpg'
        img_blob = download_blob(BUCKET_NAME, imagePath, '../quotes_for_posts_783/data')
        store_data['image_name'].append(i)
        img_temp_path = '../quotes_for_posts_783/data'
        LoadDataAndDoEssentials(img_temp_path, 224, 224)

def culster_model(df_im):
    ReadAndStoreMyImages(df_im)
    Training_Feature_vector = np.array(store_data['flattenPhoto'], dtype = 'float64')
    kmeans = AgglomerativeClustering(n_clusters = 5)
    k = kmeans.fit_predict(Training_Feature_vector)
    store_data['photoclasskmeans'] = k

def display_data():
    data = pd.DataFrame(store_data)
    data = data.sort_values('photoclass').reset_index()
    print(data.shape)
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
    return img_0, img_1, img_2, img_3, img_4

def convert_df(df_comments):
    #path = os. getcwd()
    df_name = {
        'image_name' : [],
        'comments' : []
    }

    for name in image_list:
        for kn in range(5):
            df_name['image_name'].append(name)
            df_name['comments'].append(df_comments.loc[lambda df_comments: df_comments['image_name'] == name].reset_index()['comment'][kn])

    df_name['image_name'] = np.array(df_name['image_name'])
    df_name['comments'] = np.array(df_name['comments'])
    df_name = pd.DataFrame(df_name)
    print('convertet')
    #path = os. getcwd()
    df_name.to_csv(f'gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}/{df_name}.csv',index=False)

def get_clust():
    data = pd.DataFrame(store_data)
    data = data.sort_values('photoclass').reset_index()
    for i in range(5):
        if i == 0:
            data_0 = data.loc[data['photoclasskmeans'] == i].groupby('photoclass').count()
            data_0 = data_0.reset_index()
            data_0['photoclasskmeans'] = data_0['photoclasskmeans'].apply(lambda x : i)
        elif i == 1:
            data_1 = data.loc[data['photoclasskmeans'] == i].groupby('photoclass').count()
            data_1 = data_1.reset_index()
            data_1['photoclasskmeans'] = data_1['photoclasskmeans'].apply(lambda x : i)
        elif i == 2:
            data_2 = data.loc[data['photoclasskmeans'] == i].groupby('photoclass').count()
            data_2 = data_2.reset_index()
            data_2['photoclasskmeans'] = data_2['photoclasskmeans'].apply(lambda x : i)
        elif i == 3:
            data_3 = data.loc[data['photoclasskmeans'] == i].groupby('photoclass').count()
            data_3 = data_3.reset_index()
            data_3['photoclasskmeans'] = data_3['photoclasskmeans'].apply(lambda x : i)
        elif i == 4:
            data_4 = data.loc[data['photoclasskmeans'] == i].groupby('photoclass').count()
            data_4 = data_4.reset_index()
            data_4['photoclasskmeans'] = data_4['photoclasskmeans'].apply(lambda x : i)
    photoclass_df = data.groupby('photoclass').count().reset_index()
    print(photoclass_df.shape)
    classes_df = {
    'classnumber' : [],
    'clusternumber': []
                }
    for c in list(photoclass_df['photoclass']):
        if c in list(data_0['photoclass']):
            classes_df['classnumber'].append(c)
            classes_df['clusternumber'].append(0)
        if c in list(data_1['photoclass']):
            classes_df['classnumber'].append(c)
            classes_df['clusternumber'].append(1)
        if c in list(data_2['photoclass']):
            classes_df['classnumber'].append(c)
            classes_df['clusternumber'].append(2)
        if c in list(data_3['photoclass']):
            classes_df['classnumber'].append(c)
            classes_df['clusternumber'].append(3)
        if c in list(data_4['photoclass']):
            classes_df['classnumber'].append(c)
            classes_df['clusternumber'].append(4)
    classes_df['classnumber']=np.array(classes_df['classnumber'])
    classes_df['clusternumber']=np.array(classes_df['clusternumber'])
    classes_df = pd.DataFrame(classes_df)
    l = classes_df.groupby('classnumber')
    l =  pd.DataFrame(l)
    p = []
    g_c = []
    for a, b in l.itertuples(index=False):
        p.append(b)
    for i in range(len(p)):
        g_c.append(list(p[i]['clusternumber']))
    grouped_classes_df = {
    'classnumber' : [],
    'clusternumber': []
                    }
    grouped_classes_df['classnumber']=photoclass_df['photoclass']
    g_c = np.array(g_c).reshape(len(g_c))
    grouped_classes_df['clusternumber'] = g_c

    grouped_classes_df = pd.DataFrame(grouped_classes_df)
    print(grouped_classes_df)#you can comment this line
    #path = os. getcwd()
    grouped_classes_df.to_csv(f'gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}/grouped_classes_df.csv',index=False)

if __name__ == "__main__":
    df_im,df_comments = get_data()
    store_data = store_data()
    classefier_model()
    culster_model(df_im)
    convert_df(df_comments)
    get_clust()
    img_0, img_1, img_2, img_3, img_4 = display_data()
    convert_df('df_images_0', img_0)
    convert_df('df_images_1', img_1)
    convert_df('df_images_2', img_2)
    convert_df('df_images_3', img_3)
    convert_df('df_images_4', img_4)
