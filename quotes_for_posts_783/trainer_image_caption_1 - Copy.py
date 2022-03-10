import numpy as np
from PIL import Image
import os
import string
from pickle import dump
from pickle import load
from keras.applications.xception import Xception #to get pre-trained model Xception
from keras.applications.xception import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.text import Tokenizer #for text tokenization
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.merge import add
from keras.models import Model, load_model
from keras.layers import Input, Dense#Keras to build our CNN and LSTM
from keras.layers import LSTM, Embedding, Dropout
from tqdm import tqdm_notebook as tqdm #to check loop progress
tqdm().pandas()
from google.cloud import storage
### GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'quotes_for_posts_783'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
# /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
# or if you want to use the full dataset (you need need to upload it first of course)


##### Training  - - - - - - - - - - - - - - - - - - - - - -

BUCKET_TRAIN_DATA_PATH = 'raw_data/image_dataset/image_dataset'
BUCKET_TRAIN_DATA_PATH_1= 'raw_data/image_dataset/df_images_0.csv'
glove_path = 'raw_data/image_dataset/glove.6B.300d.txt'
##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'image_description'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# not required here

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_data():
    """method to get the training data (or a portion of it) from google cloud bucket"""
    df_im_0 = pd.read_csv(f'gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH_1}')
    df_im_0 = df_im_0.head(10)
    print(df_im_0)
    download_blob(BUCKET_NAME, glove_path, 'temp.txt')
    glove = open('temp.txt', encoding="utf-8")
    return df_im_0, glove

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def load_fp(filename):
    # Open file to read
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text
# get all images with their captions
def img_capt(filename):
    file = load_doc(filename)
    captions = file.split('n')
    descriptions ={}
    for caption in captions[:-1]:
        img, caption = caption.split('t')
    if img[:-2] not in descriptions:
            descriptions[img[:-2]] = [ caption ]
    else:
            descriptions[img[:-2]].append(caption)
    return descriptions
#Data cleaning function will convert all upper case alphabets to lowercase, removing punctuations and words containing numbers
def txt_clean(captions):
    table = str.maketrans('','',string.punctuation)
    for img,caps in captions.items():
        for i,img_caption in enumerate(caps):
                img_caption.replace("-"," ")
                descp = img_caption.split()
                #uppercase to lowercase
                descp = [wrd.lower() for wrd in descp]
                #remove punctuation from each token
                descp = [wrd.translate(table) for wrd in descp]
                #remove hanging 's and a
                descp = [wrd for wrd in descp if(len(wrd)>1)]
                #remove words containing numbers with them
                descp = [wrd for wrd in descp if(wrd.isalpha())]
                #converting back to string
                img_caption = ' '.join(desc)
                captions[img][i]= img_caption
    return captions
def txt_vocab(descriptions):
    # To build vocab of all unique words
    vocab = set()
    for key in descriptions.keys():
        [vocab.update(d.split()) for d in descriptions[key]]
    return vocab
#To save all descriptions in one file
def save_descriptions(descriptions, filename):
   lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
                lines.append(key + 't' + desc )
    data = "n".join(lines)
    file = open(filename,"w")
    file.write(data)
    file.close()
# Set these path according to project folder in you system, like i create a folder with my name shikha inside D-drive
dataset_text = "D:shikhaProject - Image Caption GeneratorFlickr_8k_text"
dataset_images = "D:shikhaProject - Image Caption GeneratorFlicker8k_Dataset"
#to prepare our text data
filename = dataset_text + "/" + "Flickr8k.token.txt"
#loading the file that contains all data
#map them into descriptions dictionary 
descriptions = img_capt(filename)
print("Length of descriptions =" ,len(descriptions))
#cleaning the descriptions
clean_descriptions = txt_clean(descriptions)
#to build vocabulary
vocabulary = txt_vocab(clean_descriptions)
print("Length of vocabulary = ", len(vocabulary))
#saving all descriptions in one file
save_descriptions(clean_descriptions, "descriptions.txt")
 def preco_voc(df_im_0, glove):
    train = list(df_im_0['image_name'].map(str))
    descriptions = df_im_0.groupby('image_name')['comments'].apply(list).to_dict()
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

    embeddings_index = {} 
   
    for line in glove:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    embedding_dim = 300
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in wordtoix.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return max_length, vocab_size, embedding_matrix, train_descriptions, max_length, wordtoix

def encode(img,model_new):

    return fea_vec

def model_training(max_length, vocab_size):
    inputs0 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs0)
    fe2 = Dense(256, activation='relu')(fe1)
    
    inputs1 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 300, mask_zero=True)(inputs1)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs0, inputs1], outputs=outputs)
    model.summary()
    model.layers[2].set_weights([embedding_matrix])
    model.layers[2].trainable = False
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def LoadDataAndDoEssentials(df_im_0,model_new):
    images = list(df_im_0['image_name'])
    for i in images:
        imagePath = f'{BUCKET_TRAIN_DATA_PATH}/{i}.jpg'
        download_blob(BUCKET_NAME, imagePath, 'temp.jpg')
        img = image.load_img('temp.jpg', target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        fea_vec = model_new.predict(img) 
        fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
        img = img.reshape((1,2048))
 
    return img

def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    print(type(descriptions))
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            key = int(key)
            n+=1
            # retrieve the photo feature
            #photo = photos
            for i in range(len(photos)):
                X1.append(f"{photos[0][i]}.jpg")

            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X2.append(in_seq)
                    y.append(out_seq)

            if n==num_photos_per_batch:
                yield ([array(X1), array(X2)], array(y))
                X1, X2, y = list(), list(), list()
                n=0
    return np.array(X_1), np.array(X_2), np.array(y)

STORAGE_LOCATION = 'quotes_for_posts_783/images_caption_model_1.joblib'

def upload_model_to_gcp_1():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STORAGE_LOCATION)
    blob.upload_from_filename('images_caption_model_1.joblib')

def save_model(model):
    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    joblib.dump(model, 'model.joblib')
    print("saved model.joblib locally")

    # Implement here
    upload_model_to_gcp()
    print(f"uploaded images_caption_model_1 to gcp cloud storage under \n => {STORAGE_LOCATION}")


if __name__ == '__main__':
    # get training data from GCP bucket
    df_im_0, glove = get_data()
    model = InceptionV3(weights='imagenet')
    model_new = Model(model.input, model.layers[-2].output)
    max_length, vocab_size, embedding_matrix, train_descriptions, max_length, wordtoix = preco_voc(df_im_0, glove)
    # preprocess data

    epochs = 10
    batch_size = 3
    steps = len(train_descriptions)//batch_size
    model = model_training(max_length, vocab_size)
    
    # train model (locally if this file was called through the run_locally command
    # or on GCP if it was called through the gcp_submit_training, in which case
    # this package is uploaded to GCP before being executed)
    train_features = LoadDataAndDoEssentials(df_im_0,model_new)
    generator = data_generator(train_descriptions, train_features, wordtoix, max_length, batch_size)
    model.fit(generator, epochs=epochs, steps_per_epoch=steps, verbose=1)

    # save trained model to GCP bucket (whether the training occured locally or on GCP)
    save_model(model)
