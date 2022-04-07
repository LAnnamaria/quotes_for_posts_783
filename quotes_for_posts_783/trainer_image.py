import nltk
#nltk.download('all')
#nltk.download('wordnet')
#nltk.download('stopwords')

import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from pickle import load
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
from keras.models import Model
from keras.layers import Input , Dense , LSTM , Embedding , Dropout
from keras.layers.merge import add
from keras.callbacks import EarlyStopping
from keras.models import load_model
from google.cloud import storage
### GCP configuration - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'quotes_for_posts_783'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

BUCKET_TRAIN_DATA_PATH = 'raw_data/image_dataset/image_dataset'
BUCKET_TRAIN_DATA_PATH_1= 'raw_data/image_dataset/df_images_0.csv'

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'image_description'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def extract_features(df_im_0):
    
    model = VGG16()
    #remove last layer
    model.layers.pop()
    model = Model(inputs = model.inputs , outputs = model.layers[-1].output)
    print(model.summary())
    features = dict()
    images = df_im_0['image_name']
    for i in images:
        imagePath =f'{BUCKET_TRAIN_DATA_PATH}/{i}'
        download_blob(BUCKET_NAME, imagePath, 'temp.jpg')
        image = load_img('temp.jpg' , target_size=(224 , 224))
        image = img_to_array(image)
        image = image.reshape((1 , image.shape[0] , image.shape[1] ,image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image , verbose = 0)
        # get image id
        image_id = i.split(".")[0]
        # store features
        features[image_id] = feature
        print(i)
    return features

def image_set(df_im_0):
    df_im_0['image_name'] = df_im_0['image_name'].astype('str') + '.jpg'
    features = extract_features(df_im_0)
    print('extracted features :',len(features))
    dump(features , open('features.pkl' , 'wb'))

def load_decriptions(doc):
    mapping = dict()
    
    for i in range(len(doc)):
        image_id = doc['image_name'][i]
        image_desc = doc['comments'][i]
        
        if image_id not in mapping:
            mapping[image_id] = list()
        
        mapping[image_id].append(image_desc)
        
    return mapping

def clean_text(desc):
    
    lemma = WordNetLemmatizer()
    # clean punctuation
    desc = re.sub(r'[^\w\s]' ,'', desc)
    # tokenize the words
    desc = desc.split()
    # convert to lower case
    desc = [token.lower() for token in desc]
    # lemmatization
    desc = [lemma.lemmatize(token) for token in desc]
    # remove numerical values
    desc = [token for token in desc if token.isalpha()]
    # join whole token
    desc = ' '.join(desc)
    return desc

# convert loaded descriptions into vocablury
def to_vocabluary(descriptions):
    all_desc = set()
    
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
        
    return all_desc

def save_descriptions(descriptions , filename):
    lines = list()
    
    for key , desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key +' '+ desc)
            
    data = '\n'.join(lines)
    file = open(filename , 'w')
    file.write(data)
    file.close()

def load_doc(filename):
    file = open(filename , 'r')
    text = file.read()
    file.close()
    return text

def load_clean_descriptions(filename , dataset):
    doc = load_doc(filename)
    descriptions = dict()
    
    for line in doc.split('\n'):
        tokens = line.split()
        image_name , image_desc = tokens[0] , tokens[1:]
        
        if image_name in dataset:
            
            if image_name not in descriptions:
                descriptions[image_name] = list()
            
            # we add two tage at start and at end of the descitpion to identify to start and 
            # end of desc.
            desc = 'startseq '+ ' '.join(image_desc)+ ' endseq'
            descriptions[image_name].append(desc)
            
    return descriptions

# laod photo features
def load_photo_features(filename , dataset):
    all_features = load(open(filename,'rb'))
    features = {k+'.jpg' : all_features[k] for k in dataset}
    return features

def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
        
    return all_desc

def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# calculate the length with most words
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max([len(line.split())for line in lines])
    
# create sequences of images,input sequences and output sequences
def create_sequences(tokenizer , max_length , desc_list , photo):
    X1 , X2 , y = list() , list() , list()
    for desc in desc_list:
        # convert words to number value
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
 
            in_seq , output_seq = seq[:i] , seq[i]
            in_seq = pad_sequences([in_seq] , maxlen = max_length)[0]
            output_seq = to_categorical([output_seq] , num_classes = vocab_size)[0]
            
            X1.append(photo)
            X2.append(in_seq)
            y.append(output_seq)
            
    return np.array(X1) , np.array(X2) , np.array(y)

def define_Model(vocab_size , max_length):
    
    # feature extractor model
    inputs1 = Input(shape=(1000, ))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(512 , activation='relu')(fe1)
    fe3 = Dense(256 , activation = 'relu')(fe2)
    
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size,512,mask_zero=True )(inputs2) # mask_zero = ignore padding
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(512 , return_sequences=True)(se2)
    se4 = Dropout(0.5)(se3)
    se5 = LSTM(256)(se4)
    
    
    #decoder Model
    decoder1 = add([fe3 , se5])
    decoder2 = Dense(256 , activation='relu')(decoder1)
    decoder3 = Dense(512 , activation='relu')(decoder2)
    outputs = Dense(vocab_size , activation='softmax')(decoder3)
    
    # combine both image and text
    model = Model([inputs1 , inputs2] , outputs)
    model.compile(loss='categorical_crossentropy' , optimizer = 'adam')
    
    # summary
    print(model.summary())
    
    return model

def data_generator(descriptions , photos , tokenizer , max_length):
    while 1:
        for key , desc_list in descriptions.items():
            photo = photos[key][0]
            in_img , in_seq , out_seq = create_sequences(tokenizer , max_length , desc_list , photo)
            
            yield[[in_img , in_seq] , out_seq]

def word_for_id(integer , tokenizer):
    for word,index in tokenizer.word_index.items():
        if index == integer:
            return word

def generate_desc(model , tokenizer , photo , max_length):
    
    input_text = 'startseq'
    
    for i in range(max_length):
        
        sequence = tokenizer.texts_to_sequences([input_text])[0]
        sequence = pad_sequences([sequence] , maxlen=max_length)
        
        # predict the next word
        next_word_id = model.predict([photo,sequence],verbose = 0)
        
        # get highest probality word from list of words
        next_word_id = np.argmax(next_word_id)
        
        # get word from id
        word = word_for_id(next_word_id , tokenizer)
        
        if word is None:
            break
            
        # update input text
        input_text += ' '+ word
        
        if word == 'endseq':
            break
            
    return input_text

if __name__ == '__main__':
    df_im_0 = pd.read_csv(f'gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH_1}')
    df_im_0 = df_im_0.head(120)
    image_set(df_im_0)
    df_im_0['comments'] = df_im_0['comments'].apply(lambda x : clean_text(str(x)))
    desc_map = load_decriptions(df_im_0)
    vocabulary = to_vocabluary(desc_map)
    save_descriptions(desc_map , 'descriptions.txt')
    train = set(df_im_0['image_name'])
    print('len of train image',len(train))
    train_descriptions = load_clean_descriptions('descriptions.txt' , train)
    train = pd.DataFrame(train)
    train2 = train[0].apply(lambda x : x.replace('.jpg' , ''))
    train_features = load_photo_features('features.pkl' , train2)
    print('photos train :',len(train_features))
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    print('vocab size' , vocab_size)
    max_len = max_length(train_descriptions)
    model = define_Model(vocab_size , max_len)
    epochs = 5
    steps = len(train_descriptions)

    for i in range(epochs):
        generator = data_generator(train_descriptions , train_features , tokenizer , max_len)
        
        model.fit(generator , epochs = 1 , steps_per_epoch = steps , verbose = 1)
        
        model.save('model_'+ str(i+1) + '.h5')