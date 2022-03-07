import numpy as np  
import pandas as pd  
import os
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import Flatten, Dense, LSTM, Dropout, Embedding, Activation
from keras.layers import concatenate, BatchNormalization, Input
from keras.layers.merge import add
from keras.utils import to_categorical, plot_model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import matplotlib.pyplot as plt 
import cv2

### GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'image_caption'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
# /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
# or if you want to use the full dataset (you need need to upload it first of course)
BUCKET_TRAIN_DATA_PATH = 'raw_data/'

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'image_description'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# not required here

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -



#Hyperparameters
embed_size = 300
hidden_size = 512
enc_dim = 2048
att_dim = 49
batch_size = 80
train_embed = False
train_CNN = False
lr_enc = 1e-5
lr_dec = 5e-4
num_layers = 1
num_epochs = 10
freq_threshold = 5


train_imgs = list()
train_labels = list()
val_imgs = list()
val_labels = list()
test_imgs = list()
test_labels = list()
all_labels = list()

for img in dataset['images']:   
    for sentence in img['sentences']:
        
        if img['split'] == 'train':
            
                train_imgs.append(img['filename'])
                train_labels.append(sentence['tokens'])
                all_labels.append(sentence['tokens'])

                
        if img['split'] == 'test':
            
                test_imgs.append(img['filename'])
                test_labels.append(sentence['tokens'])
                all_labels.append(sentence['tokens'])
    
            
        if img['split'] == 'val':
                 
                val_imgs.append(img['filename'])
                val_labels.append(sentence['tokens'])
                all_labels.append(sentence['tokens'])

dict_tokens = dict()
imgs_idx = dict()
idx_imgs = dict()

for idx, img in enumerate(dataset['images']):
    dict_tokens[idx] = [sentence['tokens'] for sentence in img['sentences']]   
    imgs_idx[img['filename']] = idx
    idx_imgs[idx] = img['filename']

class Vocabulary:
    
    def __init__(self, freq_threshold):
        self.itos = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
    
        print('Start building vocabulary!')

        for sentence in sentence_list: 

            for word in sentence:

                if word not in frequencies:
                    frequencies[word]=1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:

                    self.stoi[word]  = idx
                    self.itos[idx] = word
                    idx += 1

        print('Vocabulary built!')
  
    def numericalize(self, tokenized_text):

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

class FlickrDataset(Dataset):
    def __init__(self, root_dir, imgs, labels, vocab, transforms = None):
        self.root_dir = root_dir
        self.transforms = transforms

        self.imgs = imgs
        self.captions = labels

        self.vocab = vocab

    def __len__(self):
        return (len(self.imgs))

    def __getitem__(self, index):
        
        caption = self.captions[index]
        img_id = self.imgs[index]

        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return (img, torch.tensor(numericalized_caption), torch.tensor([imgs_idx[img_id]]))

class MyCollate:
    
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim = 0)
        img_ids = [item[2].unsqueeze(0) for item in batch]
        img_ids = torch.cat(img_ids, dim = 0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first = False, padding_value = self.pad_idx)

        return imgs, targets, img_ids
  

def get_loader(root_folder, 
               imgs, 
               labels,
               vocab,
               transforms, 
               batch_size = 32, 
               shuffle=True):
    dataset = FlickrDataset(root_folder, imgs, labels, vocab, transforms)
    print('Dataset made!')
    loader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        collate_fn = MyCollate(pad_idx = dataset.vocab.stoi["<PAD>"])
     )

    return loader, dataset

vocab = Vocabulary(freq_threshold)
vocab.build_vocabulary(all_labels)
vocab_size = len(vocab)

words = []
glove = {}
with open(f'../raw_data/glove.6B.300d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word) 
        vect = np.array(line[1:]).astype(np.float)
        glove[word] = vect

weights_matrix = np.zeros((vocab_size, embed_size))

for key, index in vocab.stoi.items():
    
    if key in glove:
        weights_matrix[index] = glove[key]
    
    else:
        weights_matrix[index] = np.random.normal(scale=0.6, size=(embed_size, ))


class EncoderCNN(nn.Module):
    def __init__(self, enc_dim, embed_size, hidden_size):
        super(EncoderCNN, self).__init__()
        resnet = torchvision.models.resnet101(pretrained = True)
        all_modules = list(resnet.children())
        modules = all_modules[:-2]
        self.resnet = nn.Sequential(*modules) 
        self.avgpool = nn.AvgPool2d(7)
        self.V_affine = nn.Linear(enc_dim, hidden_size)
        self.vg_affine = nn.Linear(enc_dim, embed_size)
        self.fine_tune()
       
    def forward(self, images):
        encoded_image = self.resnet(images)
        batch_size = encoded_image.shape[0]
        features = encoded_image.shape[1]
        num_pixels = encoded_image.shape[2] * encoded_image.shape[3]
        global_features = self.avgpool(encoded_image).view(batch_size, -1)
        global_features = F.relu(self.vg_affine(global_features))
        enc_image = encoded_image.permute(0, 2, 3, 1)  
        enc_image = enc_image.view(batch_size,num_pixels,features)
        enc_image = F.relu(self.V_affine(enc_image))
  
        return enc_image, global_features

    def fine_tune(self, train_CNN = False):
        
        if not train_CNN:
            for param in self.resnet.parameters():
                param.requires_grad = False
                        
        else:
            for module in list(self.resnet.children())[len(list(self.resnet.children()))-1:]:    #1 layer only. len(list(resnet.children())) = 8
                for param in module.parameters():
                    param.requires_grad = True 



class AdaptiveLSTMCell(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(AdaptiveLSTMCell, self).__init__()
        self.lstm_cell = nn.LSTMCell(embed_size, hidden_size)
        self.x_affine = nn.Linear(embed_size, hidden_size)
        self.h_affine = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, inp, states):
        h_prev, m_prev = states
        h_t, m_t = self.lstm_cell(inp, (h_prev, m_prev))
        g_t = torch.sigmoid(self.x_affine(inp) + self.h_affine(h_prev))
        s_t = g_t*(torch.tanh(m_t))
        
        return s_t, h_t, m_t

class AdaptiveAttention(nn.Module):
    def __init__(self, hidden_size, att_dim):
        super(AdaptiveAttention, self).__init__()
        self.v_affine = nn.Linear(hidden_size, att_dim)
        self.h_affine = nn.Linear(hidden_size, hidden_size)
        self.h_att = nn.Linear(hidden_size, att_dim)
        self.s_affine = nn.Linear(hidden_size, hidden_size)
        self.s_att = nn.Linear(hidden_size, att_dim)
        self.wh_affine = nn.Linear(att_dim, 1)
        self.c_hidden = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, V, h_t, s_t):
        num_pixels = V.shape[1]
        v_att = self.v_affine(V)
        h_affine = F.relu(self.h_affine(h_t))
        h_att = self.h_att(h_affine)
        h_att_r = h_att.unsqueeze(1).expand(h_att.size(0), num_pixels, h_att.size(1)) 
        
        z_t = self.wh_affine(torch.tanh(v_att + h_att_r)).squeeze(2)
        alpha_t = F.softmax(z_t, dim = 1)
        c_t = torch.sum((alpha_t.unsqueeze(2).expand(V.size(0), num_pixels, V.size(2)))*V, dim = 1)
        
        s_affine = F.relu(self.s_affine(s_t))
        s_att = self.s_att(s_affine)
        z_hat_t = torch.cat((z_t, self.wh_affine(torch.tanh(s_att + h_att))), dim = 1)
        alpha_hat_t = F.softmax(z_hat_t, dim = 1)
        
        beta_t = alpha_hat_t[:,-1].view(alpha_hat_t.size(0), -1)
        
        c_hat_t = beta_t*s_t + (1-beta_t)*c_t
        
        out_l = c_hat_t + h_affine
        
        return out_l

class DecoderLSTMAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, att_dim, vocab_size, train_embed = False):
        super(DecoderLSTMAttention, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(p=0.5)
        
        if not train_embed:  
            self.embed.load_state_dict({'weight': torch.tensor(weights_matrix)})
            
        self.vocab_size = vocab_size
        self.adaptive_lstm = AdaptiveLSTMCell(embed_size*2, hidden_size)
        self.adaptive_att = AdaptiveAttention(hidden_size, att_dim)
        self.p_affine = nn.Linear(hidden_size, vocab_size)
        self.fine_tune(train_embed)
    
    def init_hidden_states(self, batch_size, hidden_size):
        h_0 = torch.zeros((batch_size, hidden_size))
        m_0 = torch.zeros((batch_size, hidden_size))
        
        return h_0, m_0
        
    def forward(self, V, vg, captions, cap_len, device = 'cpu'):
        
        inp = self.embed(captions)
        
        h_0, m_0 = self.init_hidden_states(V.size(0), V.size(2))
        
        states = h_0.to(device), m_0.to(device)
        
        predictions = torch.zeros((inp.size(0), inp.size(1), self.vocab_size)).to(device)
        
        for timestep in range(cap_len):
            
            x_t = torch.cat((vg, inp[timestep]), dim = 1)
            
            s_t, h_t, m_t = self.adaptive_lstm(x_t, states)
            
            states = (h_t, m_t)
            
            out_l = self.adaptive_att(V, h_t, s_t)
            
            output = self.p_affine(self.dropout(out_l))
            
            predictions[timestep, :, :] = output
            
        return predictions
    
    def fine_tune(self, train_embed = False):
        
        self.embed.weight.requires_grad = train_embed

def pred(inp, V, vg, states, decoder):
    
    inp_c = torch.cat((vg, inp), dim = 1)
    
    h_t, m_t = states
    
    s_t, h_t, m_t = decoder.adaptive_lstm(inp_c, states)
            
    states = (h_t, m_t)
            
    out_l = decoder.adaptive_att(V, h_t, s_t)
            
    output = decoder.p_affine(decoder.dropout(out_l))
    
    output = F.softmax(output, dim = 1)
    
    return output.view(output.size(1)).detach().cpu().numpy(), states
    
def caption_image_beam(image, vocabulary, encoder, decoder, device = 'cpu', k = 10, max_length=50):
    result_caption = []

    with torch.no_grad():
        V, vg = encoder(image)
        states = (torch.zeros((1, V.size(2))).to(device), torch.zeros((1, V.size(2))).to(device))
        sequences = [[list(), 0.0, states]]
        inp = vocabulary.stoi['<SOS>']

        for _ in range(max_length):
            
            all_candidates = list()
            
            for i in range(len(sequences)):
                seq, score, states = sequences[i]
                
                if len(seq) != 0:
                    inp = seq[-1]
                    
                    if vocabulary.itos[inp] == "<EOS>":
                        all_candidates.append(sequences[i])
                        continue
                        
                inp = decoder.embed(torch.tensor([inp]).to(device))
                    
                predictions, states = pred(inp, V, vg, states, decoder)
                
                word_preds = np.argsort(predictions)[-k:]
                
                for j in word_preds:
                    candidate = (seq + [j], score - math.log(predictions[j]), states)
                    all_candidates.append(candidate)
                    
            ordered = sorted(all_candidates, key=lambda tup:tup[1]/(len(tup[0])))
            sequences = ordered[:k]     
            
    output_arr = sequences[0][0]
            
    if vocabulary.itos[sequences[0][0][-1]] == '<EOS>':
        output_arr = sequences[0][0][:-1]
        
    if vocabulary.itos[sequences[0][0][0]] == '<SOS>':
        output_arr = output_arr[1:]    
        
    return [vocabulary.itos[idx] for idx in output_arr]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss(ignore_index = vocab.stoi["<PAD>"])
enc = EncoderCNN(enc_dim, embed_size, hidden_size).to(device)
dec = DecoderLSTMAttention(embed_size, hidden_size, att_dim, vocab_size, train_embed).to(device)
optim_enc = optim.Adam(enc.parameters(), lr = lr_enc, betas = (0.8, 0.999))
optim_dec = optim.Adam(dec.parameters(), lr = lr_dec, betas = (0.8, 0.999))


val_iter = iter(val_loader)
train_losses = list()
val_losses = list()
cumulative_bleu_scores = list()
cider_scores = list()
meteor_scores = list()
rougel_scores = list()
bleu1_scores = list()
bleu2_scores = list()
bleu3_scores = list()
bleu4_scores = list()

for epoch in range(num_epochs):
    
    if (epoch+1) >= 21 and train_CNN == False:
        train_CNN = True
        enc.fine_tune(True)
        
    for batch_idx, (imgs, captions, img_ids) in enumerate(train_loader):
        
        enc.train()
        dec.train()
                
        imgs = imgs.to(device)
        captions = captions.to(device)

        enc_imgs, global_features = enc(imgs)
        outputs = dec(enc_imgs, global_features, captions[:-1], captions[:-1].size(0), device)

        loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions[1:].reshape(-1))
        
                
        optim_enc.zero_grad()
        optim_dec.zero_grad()
        loss.backward()
                
        optim_enc.step()             
        optim_dec.step()

        if batch_idx%600 == 0:
            
            with torch.no_grad():
                
                enc.eval()
                dec.eval()
                
                try:
                    val_imgs, val_captions, val_img_ids = next(val_iter)
                
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_imgs, val_captions, val_img_ids = next(val_iter)
                
                val_imgs = val_imgs.to(device)
                val_captions = val_captions.to(device)

                enc_val_imgs, global_val_features = enc(val_imgs)
                val_outputs = dec(enc_val_imgs, global_val_features, val_captions[:-1], val_captions[:-1].size(0), device)

                val_loss = criterion(val_outputs.reshape(-1, val_outputs.shape[2]), val_captions[1:].reshape(-1))
                
                val_losses.append(val_loss.item())
                train_losses.append(loss.item())
                
                val_img_ids = val_img_ids.squeeze(1).numpy()
                
                r_set = np.arange(batch_size)
                
                np.random.shuffle(r_set)
                
                index = r_set[0]
                
                val_img = val_imgs[index]
                candidate = caption_image_beam(val_img.unsqueeze(0), vocab, enc, dec, device)
                ref_tokens = dict_tokens[val_img_ids[index]]
                cumulative_bleu_score = sentence_bleu(ref_tokens, candidate, weights=(0.25, 0.25, 0.25, 0.25))
                    
                hyp = ' '.join(candidate)
                refs = list()
                    
                for sentence in ref_tokens:
                    ref = ' '.join(sentence)
                    refs.append(ref)
                    
                metrics = compute_individual_metrics(refs, hyp)
                
                np.random.shuffle(refs)
                
               # bleu1_scores.append(metrics['Bleu_1'])
               # bleu2_scores.append(metrics['Bleu_2'])
               # bleu3_scores.append(metrics['Bleu_3'])
               # bleu4_scores.append(metrics['Bleu_4'])
               # cider_scores.append(metrics['CIDEr'])
               # cumulative_bleu_scores.append(cumulative_bleu_score)
               # rougel_scores.append(metrics['ROUGE_L'])
               # meteor_scores.append(metrics['METEOR'])
                
                pil_img = Image.open(os.path.join('../raw_data/flicker30k_images', idx_imgs[val_img_ids[index]])).convert("RGB")
                plt.imshow(np.asarray(pil_img))
                print(f"Predicted Caption: {hyp}\n\n")
                #print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Training loss: {loss.item()} Validation loss: {val_loss.item()} Cumulative BLEU Score: {cumulative_bleu_score} CIDEr Score: {metrics['CIDEr']} METEOR Score: {metrics['METEOR']} ROUGE_L: {metrics['ROUGE_L']} \n")
            


STORAGE_LOCATION = '/model.joblib'


def upload_model_to_gcp():


    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(STORAGE_LOCATION)

    blob.upload_from_filename('model.joblib')


def save_model(reg):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""

    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    joblib.dump(reg, 'model.joblib')
    print("saved model.joblib locally")

    # Implement here
    upload_model_to_gcp()
    print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")


if __name__ == '__main__':
    # get training data from GCP bucket
    df = get_data()

    # preprocess data
    filenames = [img for img in glob.glob("../raw_data/flickr30k_images/*.jpg")]

    filenames_small = filenames[0:1000]

    images = []
    for img in filenames_small:
        print(img)
        n= mpimg.imread(img)
        images.append(n)

    f = open('../raw_data/dataset_flickr30k.json')
    dataset = json.load(f)

    transformations = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    # train model (locally if this file was called through the run_locally command
    # or on GCP if it was called through the gcp_submit_training, in which case
    # this package is uploaded to GCP before being executed)
    train_loader , train_dataset = get_loader(
        root_folder='../raw_data/flicker30k_images',
        imgs=train_imgs,
        labels=train_labels,
        vocab=vocab,
        transforms=transformations,
        batch_size=batch_size
    )
    val_loader , val_dataset = get_loader(
            root_folder='../raw_data/flicker30k_images',
            imgs=val_imgs,
            labels=val_labels,
            vocab=vocab,
            transforms=transformations,
            batch_size=batch_size
        )

    # save trained model to GCP bucket (whether the training occured locally or on GCP)
    save_model(reg)
