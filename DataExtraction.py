import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
import tensorflow.keras.backend as K
import pickle5 as pickle
import os
import string

def text_data():
    captions=dict()
    doc=open("Data/Flickr8k.token.txt",'r')
    for line in doc:
        token=line.split()
        id=token[0]
        id=id.split('.')[0]
        caplist=token[1:]
        capstring=' '.join(caplist)
        if id not in captions:
            captions[id]=list()

        captions[id].append(capstring) 
    return captions

def clean_text_data():
    table = str.maketrans('', '', string.punctuation)
    captions=text_data()
    for cap in captions.values():
        for i in range(len(cap)):
            caplist=cap[i]
            caplist=caplist.split()
            caplist=[w.lower() for w in caplist]
            caplist=[w.translate(table) for w in caplist]
            caplist=[w for w in caplist if len(w)>1]
            caplist=[w for w in caplist if w.isalpha()]
            cap[i]=" ".join(caplist)            
    return captions

# clean_text_data()

def image_data():
    img_feat=dict()
    for name in os.listdir("Flicker8k_Dataset"):
        model=VGG19(weights='imagenet',include_top=False)
        print(model.summary())
        image=load_img(os.path.join("Flicker8k_Dataset",name),target_size=(256,256))
        image=img_to_array(image)
        image=np.expand_dims(image,axis=0)
        image=preprocess_input(image)
        feat=model.predict(image)
    
        id=name.split('.')[0]
        img_feat[id]=feat
        print(id,end=" ")
        K.clear_session()
    pickle.dump(img_feat,open('features.pkl','wb'))

# feat=np.load('img_features.npy',allow_pickle=True).item()
# print(feat['1000268201_693b08cb0e'])



def create_test_set():
    #Separate out the train set based on the doc file provided
    doc=open("Data/Flickr_8k.trainImages.txt")
    feat=np.load('img_features.npy',allow_pickle=True).item()
    trainid=[]
    for line in doc:
        line=line.split(".")
        trainid.append(line[0])
    captions=clean_text_data()
    traincaptions=dict()
    trainfeat=dict()
    for id in captions.keys():
        if(id in trainid):
            traincaptions[id]=captions[id]
            trainfeat[id]=feat[id]

    return traincaptions,trainfeat


