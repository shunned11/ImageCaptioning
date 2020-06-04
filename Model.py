import numpy as np
import pandas as pd
import tensorflow as tf
import Utilities as ut
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model,model_from_json
from tensorflow.keras.layers import Input,Embedding, LSTM, Dense, Dropout,Activation,Bidirectional,Add
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


if __name__ == "__main__":
    doc1="Data/Flickr_8k.trainImages.txt"
    doc2="Data/Flickr_8k.devImages.txt"
    tk=ut.create_tokens(doc1)
    vocabulary_size = len(tk.word_counts.keys())+1
    emb_matrix=ut.embedding(vocabulary_size,tk)

    TX1=np.load("ExtractedData/Train_X1.npy")
    TX2=np.load("ExtractedData/Train_X2.npy")
    TY=np.load("ExtractedData/Train_Y.npy")

    DX1=np.load("ExtractedData/Dev_X1.npy")
    DX2=np.load("ExtractedData/Dev_X2.npy")
    DY=np.load("ExtractedData/Dev_Y.npy")

    print(DX1.shape)
    print(TX1.shape)

    train_steps=120
    validation_steps=20

    callbackslist=[TensorBoard(log_dir='logs'),ModelCheckpoint(filepath='Model.h5',monitor='val_loss',save_best_only=True)]


    input1=Input(shape=(2048,))
    imodel1=Dropout(0.5)(input1)
    imodel2=Dense(512,activation='relu')(imodel1)

    input2=Input(shape=(34,))
    tmodel1=Embedding(vocabulary_size,50,mask_zero=True,trainable=False)(input2)
    tmodel2=Dropout(0.4)(tmodel1)
    tmodel3=Bidirectional(LSTM(256,return_sequences=True))(tmodel2)
    tmodel4=Dropout(0.4)(tmodel3)
    tmodel5=Bidirectional(LSTM(256,return_sequences=False))(tmodel4)

    decoder1=Add()([imodel2,tmodel5])
    decoder2=Dense(256,activation='relu')(decoder1)
    outputs=Dense(vocabulary_size,activation='softmax')(decoder2)

    model=Model(inputs=[input1,input2],outputs=outputs)
    model.summary()
    model.layers[1].set_weights([emb_matrix])
    model.compile(loss='binary_crossentropy', optimizer='adam')

    model.fit([TX2,TX1],TY,epochs=10,callbacks=callbackslist,batch_size=64,validation_data=([DX2,DX1],DY))

# making a directory models to save our models



