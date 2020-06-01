from tensorflow.keras.applications.xception import Xception,preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import Utilities as ut

def extract_features(path):
    img=load_img(path,target_size=(299,299))
    img=img_to_array(img)
    plt.imshow(img/255.)
    # img=mpimg.imread(path)
    image=np.expand_dims(img,axis=0)
    image=preprocess_input(image)
    model=Xception(weights='imagenet')
    model=Model(inputs=model.inputs,outputs=model.layers[-2].output)
    feat=model.predict(image)
    # plt.show()

    return feat


def greedy_search(feat,tk,model):
    reverse_word_map = dict(map(reversed, tk.word_index.items()))
    in_seq="start"
    for i in range(34):
        seq=tk.texts_to_sequences([in_seq])[0]
        seq=pad_sequences([seq],maxlen=34)
        curseq=model.predict([feat,seq],verbose=0)
        print(curseq[0])
        word=np.argmax(curseq)
        word=reverse_word_map[word]
        print(word,end=" ")
        in_seq=in_seq+" "+word
        if(word=="end"):
            print(i)
            break

def beam_search():
    pass
doc="Data/Flickr_8k.trainImages.txt"
feat=extract_features("Example.jpg")
tk=ut.create_tokens(doc)
model=load_model("Model.h5")
greedy_search(feat,tk,model)