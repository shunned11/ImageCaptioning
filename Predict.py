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
    plt.show()

    return feat


def greedy_search(feat,tk,model):
    reverse_word_map = dict(map(reversed, tk.word_index.items()))
    in_seq="startseq"
    print("Greedy Search")

    for i in range(34):
        seq=tk.texts_to_sequences([in_seq])[0]
        seq=pad_sequences([seq],maxlen=34)
        curseq=model.predict([feat,seq],verbose=0)
        word=np.argmax(curseq)
        word=reverse_word_map[word]
        if word is None:
            break
        in_seq=in_seq+" "+word
        if(word=="endseq"):
            print(in_seq)
            break

def beam_search(feat,tk,model,beam_width):
    print("Beam Search")
    reverse_word_map = dict(map(reversed, tk.word_index.items()))
    in_sequences=np.full((beam_width),"startseq",dtype=object)
    # prob=np.full((beam_width,1),1)
    finalpred=[]
    finalprob=[]
    prob=np.array([[1]],dtype=np.float64)
    for i in range(beam_width-1):
        prob=np.append(prob,[[0.0]],axis=0)
        
    for i in range(34):
        allseq=np.empty((beam_width,7579))
        j=0
        for in_seq in in_sequences:
            seq=tk.texts_to_sequences([in_seq])[0]
            seq=pad_sequences([seq],maxlen=34)
            curseq=model.predict([feat,seq],verbose=0)
            allseq[j][:]=curseq
            j+=1
        # temp=-(np.log(prob)+np.log(allseq))/7579
        temp=(prob*allseq)
        temp=temp.reshape(beam_width*7579,1)
        topprob=(np.argsort(temp,axis=0)[-beam_width:]).reshape((beam_width,))
        l=0
        temparr=np.full((beam_width),"",dtype=object)

        for k in topprob:
            prob[l]=temp[k]
            tempstr=""
            word=reverse_word_map[k%7579]
            tempstr=in_sequences[int(k/7579)]+' '+word
            temparr[l]=tempstr
            if(word=="endseq"):
                finalpred.append(temparr[l])
                finalprob.append(prob[l])
                prob[l]=0
            l+=1
 
        in_sequences=temparr
    print(finalpred)
    finalpred=finalpred[:10]
    finalprob=np.array(finalprob)
    finalprob=finalprob.reshape((finalprob.shape[0],))[:10]
    best=np.argsort(finalprob,axis=0)[-beam_width:].reshape((beam_width,1)).reshape((beam_width,))
    for i in best:
        print(finalpred[i])


doc="Data/Flickr_8k.trainImages.txt"
feat=extract_features("Example.jpg")
tk=ut.create_tokens(doc)
model=load_model("Model.h5")
beam_search(feat,tk,model,4)
print()
greedy_search(feat,tk,model)