from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import DataExtraction as de
import numpy as np

def create_tokens(doc):    
    traincap,trainfeat=de.create_train_set(doc)
    caplist=[]
    for id in traincap.keys():
        for cap in traincap[id]:
            caplist.append(cap)
    tk=Tokenizer()
    tk.fit_on_texts(caplist)
    return tk

def data_generator(tk,doc,name):
    traincap,trainfeat=de.create_train_set(doc)
    n=0
    vocab_size=len(tk.word_counts.keys())+1
    X1,X2,Y=[],[],[]
    for id,caplist in traincap.items():
        n+=1
        print(n)
        for cap in caplist:
            seq=tk.texts_to_sequences([cap])[0]
            for i in range(1,len(seq)):
                inseq=seq[:i]
                inseq=pad_sequences([inseq],maxlen=34)[0]
                outseq=seq[i]
                outseq=to_categorical([outseq],num_classes=vocab_size)[0]
                X1.append(inseq)
                X2.append(trainfeat[id][0])
                Y.append(outseq)
        if(n==len(traincap.keys())):
            print("Done")
            np.save("{}_{}.npy".format(name,"X1"),np.array(X1))
            del X1
            np.save("{}_{}.npy".format(name,"X2"),np.array(X2))
            del X2
            np.save("{}_{}.npy".format(name,"Y"),np.array(Y))
            del Y
            return

if __name__ == "__main__":
    doc1="Data/Flickr_8k.trainImages.txt"
    doc2="Data/Flickr_8k.devImages.txt"
    tk=create_tokens(doc1)
    data_generator(tk,doc1,"Train")
    data_generator(tk,doc2,"Dev")


def glovevec():
    with open("glove.6B.50d.txt",'r',encoding='utf8') as f:
        words=set()
        word_to_vec_map={}
        for line in f:
            line=line.strip().split()
            curr_word=line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word]=np.array(line[1:],dtype=np.float64)
        
    return word_to_vec_map

def embedding(vocabulary_size,tk):
    emb_matrix = np.zeros((vocabulary_size, 50))

    word_to_vec_map=glovevec()
    for w, i in tk.word_index.items():
        temp= word_to_vec_map.get(w)
        if temp is not None:
            emb_matrix[i, :]=temp
    return emb_matrix

