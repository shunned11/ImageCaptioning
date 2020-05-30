from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import DataExtraction as de
import numpy as np

def create_tokens():    
    traincap,trainfeat=de.create_test_set()
    print(type(traincap))
    caplist=[]
    for id in traincap.keys():
        for cap in traincap[id]:
            caplist.append("<START>"+cap+"<END>")
    
    tk=Tokenizer()
    tk.fit_on_texts(caplist)
    return tk

def data_generator(tk,batch_size=5):
    traincap,trainfeat=de.create_test_set()
    X1=[]
    X2=[]
    Y=[]
    n=0
    vocab_size=len(tk.word_counts.keys())+1
    while 1:
        for id,caplist in traincap.items():
            n+=1
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
            if(n==batch_size):
                X1=np.array(X1)
                X2=np.array(X2)
                Y=np.array(Y)
                print(X1)
                print(X2.shape)
                print(Y.shape)
                yield [[X1,X2],Y]
                X1=[]
                X2=[]
                Y=[]
                n=0
                
tk=create_tokens()