




from collections import defaultdict
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
import math
import pandas as pd
from functools import reduce
from os import listdir
from os.path import isfile, join


# Creating a function to generate N-Grams
import pandas as pd


def generate_ngrams(text, WordsToCombine):
    words = text.split()
    output = []
    for i in range(len(words) - WordsToCombine + 1):
        output.append(words[i:i + WordsToCombine])
    return output

def generate_gram_freq(text,n):
    n_grams = generate_ngrams(text=text, WordsToCombine=n)
    nfreq = defaultdict(int)
    for word in n_grams:
        nfreq[' '.join(word)] += 1
    return(nfreq)

def cos_sim(u,v):
    return(np.dot(u,v)/(npla.norm(u)*npla.norm(v)))

########

DOCS_DIR="BOOK1_DOCS/"



onlyfiles = [f for f in listdir(DOCS_DIR) if isfile(join(DOCS_DIR, f))]
onlyfiles.sort()
ys=['CLEANED' in i for i in onlyfiles]
cleaned_files=np.array(onlyfiles)[np.array(ys)]

dfs=[]
for docname in cleaned_files :
    doc_input=DOCS_DIR+docname
    name_column=docname[:-12]

    with open(doc_input, 'r') as infile:
        doc = infile.readlines()

    tdoc = generate_gram_freq(doc[0], 1)
    df_doc = pd.DataFrame(tdoc.items(), columns=['word', name_column])

    dfs.append(df_doc)

d = reduce(lambda left, right: pd.merge(left, right, on='word', how='outer'), dfs)
d.fillna(0, inplace=True)


tf=d.copy()
dummy=d.copy()

# TF NORMALIZATION:

np.seterr(divide='ignore')
for i in tf.columns[1:len(tf.columns)]:
    col_name=i
    # average word frequency per document:
    ad=tf[col_name].mean()

    #Normalizing function:
    #turning off divide by zero warning

    tf[col_name]=np.where(tf[col_name]>0,(1+np.log(tf[col_name]))/(1+np.log(ad)),0)
    #warning back on
    dummy[col_name]=np.where(dummy[col_name]>0,1,0)
np.seterr(divide = 'warn')


#IDF: INVERSE DOCUMENT FREQUENCY (WE WANT TO REDUCE THE IMPACT OF COMMOM WORDS THAT APPEAR IN OTHER DOCUMENTS)


# TF-IDF : we don't want frequent words to overwhelm the frequency of others!

#NUMBER OF DOCUMENTS:
N_DOCS=len(d.columns)-1

dummy['doc_freq']=dummy[dummy.columns[1:len(dummy.columns)]].sum(axis=1)
dummy['doc_freq_norm']=1+ (np.log(40/dummy['doc_freq'])) #this is the IDF

#TF-IDF  DOC-TERM MATRIX:

for i in tf.columns[1:len(tf.columns)]:
    col_name=i
    tf[col_name]=tf[col_name]*dummy['doc_freq_norm']

#Similarity metrics:

#cosine similarities:

df1=pd.DataFrame()
for i in tf.columns[1:len(tf.columns)]:
    #i=tf.columns[1:len(tf.columns)][0]
    row_name=i
    df2 = {'DOC': [row_name]}
    df2=pd.DataFrame(df2)
    for j in tf.columns[1:len(tf.columns)]:
        #j=tf.columns[1:len(tf.columns)][0]
        col_name=j
        cs=cos_sim(tf[row_name],tf[col_name])
        df_temp={col_name:[cs]}
        df_temp=pd.DataFrame(df_temp)
        df2=pd.concat([df2,df_temp],join='outer',axis=1)
    df1=pd.concat([df1,df2],join='outer')

df1.to_csv('book1_similarities.csv',sep=';',index=False)








# old code:
doc1_input='BOOK1_DOCS/BOOK1_ZHWA4_W04_CLEANED.txt'
doc2_input='BOOK1_DOCS/BOOK1_ZHWA4_W05_CLEANED.txt'
doc3_input='BOOK1_DOCS/BOOK1_ZHWA4_W06_CLEANED.txt'

with open(doc1_input, 'r') as infile:
    doc1 = infile.readlines()

with open(doc2_input, 'r') as infile:
    doc2 = infile.readlines()

with open(doc3_input, 'r') as infile:
    doc3 = infile.readlines()


# n-grams
tdoc1=generate_gram_freq(doc1[0],1)
tdoc2=generate_gram_freq(doc2[0],1)
tdoc3=generate_gram_freq(doc3[0],1)


#TF: RAW Term FREQUENCY table

df_doc1=pd.DataFrame(tdoc1.items(),columns=['word','doc1'])
df_doc2=pd.DataFrame(tdoc2.items(),columns=['word','doc2'])
df_doc3=pd.DataFrame(tdoc3.items(),columns=['word','doc3'])

dfs = [df_doc1,df_doc2,df_doc3]

dfs.append(df_doc1)

d = reduce(lambda left,right: pd.merge(left,right,on='word',how='outer'), dfs)
d.fillna(0,inplace=True)

tf=d.copy()
dummy=d.copy()

# Normalizing TERM FREQUENCY per document: we don't want frequent words to overwhelm the frequency of others!


# TF NORMALIZATION:

np.seterr(divide='ignore')
for i in tf.columns[1:len(tf.columns)]:
    col_name=i
    # average word frequency per document:
    ad=tf[col_name].mean()

    #Normalizing function:
    #turning off divide by zero warning

    tf[col_name]=np.where(tf[col_name]>0,(1+np.log(tf[col_name]))/(1+np.log(ad)),0)
    #warning back on
    dummy[col_name]=np.where(dummy[col_name]>0,1,0)
np.seterr(divide = 'warn')


#IDF: INVERSE DOCUMENT FREQUENCY (WE WANT TO REDUCE THE IMPACT OF OCMMOM WORDS THAT APPEAR IN OTHER DOCUMENTS)


# TF-IDF : we don't want frequent words to overwhelm the frequency of others!

#NUMBER OF DOCUMENTS:
N_DOCS=len(d.columns)-1

dummy['doc_freq']=dummy[dummy.columns[1:len(dummy.columns)]].sum(axis=1)
dummy['doc_freq_norm']=1+ (np.log(40/dummy['doc_freq'])) #this is the IDF

#TF-IDF  DOC-TERM MATRIX:

for i in tf.columns[1:len(tf.columns)]:
    col_name=i
    tf[col_name]=tf[col_name]*dummy['doc_freq_norm']




# aplicar codigo previo para todos los documentos
# CALCULATE COSINE SIMILARITIES AND/OR JACCARD SIMILARITY

u=tf['doc1']
v=tf['doc2']

cos_sim(u,v)