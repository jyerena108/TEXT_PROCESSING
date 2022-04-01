


from os import listdir
from os.path import isfile, join
import gensim
from gensim.models.doc2vec import TaggedDocument
import pandas as pd
INPUT_DIR='BOOK1_CLEANED/'


#LOAD DOC NAMES:

docLabels = []
docLabels = [f for f in listdir(INPUT_DIR) if f.endswith('.txt')]
docLabels.sort()

doctags=[]
docs = []
for doc in docLabels[0:len(docLabels)]:
    doc_name=doc[:-12]
    infilename=INPUT_DIR+doc

    with open(infilename, 'r') as infile:
        lines = infile.readlines()
        line=lines[0]
        tag=doc_name
        doctags.append(tag)
        docs.append(TaggedDocument(words=line.split(), tags=[tag]))


model = gensim.models.Doc2Vec(vector_size=300, window=5, min_count=5, workers=4, epochs=20)
model.build_vocab(docs)

model.train(docs, total_examples=model.corpus_count,epochs=500)

model.save('modelo01.model')

#loading the model
d2v_model = gensim.models.doc2vec.Doc2Vec.load('modelo01.model')

#to get most similar document with similarity scores using document- name
sims = d2v_model.dv.most_similar('BOOK1_ZHWA4_W04',topn=10)
print(sims)

#visualization

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

X=d2v_model[doctags]
print(X)
tsne=TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
df = pd.DataFrame(X_tsne, index=doctags, columns=['x', 'y'])


for idx in range(40):
    plt.scatter(df['x'],df['y'], color='steelblue')
    plt.annotate(doctags[idx], (df['x'][idx], df['y'][idx]), alpha=0.7)

