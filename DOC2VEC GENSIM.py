


# source code: https://medium.com/@klintcho/doc2vec-tutorial-using-gensim-ab3ac03d3a1




from os import listdir
from os.path import isfile, join
import gensim
from gensim.models.doc2vec import TaggedDocument

INPUT_DIR='BOOK1_CLEANED/'


#LOAD DOC NAMES:

docLabels = []
docLabels = [f for f in listdir(INPUT_DIR) if f.endswith('.txt')]
docLabels.sort()

#Load content of the DOCS:

data = []
for doc in docLabels:
    data.append(open(INPUT_DIR + doc,'r'))

# Preparing the data for Gensim Doc2vec



it =TaggedDocument(data, docLabels)


model = gensim.models.Doc2Vec(vector_size=300, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025) # use fixed learning rate
model.build_vocab(it)
for epoch in range(10):
    model.train(it)
    model.alpha -= 0.002 # decrease the learning rate
    model.min_alpha = model.alpha # fix the learning rate, no deca
    model.train(it)


s1 = 'the quick fox brown fox jumps over the lazy dog the quick fox brown fox jumps over the lazy dog the quick fox brown fox jumps over the lazy dog'
s1_tag = '001'
s2 = 'i want to burn a zero-day the quick fox brown fox jumps over the lazy dog the quick fox brown fox jumps over the lazy dog the quick fox brown fox jumps over the lazy dog'
s2_tag = '002'
s3=' i want to break free i want to break free i want to break free i want to break free i want to love you i want to love you i want to love you i want to love you'
s3_tag='003'
s4='  lucas talked about freedom because free is his love for science lucas talked about freedom because free is his love for science '
s4_tag='004'
s5 = 'the quick fox brown fox jumps over the lazy dog the quick fox brown fox jumps over the lazy dog the quick fox brown fox jumps over the lazy dog bert'
s5_tag = '005'

docs = []
docs.append(TaggedDocument(words=s1.split(), tags=[s1_tag]))
docs.append(TaggedDocument(words=s2.split(), tags=[s2_tag]))
docs.append(TaggedDocument(words=s3.split(), tags=[s3_tag]))
docs.append(TaggedDocument(words=s4.split(), tags=[s4_tag]))
docs.append(TaggedDocument(words=s5.split(), tags=[s5_tag]))


model = gensim.models.Doc2Vec(vector_size=300, window=5, min_count=5, workers=4, epochs=20)
model.build_vocab(docs)

model.train(docs, total_examples=model.corpus_count, epochs=5)

model.save('modelo01.model')


#loading the model
d2v_model = gensim.models.doc2vec.Doc2Vec.load('modelo01.model')
#start testing:


#printing the vector of document at index 1 in docLabels
docvec = d2v_model.dv[2]
print(docvec)
#printing the vector of the file using its name
docvec = d2v_model.dv['001'] #if string tag used in training
print(docvec)
#to get most similar document with similarity scores using document-index
similar_doc = d2v_model.dv.most_similar(1)
print(similar_doc)
#to get most similar document with similarity scores using document- name
sims = d2v_model.dv.most_similar('001')
print(sims)
#to get vector of document that are not present in corpus
docvec = d2v_model.docvecs.infer_vector(‘war.txt’)
print docvec














