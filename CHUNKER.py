


# WE USE OUTFILE1 ( STRIPPING THE FIRST PAGES AND LAST PAGES FIRST)

from collections import defaultdict
import numpy
import matplotlib.pyplot as plt
import math


filename='outfile1.txt'

with open(filename, 'r') as infile:
    lines = infile.readlines()

# Creating a function to generate N-Grams
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

def trim_nga(ngramfreq,value):
    ngnew = dict()
    for (word, freq) in ngramfreq.items():
        if freq >= value:
            ngnew[word] = freq
    return(ngnew)

def trim_nga2(ngramfreq,umbral_rel_freq):
    max_freq = max(list(ngramfreq.values()))
    ngnew = dict()
    for (word, freq) in ngramfreq.items():
        if (freq/max_freq) >= umbral_rel_freq:
            ngnew[word] = freq
    return(ngnew)

def ngram_reduct1(ng1,ng2,umbral):
    #ng1 = smaller n-gram e.g. 8-gram
    #ng2 = larger N-gram e.g. 9-gram
    ngnew=dict()
    #counter=0
    for (word2, freq2) in ng2.items():
        #counter+=1
        #if (counter%25==0):
         #   print(counter)
        for (word1,freq1) in ng1.items():
            if word1.split()==word2.split()[:-1] :
                if  (freq2/freq1)>=umbral:
                     ngnew[word2]=freq2
    return (ngnew)

def ngram_reduct2(ng1,ng2,umbral):
    #ng1 = smaller n-gram e.g. 8-gram
    #ng2 = larger N-gram e.g. 9-gram
    ng1new=dict()
    ng2new=dict()
    #counter=0
    for (word2, freq2) in ng2.items():
        #counter+=1
        #if (counter%25==0):
         #   print(counter)
        for (word1,freq1) in ng1.items():
            if word1.split()==word2.split()[:-1] :
                if  (freq2/freq1)>=umbral:
                     ng2new[word2]=freq2
                     ng1new[word1]=freq1-freq2
                else :
                    ng1new[word1] = freq1
            else:
                ng1new[word1] = freq1

    return (ng1new,ng2new)

# n-grams
ng6=generate_gram_freq(lines[0],6)
ng5=generate_gram_freq(lines[0],5)
ng4=generate_gram_freq(lines[0],4)
ng3=generate_gram_freq(lines[0],3)
ng2=generate_gram_freq(lines[0],2)
ng1=generate_gram_freq(lines[0],1)


# step 1: Frequency based reduction (remove the n-grams with frequency 1)

n=2 #keep 2 or more

ng6_1=trim_nga(ng6,n)
ng5_1=trim_nga(ng5,n)
ng4_1=trim_nga(ng4,n)
ng3_1=trim_nga(ng3,n)
ng2_1=trim_nga(ng2,n)
ng1_1=trim_nga(ng1,n)


# set comparison methodology for cleaning:


ng6_2=ngram_reduct1(ng5_1,ng6_1,0.7)
ng5_2_v2,ng6_2_v2=ngram_reduct2(ng5,ng6_1,0.7)


# 6:23pm start
# 1000 at 6:



a5='mi thams cad mkhyen pa'
b4='mi thams cad mkhyen'
c4='mi cad thams mkhyen'


v = list(ng13.values())
max(v)



# identifying distribution of frequency to determine cutoff threshold
v = list(ng6.values())
numpy.quantile(v,[0.99])
# 99% of the 6-grams are noise ( each one repeats 3 times at most !)
# so we remove them:
ng6_2=trim_nga(ng6,50)

#Deep analysis:
v = list(ng6_2.values())
plt.hist(v, bins = 30)
plt.show()

n3freq3=dict()
for (word,freq) in n3freq2.items():
    if freq>15:
        n3freq3[word]=freq

v = list(n3freq3.values())
plt.hist(v, bins = 30)
plt.show()


v = list(n1freq.values())
numpy.quantile(v,[0,0.25,0.5,0.75,0.90,0.99,1])
# 75% of the 1-grams ae noise ( each one repeats 27 times at most !)
# so we remove them:
n1freq2=dict()
for (word,freq) in n1freq.items():
    if freq>27:
        n1freq2[word]=freq

v = list(n1freq2.values())
plt.hist(v, bins = 10)
plt.show()



a=400
b=80
if a>b:
    if a%b==0:
        print('a>b y a%b SI es cero')
    else:
        print('a>b PERO a%b NO es cero')
else:
    print('a no es mayor a b')



400%80
