



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

# n-grams
tokens=generate_gram_freq(lines[0],1)

