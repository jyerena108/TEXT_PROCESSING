

# Loading Tabular Data

import os
import pandas as pd
from collections import Counter, defaultdict

os. getcwd()
#dataframe with text column
df = pd.read_csv("SAMPLE_TEXT_COMMENTS_GM.csv")


#### Data Cleaning#####
# Convert text column to lowercase
df['COMMENT'] = df['COMMENT'].str.lower()
df.head()

# remove special characters:
import re
# Remove punctuation characters ( only keep letters and numbers)
df['COMMENT'] =df['COMMENT'].str.replace(r"[^a-zA-Z0-9]", " ",regex=True )
df.head()

# Frequency of Tokens:

#Create dataset:

#there are rows with missing comments we exclude those missing values and create long string
ex=['nan']
text=" ".join(w for w in df["COMMENT"].astype(str) if w not in ex )

#tokenize
words = text.split()
#Remove english stop words ( source: nltk stopwords.words("english"))
stop=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
words = [w for w in words if w not in stop]

# Auxiliary function:
def get_n_most_common(n, list_of_words,type='count'):
    ct = Counter(list_of_words)
    d = defaultdict(list)
    for word, quantity in ct.items():
        d[quantity].append(word)
    most_common = sorted(d.keys(), reverse= True)
    if type=='freq':
        return [(word, val/len(list_of_words)) for val in most_common[:n] for word in d[val]]
    else:
        return [(word, val) for val in most_common[:n] for word in d[val]]

#Top N frequency of words
get_n_most_common(1,words,'freq')

