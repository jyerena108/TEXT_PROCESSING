

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
text=" ".join(df["COMMENT"])
words = text.split()

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