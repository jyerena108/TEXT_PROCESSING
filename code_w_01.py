

# Loading Tabular Data

import os
import pandas as pd

os. getcwd()
# Extract text column from a dataframe
df = pd.read_csv("SAMPLE_TEXT_COMMENTS_GM.csv")




# Convert text column to lowercase
df['COMMENT'] = df['COMMENT'].str.lower()
df.head()

#