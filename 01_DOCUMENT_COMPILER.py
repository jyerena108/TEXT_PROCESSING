




from os import listdir
from os.path import isfile, join
import pandas as pd
import shutil

#WE READ THE FILES:

onlyfiles = [f for f in listdir("BOOK1/") if isfile(join("BOOK1/", f))]
onlyfiles.sort()

# now they are numerically ordered


# WE READ THE DOCUMENT ORDER:
doc_order=pd.read_csv('BOOK_1_DOC_ORDER.csv')

#noW WE OPEN AND CONCATENATE EACH FILE TO COMPILE THE DOCS:

for i in range(0,40) :
    start_index=doc_order['START'][i]
    end_index=doc_order['END'][i]
    doc_name=doc_order['DOC_ID'][i]

    with open('BOOK1_DOCS/'+doc_name+'.txt','wb') as wfd:
        for f in onlyfiles[start_index:end_index+1]:
            with open("BOOK1/"+f,'rb') as fd:
                shutil.copyfileobj(fd, wfd)

