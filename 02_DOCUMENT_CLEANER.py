


from os import listdir
from os.path import isfile, join
import re
import sys
from unidecode import unidecode
import shutil


DOCS_DIR="BOOK1_DOCS/"
OUTPUT_DIR='BOOK1_CLEANED/'

onlyfiles = [f for f in listdir(DOCS_DIR) if isfile(join(DOCS_DIR, f))]
onlyfiles.sort()

#for outfile names:
name_pattern={'.txt':'_CLEANED.txt'}
name_pattern = dict((re.escape(k), v) for k, v in name_pattern.items())
namepattern = re.compile("|".join(name_pattern.keys()))


#we iterate on each doc:
for doc in onlyfiles[1:len(onlyfiles)]:

    #we open the file:

    infilename=DOCS_DIR+doc
    outfilename=OUTPUT_DIR+namepattern.sub(lambda m: name_pattern[re.escape(m.group(0))], doc)


    # now we proceed to remove dirty characters and replace achun for "v"
    # dirty characters
    dic_pattern = {'[': ' ', ']': ' ', '_': ' ', '/': ' ', '—': ' ', '–': ' ', '#': ' ', '@': ' ',':':' ','.':' ','|':' ','-':' ','》':' ','(':' ',')':' ',
                   '0':' ','1':' ','2':' ','3':' ','4':' ','5':' ','6':' ','7':' ','8':' ','9':' ','=':' ','\\':' ','*':' ','>':' ','<':' ','?':' ','%':' ','¿':' ','~':' ',
                   ',':' '}
    dic_pattern = dict((re.escape(k), v) for k, v in dic_pattern.items())
    pattern = re.compile("|".join(dic_pattern.keys()))
    # ' for "v" and gi ("'i") particle :
    dic_pattern2 = {' \'': ' v', '\'i ': ' vi ','”':' ','"':' ',' +':'',' + ':''}
    dic_pattern2 = dict((re.escape(k), v) for k, v in dic_pattern2.items())
    pattern2 = re.compile("|".join(dic_pattern2.keys()))
    # ' in the rest of the cases
    dic_pattern3 = {'\'': 'v'}
    dic_pattern3 = dict((re.escape(k), v) for k, v in dic_pattern3.items())
    pattern3 = re.compile("|".join(dic_pattern3.keys()))

    with open(infilename, 'r') as infile:
        lines = infile.readlines()

    with open(outfilename,'w') as outfile:
        for line in lines:
            line = unidecode(line)
            text1 = pattern.sub(lambda m: dic_pattern[re.escape(m.group(0))], line)
            text2 = pattern2.sub(lambda m: dic_pattern2[re.escape(m.group(0))], text1)
            text3 = pattern3.sub(lambda m: dic_pattern3[re.escape(m.group(0))], text2)
            text3=text3.lower()
            newline=(' '.join(text3.split()))
            outfile.write(' '+newline)














