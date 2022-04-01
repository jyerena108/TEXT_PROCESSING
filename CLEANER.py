


from os import listdir
from os.path import isfile, join
import re
import sys
from unidecode import unidecode
import shutil

onlyfiles = [f for f in listdir("BOOK1/") if isfile(join("BOOK1/", f))]
onlyfiles.sort()

#compile the whole book into one big text file:
with open('BOOK1_COMPILATION_NOFOOTER.txt','wb') as wfd:
    for f in onlyfiles[2:978]:
        with open("BOOK1/"+f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)

# now we proceed to remove dirty characters and replace achun for "v"

    infilename='BOOK1_COMPILATION_NOFOOTER.txt'
    outfilename='outfile1.txt'
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

















#OLD CODE:

onlyfiles = [f for f in listdir("EXAMPLEFOLDER/") if isfile(join("EXAMPLEFOLDER/", f))]
onlyfiles.sort()

-
    infilename='EXAMPLEFOLDER/' + onlyfiles[1]

    infilename='EXAMPLEFOLDER/' +'BOOK1_COMPILATION_11_18.txt'
    outfilename='EXAMPLEFOLDER/'+'outfile1.txt'
    # dirty characters
    dic_pattern = {'[': ' ', ']': ' ', '_': ' ', '/': ' ', '—': ' ', '–': ' ', '#': ' ', '@': ' ',':':' ','.':' ','|':' ','-':' ','》':' ','(':' ',')':' ',
                   '0':' ','1':' ','2':' ','3':' ','4':' ','5':' ','6':' ','7':' ','8':' ','9':' '}
    dic_pattern = dict((re.escape(k), v) for k, v in dic_pattern.items())
    pattern = re.compile("|".join(dic_pattern.keys()))
    # ' for "v" and gi ("'i") particle :
    dic_pattern2 = {' \'': ' v', '\'i ': ' vi ','”':' ','"':' '}
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
            newline=(' '.join(text3.split()))
            outfile.write(' '+newline)






