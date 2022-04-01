#



# First we add the footer to each text file:

from os import listdir
from os.path import isfile, join
import shutil

DIR='BOOK6/'
OUTFILE_NAME='BOOK6_COMPILATION.txt'

onlyfiles = [f for f in listdir(DIR) if isfile(join(DIR, f))]

onlyfiles.sort()

# THIS ADDS THE PAGE NUMBER AT THE END OF EACH FILE
for i in onlyfiles[0:(len(onlyfiles)+1)] :
    with open(DIR+i, "a") as myfile:
        myfile.write("\n \n"+ i +"\n \n")

# Now we append everything into one single txt file:
with open(OUTFILE_NAME,'wb') as wfd:
    for f in onlyfiles[0:(len(onlyfiles)+1)]:
        with open(DIR+f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)

