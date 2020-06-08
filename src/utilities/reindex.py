import os, fnmatch

data_dir = '/home/kevin/data/michigan_gt/2012_05_11/'

text_file = open(data_dir + "index.txt", "w")

listOfFiles = os.listdir(data_dir + 'images')
listOfFiles.sort()

pattern = "*.png"
for entry in listOfFiles:
    if fnmatch.fnmatch(entry, pattern):
        text_file.write("%s\n" % entry[:-4])
        print (entry[:-4])

text_file.close()