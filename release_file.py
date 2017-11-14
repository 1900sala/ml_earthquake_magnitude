import tarfile
import os

def un_tarfile(file_name,tag):
    tar=tarfile.open(file_name)
    names=tar.getnames()
    if tag==1:
        for name in names:
            print(name)
            if name[-10:-7]=='kik':
              tar.extract(name, '../zip_sac/')
            if name[-10:-7]=='knt':
               tar.extract(name,  '../zip_sac/')
        tar.close()
    if tag==2:
        for name in names:
            # print(name)
            if name[-2:]=='EW' or name[-2:]=='NS' or name[-2:]=='UD' or name[-2:]=='D2' or name[-2:]=='S2' or name[-2:]=='W2' :
              tar.extract(name, '../data/')
        tar.close()

def eachFile(filepath,tag):
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath, allDir))
        print(allDir)
        un_tarfile(str(child),tag)

# zip_data = '../zip_data/'
# tag=1
# eachFile(zip_data,tag)
zip_sac = '../zip_sac/'
tag=2
eachFile(zip_sac,tag)
