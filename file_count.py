from importlib.metadata import files
import os
import glob

dir = 'C:/Users/kato/Desktop/from_app_ver2/Aug/*/*/*.jpg'

sio=[]
shoyu=[]
sosu=[]
for fold_path in glob.glob('C:/Users/kato/Desktop/from_app_ver2/Aug/sio/*/*.jpg'):
    sio.append(fold_path)
for fold_path in glob.glob('C:/Users/kato/Desktop/from_app_ver2/Aug/shoyu/*/*.jpg'):
    shoyu.append(fold_path)
for fold_path in glob.glob('C:/Users/kato/Desktop/from_app_ver2/Aug/sosu/*/*.jpg'):
    sosu.append(fold_path)
print("sio:",len(sio))
print("shoyu:",len(shoyu))
print("sosu:",len(sosu))