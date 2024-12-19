import os
import urllib.request

url = 'https://github.com/rfarkas/student_data/raw/main/images/500faces.zip'
urllib.request.urlretrieve(url,'t.zip') #a t.zip a temporálisan notebookhoz rendelt tárhelyre kerül a Google felhőjében

#kitömörítjük a zip tartalmát
import zipfile
zipfile.ZipFile('t.zip').extractall('tmp_imgs')

import cv2 as cv #OpenCV

os.listdir('tmp_imgs')

rawImages = [] # a lista minden eleme egy eredeti kép
features = [] # a lista minden eleme egy kép pixelvektora
labels = [] # életkorok

for f in os.listdir('tmp_imgs'): # könyvtárbejárás
  image = cv.imread('tmpimgs/'+f) # kép beolvasása
  label = f.split('')[0] # a fájlnév első _ előtti száma adja meg a labelt, a helyes életkort

  #flatten a 3D tömbből 1D-t csinál (egymás után fűzi a sorokat)
  pixels = cv.resize(image,(64,64)).flatten() # egyenméretűre hozzuk a képeket! 32 x 32 x 3 = 3072 érték képenként

  rawImages.append(image)
  features.append(pixels)
  labels.append(label)

#random 20% használata
from sklearn.model_selection import train_test_split

trainFeatures,testFeatures,trainLabels,testLabels = train_test_split(features, labels, test_size=0.2)

from sklearn.linear_model import LinearRegression

reg = LinearRegression() # lineáris gép regresszióra
reg.fit(trainFeatures, trainLabels)
prediction = reg.predict(testFeatures)

from sklearn.metrics import mean_squared_error # MSE
mean_squared_error(prediction, testLabels)
