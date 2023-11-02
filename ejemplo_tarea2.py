import numpy as np
import os, sys
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import cv2 as cv

ruta_proyecto = os.getcwd()
ruta_positivas = os.path.join(ruta_proyecto, 'positivos')
ruta_negativas = os.path.join(ruta_proyecto, 'negativos')

lista_positivas = os.listdir(ruta_positivas)
lista_negativas = os.listdir(ruta_negativas)
#print(ListaPositivas)

Yout = []
hog = cv.HOGDescriptor((128,64), (16,16), (8,8), (8,8), 9)
img_neg_orig = cv.imread(ruta_negativas+"/"+lista_negativas[0],0)
img_neg_resize = cv.resize(img_neg_orig, (128,64), interpolation = cv.INTER_AREA)
desc= hog.compute(img_neg_resize)
vectorTraining=desc.T
Yout.append(0);
#Cargar imagen
for i in range (1,len(lista_negativas)):#dirs.__len__() dirsNegativas.__len__()
    img_neg_orig = cv.imread(ruta_negativas+"/"+lista_negativas[i],0)
    img_neg_resize = cv.resize(img_neg_orig, (128,64), interpolation = cv.INTER_AREA)
    desc= hog.compute(img_neg_resize)
    vectorTraining=np.vstack([vectorTraining,desc.T])
    Yout.append(0);
    print (i)
i=i+1
for j in range (0,len(lista_positivas)):#dirs.__len__() dirsPositivas.__len__()
    img_pos_orig = cv.imread(ruta_positivas+"/"+lista_positivas[j],0)
    img_pos_resize = cv.resize(img_pos_orig, (128,64), interpolation = cv.INTER_AREA)
    desc= hog.compute(img_pos_resize)
    vectorTraining=np.vstack([vectorTraining,desc.T])
    Yout.append(1);
    print (i+j)
    #a.append(desc)
    #np.savetxt(pathHOG+dirs[i]+".txt", desc, delimiter=',')
#Finaliza el contador de segundos
#np.save("C:/numpyArrayExample",a)
#Guarda la matriz de training y los target
np.save(ruta_proyecto + '/X_train', vectorTraining)
np.save(ruta_proyecto + '/Y_train',Yout)



#toc = time.time()
#elapsed=toc-tic