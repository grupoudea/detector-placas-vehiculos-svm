import numpy as np
import os
import cv2 as cv

ruta_proyecto = os.getcwd()
ruta_positivas = os.path.join(ruta_proyecto, 'positivos')
ruta_negativas = os.path.join(ruta_proyecto, 'no_placas_64_128')

lista_positivas = os.listdir(ruta_positivas)
lista_negativas = os.listdir(ruta_negativas)
print(f"cantidad de positivas {len(lista_positivas)}")
print(f"cantidad negativas {len(lista_negativas)}")

win_size = (64, 128)

Yout = []
hog = cv.HOGDescriptor(win_size, (16, 16), (8, 8), (8, 8), 9)
vectorTraining = []


# Procesamiento de imágenes negativas
for i in range(len(lista_negativas)):
    img_neg_orig = cv.imread(ruta_negativas + "/" + lista_negativas[i])
    if img_neg_orig is None:
        continue  # Omitir si la imagen no se pudo leer
    img_neg_resize = cv.resize(img_neg_orig, win_size, interpolation=cv.INTER_AREA)
    desc = hog.compute(img_neg_resize)
    vectorTraining.append(desc.T)
    Yout.append(0) # SI ES NEGATIVA YO DECIDO PONERLE CERO
    print(i)

# Procesamiento de imágenes positivas
for j in range(len(lista_positivas)):
    img_pos_orig = cv.imread(ruta_positivas + "/" + lista_positivas[j])
    if img_pos_orig is None:
        continue  # Omitir si la imagen no se pudo leer
    img_pos_resize = cv.resize(img_pos_orig, win_size, interpolation=cv.INTER_AREA)
    desc = hog.compute(img_pos_resize)
    vectorTraining.append(desc.T)
    Yout.append(1) # SI ES POSITIVA DECIDO PONERLE 1
    print(len(lista_negativas) + j)

# Convertir la lista de descriptores en un array Numpy
vectorTraining = np.array(vectorTraining)

# Guardar los datos procesados
np.save(ruta_proyecto + '/X_train', vectorTraining) # LUCE COMO MATRIZ
np.save(ruta_proyecto + '/Y_train', Yout) # LUCE COMO COLUMNA

# toc = time.time()
# elapsed=toc-tic
