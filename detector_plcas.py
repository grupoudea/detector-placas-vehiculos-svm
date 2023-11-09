import numpy as np
import os, sys
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import cv2 as cv
import joblib

# cargar modelo
loaded_model = joblib.load('modelo_entrenado2.pkl')


# Abre el video
video_capture = cv.VideoCapture('./videos/placas.avi')
fps = video_capture.get(cv.CAP_PROP_FPS)
print(fps)

window_width, window_height = 64, 128

win_size = (64, 128)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
num_bins = 9

hog = cv.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
extension = ".jpg"
contador = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = frame

    x_detected, y_detected = 0, 0
    for y in range(0, frame_gray.shape[0] - win_size[1], int(win_size[1] / 2)):
        for x in range(0, frame_gray.shape[1] - win_size[0], int(win_size[0] / 2)):
            window = frame_gray[y:y + win_size[1], x:x + win_size[0]]

            features = hog.compute(window)
            # window_reshape = features.reshape(1, -1)

            window_reshape = np.vstack([features.T])  # transpuesta

            prediction = loaded_model.predict(window_reshape)
            # print(prediction)
            if prediction == 1:
                cv.imwrite(f"./no_placas_temp/placa_{contador}_{x}_{y}{extension}", window)
                contador = contador + 1
                x_detected, y_detected = x, y
                cv.imshow('Video_2', window)
                # cv.rectangle(frame, (x, y), (x + window_width, y + window_height), (0, 255, 0), 2)

    # cv.rectangle(frame, (x_detected, y_detected), (x_detected + window_width, y_detected + window_height), (0, 255, 0), 2)
    cv.imshow('Video', frame)

    if cv.waitKey(int(1000/fps)) & 0xFF == ord('q'):
        break

video_capture.release()
cv.destroyAllWindows()
