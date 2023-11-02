import numpy as np
import os, sys
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import cv2 as cv

ruta_proyecto = os.getcwd()

X = np.load(ruta_proyecto + '/X_train.npy')
Y = np.load(ruta_proyecto + '/Y_train.npy')
print(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=109)

print("train")
print(X_train.shape)
print(y_train.shape)
print("test")
print(X_test.shape)
print(y_test.shape)

# Create a svm Classifier
clf = svm.SVC(kernel='linear')  # Linear Kernel

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)
# Model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:", metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:", metrics.recall_score(y_test, y_pred))

# Abre el video
video_capture = cv.VideoCapture('placas.avi')

window_width, window_height = 100, 100

win_size = (128, 64)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
num_bins = 9

# ejemplo del profe
hog = cv.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
extension = ".jpg"
contador = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    x_detected, y_detected = 0, 0
    for y in range(0, frame_gray.shape[0] - win_size[1], int(win_size[1] / 2)):
        for x in range(0, frame_gray.shape[1] - win_size[0], int(win_size[0] / 2)):
            window = frame_gray[y:y + win_size[1], x:x + win_size[0]]


            features = hog.compute(window)
            # window_reshape = features.reshape(1, -1)

            window_reshape = np.vstack([features.T])  # transpuesta

            prediction = clf.predict(window_reshape)
            # print(prediction)
            if prediction == 1:
                cv.imwrite(f"./no_placas/placa_{x}_{y}_{contador}{extension}", window)
                contador = contador + 1
                x_detected, y_detected = x, y
                cv.imshow('Video_2', window)
                # cv.rectangle(frame, (x, y), (x + window_width, y + window_height), (0, 255, 0), 2)

    cv.rectangle(frame, (x_detected, y_detected), (x_detected + window_width, y_detected + window_height), (0, 255, 0),
                 2)
    cv.imshow('Video', frame)

    if cv.waitKey(33) & 0xFF == ord('q'):
        break

video_capture.release()
cv.destroyAllWindows()
