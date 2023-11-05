import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import joblib


def classification_error(y_est, y_real):
    err = 0
    for y_e, y_r in zip(y_est, y_real):

        if y_e != y_r:
            err += 1

    return err / np.size(y_est)


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
model = clf.fit(X_train, y_train)

joblib.dump(model, 'modelo_entrenado.pkl')

# Predict the response for test dataset
y_pred = model.predict(X_test)

error = classification_error(y_pred, y_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:", metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:", metrics.recall_score(y_test, y_pred))
