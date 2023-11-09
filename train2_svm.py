from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import joblib
import numpy as np

ruta_proyecto = os.getcwd()

# Cargar datos procesados
X_train = np.load(ruta_proyecto + '/X_train.npy')
Y_train = np.load(ruta_proyecto + '/Y_train.npy')

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=42)

# Inicializar el clasificador SVM
svm_classifier = SVC(kernel='linear')  # Puedes ajustar el tipo de kernel según tu problema

# Entrenar el clasificador SVM
model = svm_classifier.fit(X_train, Y_train)
joblib.dump(model, 'modelo_entrenado2.pkl')


# Predecir las etiquetas para los datos de prueba
predictions = model.predict(X_test)

# Calcular la precisión del clasificador
accuracy = accuracy_score(Y_test, predictions)
print("Precisión del clasificador SVM: {:.2f}%".format(accuracy * 100))
