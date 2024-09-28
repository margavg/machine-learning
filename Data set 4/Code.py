import pandas as pd

# Cargar el dataset desde un archivo local o una URL
data = pd.read_csv('creditcard.csv')

# Mostrar las primeras filas del dataset
print(data.head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Dividir las características (X) y la variable objetivo (y)
X = data.drop(columns='Class')  # 'Class' es la columna que indica fraude (1) o no (0)
y = data['Class']

# Dividir los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos para que las características tengan la misma escala
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.svm import SVC

# Crear un modelo SVM con kernel RBF
svm_model = SVC(kernel='rbf', random_state=42)

# Entrenar el modelo con los datos de entrenamiento
svm_model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Hacer predicciones con el conjunto de prueba
y_pred = svm_model.predict(X_test)

# Calcular las métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Mostrar los resultados
print(f'Exactitud (Accuracy): {accuracy}')
print(f'Precisión (Precision): {precision}')
print(f'Recall: {recall}')
print(f'Puntaje F1 (F1 Score): {f1}')