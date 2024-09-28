import pandas as pd

# Cargar el dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
column_names = [...]  # Lista de nombres de columnas (puedes encontrarlas en el link del dataset)
df = pd.read_csv(url, header=None, names=column_names)

# Separar características y etiquetas
X = df.iloc[:, :-1]  # Características (las primeras 57 columnas)
y = df.iloc[:, -1]   # Etiqueta (spam o no spam)

# Ver las primeras filas del dataset
df.head()


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Separar en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar modelo de regresión logística
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Obtener coeficientes del modelo
feature_importance = model.coef_[0]
important_features = pd.Series(feature_importance, index=X.columns).sort_values(ascending=False)
print(important_features.head(10))  # Las 10 características más importantes

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Realizar predicciones
y_pred = model.predict(X_test)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:\n", cm)

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Exactitud: {accuracy}")
print(f"Precisión: {precision}")
print(f"Tasa de Verdaderos Positivos (Recall): {recall}")
print(f"F1-Score: {f1}")
