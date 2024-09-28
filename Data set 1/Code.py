import pandas as pd

# Cargar el dataset

data = pd.read_csv('housing.csv')

# Mostrar las primeras filas para ver las características
print(data.head())

# Eliminar filas con valores faltantes
data = data.dropna()

# Separar características (X) y el target (y)
X = data.drop('median_house_value', axis=1)
y = data['median_house_value']

# Convertir las variables categóricas en variables dummy (si las hubiera)
X = pd.get_dummies(X, drop_first=True)

from sklearn.model_selection import train_test_split

# Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

# Crear el modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# Obtener los coeficientes del modelo
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])

# Mostrar la importancia de las características
print(coefficients.sort_values(by='Coefficient', ascending=False))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Hacer predicciones con el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular las métricas de evaluación
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Mostrar los resultados de las métricas
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-Squared (R2): {r2}')