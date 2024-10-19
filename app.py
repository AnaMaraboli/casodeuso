import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import graphviz

# Conectar a la base de datos
conn = sqlite3.connect('predicciones.db')
cursor = conn.cursor()

# Crear la tabla si no existe
cursor.execute("""
CREATE TABLE IF NOT EXISTS predicciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    horas_suenio INTEGER,
    horas_estudio INTEGER,
    calificacion REAL
)
""")
conn.commit()

# Datos de ejemplo
np.random.seed(42)
horas_suenio = np.random.randint(5, 10, 50)
horas_estudio = np.random.randint(1, 5, 50)
calificaciones = horas_suenio * 2.5 + horas_estudio * 1.5 + np.random.normal(0, 2, 50)

# Crear dataframe
data = pd.DataFrame({'Horas de Sueño': horas_suenio, 
                     'Horas de Estudio': horas_estudio, 
                     'Calificación': calificaciones})

# Modelo de regresión
X = data[['Horas de Sueño', 'Horas de Estudio']]
y = data['Calificación']
modelo = LinearRegression().fit(X, y)

# Modelo de árbol de decisión
tree_model = DecisionTreeRegressor()
tree_model.fit(X, y)

# Interfaz de usuario
st.title("Predictor de Calificaciones")
suenio = st.slider("Horas de Sueño", 0, 10, 7)
estudio = st.slider("Horas de Estudio", 0, 10, 2)
prediccion = modelo.predict([[suenio, estudio]])[0]
st.write(f"Predicción de Calificación (Regresión Lineal): {prediccion:.2f}")

# Guardar la predicción
cursor.execute("INSERT INTO predicciones (horas_suenio, horas_estudio, calificacion) VALUES (?, ?, ?)", 
               (suenio, estudio, prediccion))
conn.commit()

st.write("¡Predicción guardada exitosamente!")

# Leer los datos de la tabla
data_from_db = pd.read_sql_query("SELECT * FROM predicciones", conn)

# Mostrar la tabla en Streamlit
st.subheader("Datos Guardados")
st.write(data_from_db)

# Graficar los resultados
st.subheader("Gráfico de Calificaciones")
plt.figure(figsize=(10, 6))
plt.scatter(data_from_db['horas_suenio'], data_from_db['calificacion'], color='blue', label='Calificaciones')
plt.scatter(data_from_db['horas_estudio'], data_from_db['calificacion'], color='red', label='Horas de Estudio')
plt.xlabel("Horas de Sueño / Horas de Estudio")
plt.ylabel("Calificación")
plt.title("Calificaciones por Horas de Sueño y Estudio")
plt.legend()
st.pyplot(plt)

# Visualizar el árbol de decisión
st.subheader("Visualización del Árbol de Decisión")

dot_data = export_graphviz(tree_model, out_file=None, 
                           feature_names=['Horas de Sueño', 'Horas de Estudio'],  
                           filled=True, rounded=True,  
                           special_characters=True)  
graph = graphviz.Source(dot_data)  
st.graphviz_chart(dot_data)  # Muestra el árbol de decisión en Streamlit

# Cerrar la conexión
conn.close()


