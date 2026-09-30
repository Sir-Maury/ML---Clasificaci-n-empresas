import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Clasificador de Riesgo Empresarial", page_icon="🏢")

st.title("🏢 Clasificador de Riesgo de Empresas")
st.write("Herramienta interactiva para explorar un modelo de Machine Learning básico.")

# 1. Creación de un dataset sintético para la demostración
@st.cache_resource
def entrenar_modelo():
    np.random.seed(42)
    n = 300
    
    anos = np.random.randint(1, 20, n)
    digitalizacion = np.random.randint(1, 6, n)
    margen = np.random.uniform(5, 40, n)
    
    # Regla simple para etiquetar los datos de entrenamiento
    riesgo = []
    for a, d, m in zip(anos, digitalizacion, margen):
        puntaje = a * 0.4 + d * 2 + m * 0.5
        if puntaje > 25:
            riesgo.append("Riesgo Bajo")
        elif puntaje > 15:
            riesgo.append("Riesgo Medio")
        else:
            riesgo.append("Riesgo Alto")
            
    df = pd.DataFrame({
        'Anos_Trayectoria': anos,
        'Nivel_Digitalizacion': digitalizacion,
        'Margen_Ganancia': margen,
        'Riesgo': riesgo
    })
    
    X = df[['Anos_Trayectoria', 'Nivel_Digitalizacion', 'Margen_Ganancia']]
    y = df['Riesgo']
    
    modelo = DecisionTreeClassifier(max_depth=3)
    modelo.fit(X, y)
    
    return modelo

modelo = entrenar_modelo()

# 2. Interfaz de usuario para ingresar datos
st.sidebar.header("📥 Datos de la Empresa")

anos_input = st.sidebar.slider("Años de trayectoria:", 1, 30, 5)
digi_input = st.sidebar.slider("Nivel de digitalización (1-5):", 1, 5, 3)
margen_input = st.sidebar.slider("Margen de ganancia (%):", 0.0, 50.0, 15.0)

# 3. Predicción
datos_nuevos = pd.DataFrame([[anos_input, digi_input, margen_input]], 
                            columns=['Anos_Trayectoria', 'Nivel_Digitalizacion', 'Margen_Ganancia'])

prediccion = modelo.predict(datos_nuevos)[0]
probabilidades = modelo.predict_proba(datos_nuevos)[0]

# 4. Mostrar Resultados
st.subheader("📊 Resultado del Análisis")

if prediccion == "Riesgo Bajo":
    st.success(f"**Resultado:** {prediccion} 🟢")
elif prediccion == "Riesgo Medio":
    st.warning(f"**Resultado:** {prediccion} 🟡")
else:
    st.error(f"**Resultado:** {prediccion} 🔴")

# Mostrar probabilidades por categoría
st.write("### Probabilidades asociadas:")
probs_df = pd.DataFrame({
    'Categoría': modelo.classes_,
    'Probabilidad (%)': (probabilidades * 100).round(2)
})
st.dataframe(probs_df, use_container_width=True)