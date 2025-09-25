import streamlit as st
import joblib
import pickle
import numpy as np

import psycopg2
# Fetch variables
USER = "postgres.ltamddcsopohwviwqjmy" #os.getenv("user")
PASSWORD = "Parisll"# os.getenv("password")
HOST = "aws-1-us-east-2.pooler.supabase.com" #os.getenv("host")
PORT = "6543" #os.getenv("port")
DBNAME = "postgres" #os.getenv("dbname")

# Configuración de la página
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")

# --- DB helpers ---
def get_connection():
    return psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME,
    )

def save_prediction(sepal_length, sepal_width, petal_length, petal_width, predicted_species):
    """
    Inserta un registro en table_iris.
    Columnas:
      ls = sepal_length
      "as" = sepal_width   (entre comillas por palabra reservada)
      lp = petal_length
      ap = petal_width
      prediction = predicted_species
      created_at = NOW()
    """
    sql = """
    INSERT INTO table_iris (created_at, lp, ls, ap, "as", prediction)
    VALUES (NOW(), %s, %s, %s, %s, %s);
    """
    params = (
        float(petal_length),   # lp
        float(sepal_length),   # ls
        float(petal_width),    # ap
        float(sepal_width),    # "as"
        str(predicted_species) # prediction
    )
    conn = None
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
        return True, None
    except Exception as e:
        return False, str(e)
    finally:
        if conn:
            conn.close()

# --- Probar conexión (opcional/diagnóstico) ---
try:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT NOW();")
            result_now = cursor.fetchone()
            # Muestra hora del servidor como diagnóstico
            st.caption(f"🗓️ Hora del servidor DB: {result_now[0]}")
except Exception as e:
    st.warning(f"No se pudo conectar a la BD para diagnóstico: {e}")

# --- Modelos ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load('components/iris_model.pkl')
        scaler = joblib.load('components/iris_scaler.pkl')
        with open('components/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo en la carpeta 'components/'.")
        return None, None, None

st.title("🌸 Predictor de Especies de Iris")

model, scaler, model_info = load_models()

if model is not None:
    st.header("Ingresa las características de la flor:")

    sepal_length = st.number_input("Longitud del Sépalo (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    sepal_width  = st.number_input("Ancho del Sépalo (cm)",    min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    petal_length = st.number_input("Longitud del Pétalo (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    petal_width  = st.number_input("Ancho del Pétalo (cm)",    min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    save_to_db = st.checkbox("Guardar esta predicción en la base de datos", value=True)

    if st.button("Predecir Especie"):
        # Preparar datos
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Estandarizar
        features_scaled = scaler.transform(features)

        # Predecir
        prediction_idx = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        target_names = model_info['target_names']
        predicted_species = target_names[prediction_idx]

        # Mostrar resultado
        st.success(f"Especie predicha: **{predicted_species}**")
        st.write(f"Confianza: **{max(probabilities):.1%}**")

        st.write("Probabilidades:")
        for species, prob in zip(target_names, probabilities):
            st.write(f"- {species}: {prob:.1%}")

        # Guardar en BD (opcional)
        if save_to_db:
            ok, err = save_prediction(
                sepal_length=sepal_length,
                sepal_width=sepal_width,
                petal_length=petal_length,
                petal_width=petal_width,
                predicted_species=predicted_species
            )
            if ok:
                st.toast("✅ Predicción guardada en table_iris.", icon="✅")
            else:
                st.error(f"No se pudo guardar en la BD: {err}")