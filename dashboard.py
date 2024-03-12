import streamlit as st
import mlflow.pyfunc
import pandas as pd
import shap

# from pycaret.classification import plot_model

# Décorateur pour mettre en cache les données
@st.cache_data
def load_data(nom_fichier):
    """Charger les données à partir d'un fichier CSV."""
    return pd.read_csv(nom_fichier)

# Décorateur pour mettre en cache le modèle
@st.cache_resource
def load_model(model_uri):
    return mlflow.pyfunc.load_model(model_uri)

# Chargement des données
nom_fichier = 'data/cleaned/application_train_cleaned.csv'
data = load_data(nom_fichier)

# Chargement du modèle
model_name = "model_light_gbm_best"
model_version = 3
model_uri = f"models:/{model_name}/{model_version}"
lightgbm_model = load_model(model_uri)

# Créer un widget de sélection pour choisir un SK_ID_CURR
selected_sk_id_curr = st.selectbox(
    'Choisissez un SK_ID_CURR',
    data['SK_ID_CURR'].unique()
)

# Obtenir les données pour le SK_ID_CURR sélectionné
selected_data = data[data['SK_ID_CURR'] == selected_sk_id_curr]

# Ajouter un bouton pour déclencher la prédiction
if st.button('Calculer la prédiction'):

    # Utiliser le modèle pour faire une prédiction
    prediction = lightgbm_model.predict(selected_data)

    # Afficher la prédiction
    if prediction == 0:
        st.markdown(
            '<div style="background-color: #98FB98; padding: 10px;'
            'border-radius: 5px; color: #000000;"'
            '>Le prêt est accordé.</div>',
            unsafe_allow_html=True
        )

    elif prediction == 1:
        st.markdown(
            '<div style="background-color: #FF6347; padding: 10px;'
            'border-radius: 5px; color: #000000;"'
            '>Le prêt n\'est pas accordé.</div>',
            unsafe_allow_html=True
        )

    else:
        st.write('Prédiction inconnue :', prediction)

# Ajouter un bouton pour déclencher le calcul des features importances
if st.button('Calculer les importances des caractéristiques'):

    # Créer un explainer SHAP
    explainer = shap.TreeExplainer(lightgbm_model)

    # Calculer les valeurs SHAP
    shap_values = explainer.shap_values(selected_data)

    # Afficher le résumé des importances des caractéristiques
    st.write(shap.summary_plot(shap_values, selected_data))
