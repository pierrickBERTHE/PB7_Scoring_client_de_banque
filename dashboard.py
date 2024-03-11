from sklearn import pipeline
import streamlit as st
import mlflow.pyfunc
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

# Charger les données
data = pd.read_csv('data/cleaned/application_train_cleaned.csv')

# Le nom et la version du modèle
model_name = "model_light_gbm_best"
model_version = 3

# L'URI du modèle enregistré
model_uri = f"models:/{model_name}/{model_version}"

# Charger le modèle
model_wrapper = mlflow.pyfunc.load_model(model_uri)

# Extraire le modèle LightGBM du pipeline
lightgbm_model = model_wrapper.named_steps['modele']

# Créer un widget de sélection pour choisir un SK_ID_CURR
selected_sk_id_curr = st.selectbox(
    'Choisissez un SK_ID_CURR',
    data['SK_ID_CURR'].unique()
)

# Obtenir les données pour le SK_ID_CURR sélectionné
selected_data = data[data['SK_ID_CURR'] == selected_sk_id_curr]

# Ajouter un bouton pour déclencher la prédiction et le calcul des importances des caractéristiques
if st.button('Calculer la prédiction et les importances des caractéristiques'):

    # Utiliser le modèle pour faire une prédiction
    prediction = lightgbm_model.predict(selected_data)

    # Afficher la prédiction
    st.write('Prédiction :', prediction)

    # Calculer les valeurs SHAP
    explainer = shap.TreeExplainer(lightgbm_model)
    shap_values = explainer.shap_values(selected_data)

    # Afficher le résumé des valeurs SHAP
    st.pyplot(shap.summary_plot(shap_values, selected_data, plot_type="bar"))