"""
Description: Ce fichier contient le dashboard de l'application Streamlit.

Author: Pierrick Berthe
Date: 2024-03-21
"""

# ============== étape 1 : Importation des librairies ====================

import pandas as pd
import numpy as np
import requests
import os
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st
import zipfile
from memory_profiler import profile

# ================= étape 2 : Chemins environnement ========================

# Détecteur si besoin de changer le chemin pour le déploiement Streamlit
CHEMIN_POUR_DEPLOIEMENT_STREAMLIT = True

# Environnement de l'API (local ou distant)
env_API = "distant"

# Titre de l'application
st.title('Projet 7\n')
st.title('Élaborez le modèle de scoring - Dashboard\n')

# Répertoire racine
ROOT_DIR = os.getcwd()
print("ROOT_DIR:",ROOT_DIR, "\n")

# URL de l'API Flask pour prédiction (local ou distant)
if env_API == 'local':
    URL_API_PREDICT = 'http://127.0.0.1:5000/predict'
else:
    URL_API_PREDICT = 'http://pierrickberthe.eu.pythonanywhere.com/predict'
print("URL_API_PREDICT:",URL_API_PREDICT, "\n")

# SI besoin de changer le chemin pour le déploiement Streamlit
if CHEMIN_POUR_DEPLOIEMENT_STREAMLIT:
    ROOT_DIR = os.path.join(ROOT_DIR, "Berthe_Pierrick_4_dossier_code_022024")
else:
    pass

# Chemin des données
DATA_PATH = os.path.join(
    ROOT_DIR, "data/cleaned", "application_train_cleaned_frac_1%.zip"
)
print("DATA_PATH:",DATA_PATH, "\n")

# Chemin du répertoire pour sauvegarder le plot
FIG_DIR = os.path.join(ROOT_DIR, "figure")
print("FIG_DIR:",FIG_DIR, "\n")

# Chemin du modèle
MODEL_PATH = os.path.join(ROOT_DIR, "mlflow_model", "model.pkl")
print("MODEL_PATH:",MODEL_PATH, "\n")

# ==================== étape 3 : chargement modèle ==========================

@st.cache_data
def load_model(model_path):
    return joblib.load(model_path)

# Chargement du modèle pré-entraîné
model = load_model(MODEL_PATH)

# ==================== étape 4 : chargement data ==========================

@st.cache_data
def load_data(file_path, file_name_csv, _model):
    """
    Charge les données à partir d'un fichier CSV et les transforme en
    utilisant un modèle donné.

    Args:
        file_path (str): Chemin vers le fichier CSV.
        _model (imblearn.pipeline.Pipeline): Modèle pour transformer les
        données.

    Returns:
        pd.DataFrame: Données transformées.
    """

    # Ouvrir le fichier zip en mode lecture ('r')
    with zipfile.ZipFile(file_path, 'r') as z:
        with z.open(file_name_csv) as f:
            cols = pd.read_csv(f, nrows=0).columns

    # Supprimer la première colonne
    cols = cols[1:]

    # Lire le fichier CSV sans la première colonne
    data_df = pd.read_csv(file_path, usecols=cols)

    # Isolement de la colonne SK_ID_CURR
    sk_id_curr = data_df['SK_ID_CURR']

    # Suppression des colonnes TARGET et SK_ID_CURR
    data_df_dropped = data_df.drop(columns=["TARGET", "SK_ID_CURR"])

    # Imputation des valeurs manquantes (preprocessing du modele)
    data_array = _model.named_steps['preprocess'].transform(data_df_dropped)

    # Création d'un DataFrame à partir du tableau numpy
    data_df_new = pd.DataFrame(data_array, columns=data_df_dropped.columns)

    # Re-insertion de la colonne SK_ID_CURR
    data_df_new['SK_ID_CURR'] = sk_id_curr

    return data_df_new

# Chargement des données
data = load_data(DATA_PATH, 'application_train_cleaned_frac_1%.csv', model)
print('chargement des données terminé\n')

# ====================== étape 5 : Fonctions ============================

def get_client_data(client_id):
    """
    Récupère les données d'un client spécifique.

    Args:
        client_id (int): ID du client.

    Returns:
        pd.DataFrame: Données du client.
    """
    # Isoler les données du client et compter Nan
    client_data = data[data['SK_ID_CURR'] == client_id]
    nan_client = client_data.isna().sum().sum()

    # Afficher les données du client
    st.dataframe(client_data)
    st.write(f'Nombre NaN du client {client_id} : {nan_client}')

    return client_data.drop(columns=['SK_ID_CURR'])


def request_prediction(url, data):
    """
    Envoie une requête POST de prédiction à un service web.

    Args:
        url (str): URL du service web.
        data (pd.DataFrame): Données à prédire.

    Returns:
        dict: Réponse du service web.
    """
    # Envoi de la requête POST
    response = requests.post(url, json=data.to_dict(orient='records'))

    # Vérification de la réponse
    response.raise_for_status()

    return response.json()


def display_or_save_plot(shap_values_all, data, FIG_PATH):
    """
    Affiche ou sauvegarde le plot de feature importance globale.

    Args:
        shap_values_all (np.array): SHAP values pour toutes les données.
        data (pd.DataFrame): Données.
        FIG_PATH (str): Chemin du répertoire pour sauvegarder le plot.
    """
    # Chemin de l'image de feature importance globale
    image_path = os.path.join(FIG_PATH, 'feature_importance_globale.png')

    # SI l'image existe ALORS on l'affiche
    if os.path.isfile(image_path):
        st.image(image_path)

    # SINON Enregistrer et afficher le plot
    else:
        # Affichage feature importance globale
        shap.summary_plot(
            shap_values_all,
            data.drop(columns=['SK_ID_CURR']),
            plot_type='dot'
        )
        plt.savefig(image_path)
        st.pyplot()



# ============= étape 6 : Fonction principale du dashboard ==================

@profile
def main():
    """
    Fonction principale de l'application Streamlit.
    """
    # Titre de la section
    client_id = st.selectbox('Sélection client :', data['SK_ID_CURR'].unique())

    # Récupération des données du client
    client_data = get_client_data(client_id)

    # Bouton pour calculer la prédiction => envoi de la requête POST
    if st.button('Calculer la prédiction'):
        response = request_prediction(URL_API_PREDICT, client_data)

        # Affichage de la prédiction en français
        if response["prediction"]["prediction"] == 0:
            st.markdown(
                '<div style="background-color: #98FB98; padding: 10px;'
                'border-radius: 5px; color: #000000;"'
                '>Le prêt est accordé.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="background-color: #FF6347; padding: 10px;'
                'border-radius: 5px; color: #000000;"'
                '>Le prêt n\'est pas accordé.</div>',
                unsafe_allow_html=True
            )

        # ajouter un espace
        st.write('')

        # Affichage de la prédiction
        st.dataframe(response['prediction'])

        # Transformation des données pour SHAP en array
        shap_values_subset_array = np.array(
            response['feature_importance_locale']['shap_values_subset']
        )

        # Transformation des données client en DataFrame
        client_data_subset_df = pd.DataFrame(
            [response['feature_importance_locale']['client_data_subset']],
            columns=response['feature_importance_locale']['top_features']
        )

        # Affichage feature importance locale
        st.write('Feature importance locale :')
        shap.force_plot(
            response['prediction']["explainer"],
            shap_values_subset_array,
            client_data_subset_df,
            matplotlib=True
        )
        st.pyplot()

        # # Calcul des SHAP values pour toutes les données
        explainer = shap.TreeExplainer(model[-1])
        data_dropped = data.drop(columns=['SK_ID_CURR'])
        shap_values_all = explainer.shap_values(
            data_dropped,
            check_additivity=False
        )[1]

        # Affichage ou sauvegarde du plot de feature importance globale
        st.write('Feature importance globale :')
        display_or_save_plot(shap_values_all, data, FIG_DIR)

# =================== étape 7 : Run du dashboard ==========================

if __name__ == '__main__':
    main()
