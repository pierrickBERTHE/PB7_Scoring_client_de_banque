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

# Ne pas afficher les warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# ====================== étape 2 : Généralités ============================

# Titre de l'application
st.title('Projet 7\n')
st.title('Élaborez le modèle de scoring - Dashboard\n')

# Choix du répertoire racine (local ou distant)
environment = os.getenv('ENVIRONMENT', 'distant')
print(f'Environnement : {environment}\n')
print("getcwd:", os.getcwd(), "\n")
# print("listdir:", os.listdir(), "\n")

if environment == 'local':
    ROOT_DIR = "C:\\Users\\pierr\\VSC_Projects\\Projet7_OCR_DataScientist"
    MODEL_URL_FLASK = 'http://127.0.0.1:5000/predict'

else:
    ROOT_DIR = os.path.join(
        os.getcwd(), "Berthe_Pierrick_4_dossier_code_022024"
        )
    MODEL_URL_FLASK = 'http://pierrickberthe.eu.pythonanywhere.com/predict'

# # Chemin du fichier de données nettoyées
# DATA_PATH = os.path.join(
#     ROOT_DIR, "data", "cleaned", "application_train_cleaned.csv"
# )

# chemin du répertoire pour sauvegarder le plot
FIG_PATH = os.path.join(ROOT_DIR, "figure")

# Chemin du modèle pré-entraîné
MODEL_PATH = os.path.join(ROOT_DIR, 'mlflow_model', 'model.pkl')

# Chargement du modèle pré-entraîné
model = joblib.load(MODEL_PATH)

# ==================== étape 3 : chargement data ==========================

# DEV debut

# @st.cache_data
# def load_data(file_path, _model):
#     """
#     Charge les données à partir d'un fichier CSV et les transforme en
#     utilisant un modèle donné.

#     Args:
#         file_path (str): Chemin vers le fichier CSV.
#         _model (imblearn.pipeline.Pipeline): Modèle pour transformer les
#         données.

#     Returns:
#         pd.DataFrame: Données transformées.
#     """
#     # Lire les noms de colonnes
#     cols = pd.read_csv(file_path, nrows=0).columns

#     # Supprimer la première colonne
#     cols = cols[1:]

#     # Lire le fichier CSV sans la première colonne
#     data_df = pd.read_csv(file_path, usecols=cols)

#     # Isolement de la colonne SK_ID_CURR
#     sk_id_curr = data_df['SK_ID_CURR']

#     # Suppression des colonnes TARGET et SK_ID_CURR
#     data_df_dropped = data_df.drop(columns=["TARGET", "SK_ID_CURR"])

#     # Imputation des valeurs manquantes (preprocessing du modele)
#     data_array = _model.named_steps['preprocess'].transform(data_df_dropped)

#     # Création d'un DataFrame à partir du tableau numpy
#     data_df_new = pd.DataFrame(data_array, columns=data_df_dropped.columns)

#     # Re-insertion de la colonne SK_ID_CURR
#     data_df_new['SK_ID_CURR'] = sk_id_curr

#     return data_df_new

# # Chargement des données
# data = load_data(DATA_PATH, model)

DATA_URL_FLASK = 'http://pierrickberthe.eu.pythonanywhere.com/load_data'

# Faire une requête GET à l'API Flask
response = requests.get(DATA_URL_FLASK)

# Vérifier que la requête a réussi
if response.status_code == 200:

    # Convertir le texte de la réponse en un DataFrame pandas
    data = pd.DataFrame(response.json())

    # Utiliser les données dans votre application Streamlit
    st.write("Données chargées avec succès !")
else:
    st.write(f"Échec du chargement des données : {response.status_code}")

# DEV fin

# ====================== étape 4 : Fonctions ============================

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

# ============= étape 5 : Fonction principale du dashboard ==================

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
        response = request_prediction(MODEL_URL_FLASK, client_data)

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
        shap_values_subset_array = np.array(response['feature_importance_locale']['shap_values_subset'])

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
        display_or_save_plot(shap_values_all, data, FIG_PATH)

# =================== étape 6 : Run du dashboard ==========================

if __name__ == '__main__':
    main()
