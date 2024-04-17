"""
Description: Ce fichier contient le dashboard de l'application Streamlit.

Author: Pierrick Berthe
Date: 2024-04-04
"""

# ============== étape 1 : Importation des librairies ====================

import pandas as pd
import numpy as np
import requests
import os
import shap
import matplotlib.pyplot as plt
import streamlit as st
from memory_profiler import profile
import socket
import json
from PIL import Image
from io import BytesIO

# ================= étape 2 : Chemins environnement ========================

# Indicateur pour savoir si l'API est sur le cloud ou en local
IS_API_ON_CLOUD = True

# Titre de l'application
st.title('Projet 7\n')
st.title('Élaborez le modèle de scoring - Dashboard\n')

# Affichage le chemin du répertoire courant
print("os.getcwd():",os.getcwd(), "\n")

# URL de l'API Flask (local ou distant)
if IS_API_ON_CLOUD:
    URL_API = 'http://pierrickberthe.eu.pythonanywhere.com'
else:
    URL_API = 'http://127.0.0.1:6000'
print("URL_API:",URL_API, "\n")

# URL de l'API pour les requêtes POST
URL_API_CLIENT_SELECTION = f'{URL_API}/client_selection'
URL_API_CLIENT_EXTRACTION= f'{URL_API}/client_extraction'
URL_API_PREDICT = f'{URL_API}/predict'
URL_API_FI_GLOBALE= f'{URL_API}/feature_importance_globale'

# ====================== étape 5 : Fonctions ============================

def fetch_data_and_client_selection(url):
    """
    Récupère les données et crée une liste déroulante pour sélectionner
    un client.
    """
    # Envoi de la requête POST
    response = requests.post(url)

    # SI requete OK => extraction, transformation et affichage des données
    if response.status_code == 200:
        sk_id_curr_all = response.json()
        sk_id_curr_all = pd.Series(sk_id_curr_all)
        client_id = st.selectbox('Sélection client :', sk_id_curr_all.unique())
        return int(client_id)
    else:
        st.write("Erreur lors de la récupération des id des clients.")
        return None


def get_client_data(url, client_id):
    """
    Récupère les données d'un client spécifique.
    """
    # Envoi de la requête POST
    response = requests.post(url, json={'client_id': client_id})

    # SI requete OK => load json, verif type str, transformation en dataframe
    if response.status_code == 200:
        response_dict = json.loads(response.text)
        if isinstance(response_dict["client_data"], str):
            client_data = pd.read_json(
                response_dict["client_data"],
                orient='records'
            )

            # Affichage des données du client
            st.dataframe(client_data)
            info_client = (f'Nombre NaN du client {client_id} : '
                f'{response_dict["nan_client"]}')
            st.write(info_client)

            # Vérification que 'SK_ID_CURR' est une colonne du DataFrame
            if 'SK_ID_CURR' in client_data.columns:
                return client_data.drop(columns=['SK_ID_CURR'])
            else:
                st.write("La colonne 'SK_ID_CURR' n'est pas dans le df.")
                return client_data

        else:
            st.write("'client_data' n'est pas une chaîne JSON.")
            return None

    else:
        st.write("Erreur lors de la récupération des données du client.")
        return None


@st.cache_resource
def request_prediction(url, data):
    """
    Envoie une requête POST de prédiction à un service web.
    """
    # ESSAI de la requête POST (timeout de 600 secondes)
    try:
        response = requests.post(url, json=data.to_dict(), timeout=300)
        response.raise_for_status()
        return response.json()

    # Gestion des autres erreurs
    except Exception as err:
        print(f'Erreur dans la requête POST de prediction: {err}')
        return None


def calcule_fi_globale(url):
    """
    Récupère le calcul de la feature importance globale et affiche le 
    summary plot en image. 
    """
    # Envoi de la requête POST
    response = requests.post(url)

    # SI requete OK => extraction, ouverture image et affichage
    if response.status_code == 200:
        image_bytes = BytesIO(response.content)
        img = Image.open(image_bytes)
        st.image(img)

    else:
        print("Erreur lors de la récupération de l'image.")

# ============= étape 6 : Fonction principale du dashboard ==================

@profile
def main():
    """
    Fonction principale de l'application Streamlit.
    """
    # Récupération des clients et sélection d'un client
    client_id = fetch_data_and_client_selection(
        URL_API_CLIENT_SELECTION
    )
    
    # Récupération des données du client
    client_data = get_client_data(URL_API_CLIENT_EXTRACTION, client_id)

    # Bouton pour calculer la prédiction => envoi de la requête POST
    if st.button('Calculer la prédiction'):
        response = request_prediction(URL_API_PREDICT, client_data)

        # Affichage d'une erreur si la prédiction est None
        if response is None:
            st.write('Erreur de prédiction\n')

        # Affichage de la prédiction en français
        else:
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
            ligne_prediction = {
                'explainer': response['prediction']['explainer'][1],
                'prediction': response['prediction']['prediction'],
                'probabilité': response['prediction']['probabilité']
            }
            st.dataframe(ligne_prediction)

            # Transformation des données pour SHAP en array puis en dataframe
            shap_values_subset_array = np.array(
                response['feature_importance_locale']['shap_values_subset']
            )
            client_data_subset_df = pd.DataFrame(
                [response['feature_importance_locale']['client_data_subset']],
                columns=response['feature_importance_locale']['top_features']
            )

            # Affichage feature importance locale
            st.write('Feature importance locale :')
            shap.force_plot(
                response['prediction']["explainer"][1],
                shap_values_subset_array,
                client_data_subset_df,
                matplotlib=True
            )

            # Obtention de la figure actuelle et affichage streamlit
            fig = plt.gcf()
            st.pyplot(fig)

            # Calcule et affichage de feature importance globale
            st.write('Feature importance globale :')
            calcule_fi_globale(URL_API_FI_GLOBALE)

# =================== étape 7 : Run du dashboard ==========================

if __name__ == '__main__':
    main()
