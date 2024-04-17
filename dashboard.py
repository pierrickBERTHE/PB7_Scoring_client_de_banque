"""
Description: Ce fichier contient le dashboard de l'application Streamlit
basique.

Author: Pierrick Berthe
Date: 2024-04-04
"""

# ============== étape 1 : Importation des librairies ====================

import pandas as pd
import requests
import os
import streamlit as st
from memory_profiler import profile
import json

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

# ====================== étape 3 : Fonctions ============================

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


# ============= étape 4 : Fonction principale du dashboard ==================

@profile
def main():
    """
    Fonction principale de l'application Streamlit.
    """
    # Récupération des clients et sélection d'un client
    client_id = fetch_data_and_client_selection(URL_API_CLIENT_SELECTION)
    
    # Récupération des données du client
    client_data = get_client_data(URL_API_CLIENT_EXTRACTION, client_id)

# =================== étape 5 : Run du dashboard ==========================

if __name__ == '__main__':
    main()
