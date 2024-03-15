import streamlit as st
import pandas as pd
import numpy as np
import requests
import mlflow
import os
import joblib

st.title('Dashboard - Scoring crédit')

# Récupération du chemin absolu et du répertoire du fichier api.py
chemin_racine = "C:\\Users\\pierr\\VSC_Projects\\Projet7_OCR_DataScientist"

# Chargement des données
data_train_path = os.path.join(chemin_racine, "data", "cleaned", "application_train_cleaned.csv")

# Chargement du modèle pré-entraîné
MODEL_PATH = os.path.join(chemin_racine, 'mlflow_model', 'model.pkl')
model = joblib.load(MODEL_PATH)

MODEL_URL_FLASK = 'http://127.0.0.1:5000/predict'

# Load data
@st.cache_data
def load_data(nom_fichier):
    """Charger les données à partir d'un fichier CSV."""
    return pd.read_csv(nom_fichier)

data = load_data(data_train_path)

def get_data_client(id_client):
    # filter on the choosen client 
    data_client = data[data['SK_ID_CURR'] == id_client]
    return data_client

def request_prediction(MODEL_URL_FLASK, data):
    headers = {"Content-Type": "application/json"}

    # Convertir le DataFrame en liste de dictionnaires
    data_json = data.to_dict(orient='records')

    # use the mode POST on the MODEL URI  
    response = requests.post(MODEL_URL_FLASK, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

def main():

    id_client = st.number_input('Id Client', value=100002)
    data_client = get_data_client(id_client)
    
    st.dataframe(data_client)
    
    predict_btn = st.button('Prédire')

    if predict_btn:
        response = request_prediction(MODEL_URL_FLASK, data_client)
        
        if 'prediction' in response:
            st.write(response['prediction'])

            if int(response['prediction']) == 0 :
                st.write('Accordé')
            else :
                st.write('Refusé')
        else:
            st.write("No prediction in the response")

if __name__ == '__main__':
    main()


# # pour appeler le dashboard : 
# streamlit run Berthe_Pierrick_4_dossier_code_022024/scripts/dashboard.py