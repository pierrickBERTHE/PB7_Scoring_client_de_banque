import streamlit as st
import pandas as pd
import numpy as np
import requests
import os

st.title('Dashboard - Scoring crédit')

DATA_URL = 'Data/train.csv'

MODEL_URL_FLASK = ' ' # the link of the API flask


@st.cache_data
def load_data(nom_fichier):
    """Charger les données à partir d'un fichier CSV."""
    return pd.read_csv(nom_fichier)

# Décorateur pour mettre en cache le modèle
@st.cache_resource
def load_model(model_uri):
    return mlflow.pyfunc.load_model(model_uri)

def get_data_client(id_client):
    # filter on the choosen client 
    # TO FILL
    return data_client

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'dataframe_split' : data.to_dict(orient='split')}
    
# TO FILL
# use the mode POST on the MODEL URI  
    response = requests.request(.....))

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


def main():

    id_client = st.number_input('Id Client', value=0)
    data = get_data_client(id_client)
    
    st.dataframe(data)
    
    predict_btn = st.button('Prédire')

    if predict_btn:
        response = request_prediction(MODEL_URL_FLASK, data)
        
        st.write(response['prediction'])

        if int(response['prediction']) == '0' :
            st.write('Accordé')
        else :
            st.write('refusé')


if __name__ == '__main__':
    main()
