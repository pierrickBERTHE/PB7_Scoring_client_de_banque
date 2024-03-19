import streamlit as st
import pandas as pd
import numpy as np
import requests
import mlflow
import os
import joblib
import shap
import matplotlib.pyplot as plt


# Titre de l'application
st.title('Dashboard - Scoring crédit')

# Récupération du chemin absolu et du répertoire du fichier api.py
chemin_racine = "C:\\Users\\pierr\\VSC_Projects\\Projet7_OCR_DataScientist"

# Chargement des données
data_train_path = os.path.join(
    chemin_racine, "data", "cleaned", "application_train_cleaned.csv"
)

# Chargement du modèle pré-entraîné
MODEL_PATH = os.path.join(chemin_racine, 'mlflow_model', 'model.pkl')
model = joblib.load(MODEL_PATH)

# URL du predict de l'API Flask
MODEL_URL_FLASK = 'http://127.0.0.1:5000/predict'

# Décorateur pour mettre en cache les données
@st.cache_data
def load_data(nom_fichier, _model):
    """Charger les données à partir d'un fichier CSV et appliquer le prétraitement."""
    # Lire les noms de colonnes
    cols = pd.read_csv(nom_fichier, nrows=0).columns

    # Supprimer la première colonne
    cols = cols[1:]

    # Lire le fichier CSV sans la première colonne
    data = pd.read_csv(nom_fichier, usecols=cols)

    # Isoler la colonne SK_ID_CURR et TARGET
    sk_id_curr = data['SK_ID_CURR']
    target = data['TARGET']

    # Supprimer ces 2 cols
    data = data.drop(columns=["TARGET", "SK_ID_CURR"])
    
    # Conserver les noms de colonnes
    column_names = data.columns

    # Appliquer les étapes de prétraitement (imputation + normalisation)
    preprocessed_data = model.named_steps['preprocess'].transform(data)

    # Recréer le DataFrame avec les noms de colonnes
    preprocessed_data = pd.DataFrame(preprocessed_data, columns=column_names)

    # Remettre col SK_ID_CURR
    preprocessed_data['SK_ID_CURR'] = sk_id_curr
    
    return preprocessed_data

# Chargement des données (avec preprocessing imputation du modèle)
data = load_data(data_train_path, _model=model)


def get_data_client(client_id):
    # Obtenir les données pour le SK_ID_CURR sélectionné
    client_data = data[data['SK_ID_CURR'] == client_id]
    st.dataframe(client_data)

    # Compter et afficher le nombre de valeurs manquantes
    nan_nbr = client_data.isna().sum().sum()
    st.write(f'Nombre NaN pour client {client_id} : {nan_nbr}')

    # Supprimer la col SK_ID_CURR
    client_data = client_data.drop(columns=['SK_ID_CURR'])

    return client_data


def request_prediction(MODEL_URL_FLASK, data):
    headers = {"Content-Type": "application/json"}

    # Convertir le DataFrame en liste de dictionnaires
    data_json = data.to_dict(orient='records')

    # Faire une requête POST avec les données JSON
    response = requests.post(MODEL_URL_FLASK, json=data_json)

    # Vérifier si la requête a réussi
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(
                response.status_code,
                response.text
            )
        )

    # Extraire le JSON de la réponse
    response_json = response.json()

    # Vérifier si 'error' est dans la réponse
    if 'error' in response_json:
        raise Exception(response_json['error'])

    return response_json


# def get_feature_importance_locale(response, client_data):

#     # Nombre de features à afficher
#     nbr_feature=5

#     shap_values_df = response['feature_importance'].pop('feature_values', None)

#     # # Convertir la liste en DataFrame
#     # shap_values_df = pd.DataFrame(
#     #     response["feature_importance"]["shap_values"],
#     #     columns=response["feature_importance"]["feature_names"]
#     # )

#     # Calculer la valeur absolue des valeurs SHAP
#     abs_shap_values = shap_values_df.abs()

#     # Obtenir les noms des nbr_feature ayant les valeurs SHAP les plus élevées
#     top_features = (
#         abs_shap_values.sum()
#         .sort_values(ascending=False)
#         .head(nbr_feature)
#         .index
#     )

#     # Sélectionner les top_features de X_test et shap_values_df (arrondis)
#     client_data_subset = client_data[top_features].round(2)
#     shap_values_subset = response["feature_importance"]["shap_values"][top_features].values.round(2)[0]

#     return shap_values_subset, client_data_subset


def main():

    # Créer un widget de sélection pour choisir un SK_ID_CURR
    client_id = st.selectbox(
        'Choisissez un SK_ID_CURR',
        data['SK_ID_CURR'].unique()
    )

    # Isoler les données du client sélectionné
    client_data = get_data_client(client_id)

    # Ajouter un bouton pour déclencher la prédiction
    if st.button('Calculer la prédiction'):

        # Utiliser le modèle pour faire une prédiction
        response = request_prediction(MODEL_URL_FLASK, client_data)

        # Afficher la prédiction
        if int(response['prediction']['prediction']) == 0:
            st.markdown(
                '<div style="background-color: #98FB98; padding: 10px;'
                'border-radius: 5px; color: #000000;"'
                '>Le prêt est accordé.</div>',
                unsafe_allow_html=True
            )

        elif int(response['prediction']['prediction']) == 1:
            st.markdown(
                '<div style="background-color: #FF6347; padding: 10px;'
                'border-radius: 5px; color: #000000;"'
                '>Le prêt n\'est pas accordé.</div>',
                unsafe_allow_html=True
            )

        else:
            st.write("Pas de prédiction effectuée.")

        # Afficher la probabilité
        st.dataframe(response['prediction'])
        st.dataframe(response['feature_importance'])
        st.dataframe(response['feature_importance_locale'])

        # #
        # shap_values_subset, client_data_subset = get_feature_importance_locale(response, client_data)

        # transforme shap_values_subset en array
        shap_values_subset_array = np.array(
            response['feature_importance_locale']['shap_values_subset']
        )
        st.write('shap_values_subset_array : ')
        st.dataframe(shap_values_subset_array)
        st.write(
            'len(shap_values_subset_array) : ', len(shap_values_subset_array)
        )

        # st.write(
        #     "len(response['feature_importance_locale']['client_data_subset'][0]) : ", len(response['feature_importance_locale']['client_data_subset'][0])
        # )
        # st.write(
        #     "len(response['feature_importance_locale']['top_features']) : ", len(response['feature_importance_locale']['top_features'])
        # )

        # Transforme client_data_subset en DataFrame
        client_data_subset_df = pd.DataFrame(
            [response['feature_importance_locale']['client_data_subset']],
            columns=response['feature_importance_locale']['top_features']
        )
        st.write('client_data_subset_df : ')
        st.dataframe(client_data_subset_df)
        st.write(
            'shap_values_subset_array.shape : ', shap_values_subset_array.shape
        )

        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Générer le force_plot
        force_plot = shap.force_plot(
            response['prediction']["explainer"],
            shap_values_subset_array,
            client_data_subset_df,
            matplotlib=True
        )
        st.pyplot(force_plot)

        data_sans_sk_id_curr = data.drop(columns=['SK_ID_CURR'])
        st.write('data_sans_sk_id_curr : ')
        st.dataframe(data_sans_sk_id_curr.head())
        st.write(
            'data_sans_sk_id_curr.shape : ', data_sans_sk_id_curr.shape
        )

        # Extraire le dernier estimateur du pipeline
        final_estimator = model[-1]

        # Expliquer les prédictions à l'aide de SHAP et son TreeExplainer
        explainer = shap.TreeExplainer(final_estimator)
        shap_values_all = explainer.shap_values(data_sans_sk_id_curr, check_additivity=False)

        # Pour la deuxième sortie
        shap.summary_plot(shap_values_all[1], data_sans_sk_id_curr, plot_type='dot')
        st.pyplot()


if __name__ == '__main__':
    main()


# # pour appeler le dashboard : 
# streamlit run Berthe_Pierrick_4_dossier_code_022024/scripts/dashboard.py
