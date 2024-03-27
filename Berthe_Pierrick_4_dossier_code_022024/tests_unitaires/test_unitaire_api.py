"""
Description: Ce fichier contient les tests unitaires pour l'API Flask.

Author: Pierrick Berthe
Date: 2024-03-21
"""

# ============== étape 1 : Importation des librairies ====================

import os
import sys
import unittest
import requests
import json
import pandas as pd

# Ajouter le chemin relatif du fichier api.py au sys.path pour import
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ".."))
)

from Berthe_Pierrick_1_API_022024 import app
print("Berthe_Pierrick_1_API_022024 chargé\n")

# ========== étape 2 :Class de test pour l'application Flask =============

class TestFlaskApp(unittest.TestCase):
    """
    Classe de test pour l'application Flask.
    """

    @classmethod
    def setUpClass(cls):
        """
        Cette méthode est appelée une fois pour toute la classe de test.
        Elle charge les données du premier client.
        """

        # Choix du répertoire racine (local ou distant)
        environment = os.getenv('ENVIRONMENT', 'distant')

        if environment == 'local':
            DATA_PATH = os.path.join(
                os.getcwd(), "data", "cleaned", "application_train_cleaned.csv"
            )
        else:
            DATA_PATH = os.path.join(
                os.getcwd(), "..", "..", "data", "cleaned", "application_train_cleaned.csv"
            )

        # Charge le fichier CSV dans un DataFrame pandas
        df = pd.read_csv(DATA_PATH)

        # Selectionne le premier client_id pour la prédiction
        client_id = int(df.iloc[0]['SK_ID_CURR'])

        # Isoler les données du client
        cls.client_data = df[df['SK_ID_CURR'] == client_id]

        # Supprimer les colonnes inutiles pour la prédiction
        cls.client_data = cls.client_data.drop(columns=[
            'Unnamed: 0',
            'SK_ID_CURR',
            'TARGET']
        )

    def setUp(self):
        """
        Cette méthode est appelée avant chaque test. Elle crée un client de
        test pour l'application Flask et active le mode de test.
        """
        self.app = app.test_client()
        self.app.testing = True


    def test_a_route_acceuil(self):
        """
        Teste la route d'accueil de l'application Flask.
        """
        # Envoyer une requête GET à la route d'accueil (index)
        response = self.app.get('/')

        # Vérifier que la réponse a un code de statut 200 (OK)
        self.assertEqual(response.status_code, 200, "pas code_status 200")

        # Vérifier que la réponse contient le texte attendu
        self.assertIn("Bienvenue sur l'API de Pierrick", response.data.decode())
        print("\n")


    def test_b_prediction_route(self):
        """
        Teste la route POST de la prediction de l'application Flask.
        """
        # Envoyer une requête POST à la route '/predict'
        response = self.app.post(
            '/predict',
            json=self.client_data.to_dict(orient='records')
        )

        # Vérifier que la réponse a un code de statut 200 (OK)
        self.assertEqual(response.status_code, 200, "pas code_status 200")
        print("\n")


    def test_c_prediction_result(self):
        """
        Teste la prédiction de l'API en utilisant les données du premier client.
        """

        # Envoi de la requête POST
        with app.test_client() as client:
            response = client.post(
                '/predict',
                json=self.client_data.to_dict(orient='records')
            )

            # Vérifier que la requête a réussi
            assert response.status_code == 200, "La requête a échoué."

            # Extraire la prédiction de la réponse
            prediction = json.loads(response.data)["prediction"]["prediction"]

            # Vérification que la prédiction a été effectuée correctement
            assert prediction is not None, "La prédiction a échoué."
            print("\n")


    def test_d_error_status(self):
        """
        Teste la réponse de l'API en cas d'erreur 415.
        """
        # Envoi d'une requête POST sans données
        response = self.app.post('/predict')

        # Vérifier que le code de statut est 415 (Unsupported Media Type)
        self.assertEqual(response.status_code, 415, "pas erreur 415")

# =============== étape 3 : Run des tests unitaires =======================

if __name__ == "__main__":
    unittest.main()
