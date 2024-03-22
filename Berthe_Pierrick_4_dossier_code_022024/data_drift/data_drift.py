"""
Description: Ce fichier permet de comparer les données de référence et les données actuelles (analyse du data drift) avec la librairie Evidently.

Author: Pierrick Berthe
Date: 2024-03-22
"""

# ============== étape 1 : Importation des librairies ====================

import os
import pandas as pd
import numpy as np
import time
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping

# ====================== étape 2 : Généralités ============================

# Chemin du répertoire racine
ROOT_DIR = "C:\\Users\\pierr\\VSC_Projects\\Projet7_OCR_DataScientist"

# Chemin du fichiers des données de reference (application_train)
DATA_REF_PATH = os.path.join(
    ROOT_DIR, "data", "cleaned", "application_train_cleaned.csv"
)

# Chemin du fichier des données nouvelles (application_test)
DATA_NEW_PATH = os.path.join(
    ROOT_DIR, "data", "cleaned", "application_test_cleaned.csv"
)

# ====================== étape 3 : Fonctions ============================

def load_data(path, drop_columns):
    """
    Charge les données à partir d'un fichier CSV et supprime les colonnes
    spécifiées.
    """
    data = pd.read_csv(path)
    data.drop(columns=drop_columns, inplace=True)
    return data


def get_categorical_columns(data):
    """
    Récupère les colonnes catégorielles d'un DataFrame (col avec des valeurs
    uniques de 0, 1 et NaN).
    """
    categorical_col = [
        col for col in data.columns
        if set(data[col].unique()).issubset({0, 1, np.nan})
    ]
    return categorical_col


def get_numerical_columns(data, categorical_col):
    """
    Récupère les colonnes numériques d'un DataFrame (toutes les colonnes sauf
    les colonnes catégorielles).
    """
    all_columns = set(data.columns)
    numerical_columns = list(all_columns - set(categorical_col))
    return numerical_columns


def create_data_drift_report(
    numerical_columns,
    categorical_col,
    threshold=0.05
):
    """
    Crée un rapport de data drift avec les colonnes numériques et 
    catégorielles.
    """
    # Création du column mapping (selon cat et num)
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = numerical_columns
    column_mapping.categorical_features = categorical_col

    # Création du rapport de data drift
    data_drift_rapport = Report(
        metrics=[DataDriftPreset(
            num_stattest='ks',
            cat_stattest='psi',
            num_stattest_threshold=threshold,
            cat_stattest_threshold=threshold
        )]
    )
    return data_drift_rapport, column_mapping

# =================== étape 4 : Fonction principale =====================

def main():
    """
    Fonction principale pour comparer les données de référence et les données
    actuelles (analyse du data drift).
    """
    # Chargement des données
    drop_columns = ["Unnamed: 0", "TARGET", "SK_ID_CURR"]
    data_ref = load_data(DATA_REF_PATH, drop_columns)
    data_new = load_data(DATA_NEW_PATH, drop_columns)

    # Afficher les dimensions des données
    print("Dimensions des données :")
    print("data_ref shape: ", data_ref.shape)
    print("data_new shape: ", data_new.shape)
    print("")


############################## DEV ########################################

    # Réduction de la taille des données
    data_ref = data_ref.sample(frac=0.01, random_state=1)
    data_new = data_new.sample(frac=0.01, random_state=1)

    print("Echantillon de 1% des données obtenu :")
    print("data_ref shape: ", data_ref.shape)
    print("data_new shape: ", data_new.shape)
    print("")

############################## DEV ########################################

    # Isolement des noms des colonnes numériques et catégorielles
    categorical_col = get_categorical_columns(data_ref)
    numerical_columns = get_numerical_columns(data_ref, categorical_col)

    # Vérification de la similarité des colonnes des 2 DataFrames
    assert set(data_ref.columns) == set(data_new.columns)
    print("Les 2 DataFrames ont les mêmes colonnes")
    print("")

    # Création du rapport de data drift
    start_time = time.time()
    data_drift_rapport, column_mapping = create_data_drift_report(
        numerical_columns,
        categorical_col,
        threshold=0.05
    )

    # Durée d'exécution de la création du rapport
    minutes, seconds = divmod(time.time() - start_time, 60)
    print(f"Durée data_drift_report : {int(minutes)} min {int(seconds)} sec")

    # Run du rapport
    start_time = time.time()
    data_drift_rapport.run(
        reference_data=data_ref,
        current_data=data_new,
        column_mapping=column_mapping
    )

    # Durée d'exécution du run
    minutes, seconds = divmod(time.time() - start_time, 60)
    print(f"Durée Run rapport : {int(minutes)} min {int(seconds)} sec")

    # Sauvegarde du rapport
    data_drift_rapport.save_html(os.path.join(
        ROOT_DIR,
        "Berthe_Pierrick_5_Tableau_HTML_data_drift_evidently_022024.html"
    ))

# =========== étape 5 : Exécution de la fonction principale ===============
if __name__ == "__main__":
    main()
