"""
Description: Ce fichier permet de créer un fichier fichier sous-echantillionné
avec 10% des données du dataset initial

Author: Pierrick Berthe
Date: 2024-04-02
"""

# ============== étape 1 : Importation des librairies ====================

import os
import pandas as pd
import random
import zipfile

# ============== étape 2 : Chemins environnement ========================

# Détermination de la graine (fixer aléatoire)
random.seed(42)

# Répertoire racine
ROOT_DIR = os.getcwd()
print("ROOT_DIR:", ROOT_DIR, "\n")

# Chemin des données
DATA_DIR= os.path.join(ROOT_DIR, "data/cleaned")
DATA_PATH = os.path.join(DATA_DIR, "application_train_cleaned.zip")
print("DATA_PATH:", DATA_PATH, "\n")

# ================= étape 3 : Fonctions ========================

def load_data(zip_path, csv_file):
    """
    Charge un DataFrame à partir d'un fichier CSV dans une archive ZIP.
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(csv_file) as f:
            df = pd.read_csv(f)
    return df


def sample_data(df, fraction):
    """
    Échantillonne une fraction d'un DataFrame.
    """
    return df.sample(frac=fraction)


def save_data(df, DATA_DIR, output_file):
    """
    Enregistre un DataFrame dans un fichier CSV.
    """
    output_path = os.path.join(DATA_DIR, output_file)
    df.to_csv(output_path, index=False)


def main():
    """
    Fonction principale qui orchestre le chargement, l'échantillonnage et
    l'enregistrement des données.
    """
    frac = 0.01
    df = load_data(DATA_PATH, "application_train_cleaned.csv")
    sample_df = sample_data(df, frac)
    save_data(
        sample_df,
        DATA_DIR,
        'application_train_cleaned_frac_{0}%.csv'.format(frac * 100)
    )
    print("Fichier sous-échantillionné exporté")

# ============== étape 4 : Exécution ====================

if __name__ == "__main__":
    main()
