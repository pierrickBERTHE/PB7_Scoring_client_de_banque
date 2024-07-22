# <span style='background:blue'>Contexte</span>

L'entreprise **"Prêt à dépenser"** souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un **algorithme de classification** en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).


# <span style='background:blue'>Missions</span>

**Automatiser la prise de décision d’accord de prêt grâce à un algorithme de classification**

1/ Construire le **modèle de scoring**

2/ **Analyser les features** ayant le plus d’impact sur le scoring de manière générale et au niveau d’un client

3/ **Mettre en production** le modèle de scoring dans une **API**

4/ Mettre en œuvre une approche globale **MLOps** de bout en bout (tracking expérimentation => data drift)


# <span style='background:blue'>Dataset</span>

Home Credit est une institution financière internationale de prêts à la consommation. Elle nous fournit un jeu de données comportant des informations sur les clients, les crédits qu'ils ont contractés, leur revenus, etc. Ces données sont utilisées pour construire un modèle de scoring de crédit pour prédire la probabilité de capacité de remboursement d'un client, et donc de déterminer si un crédit doit lui être accordé ou non.

Source : [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/overview) sur Kaggle.com<br>

10 fichiers CSV :
- application_train.csv
- application_test.csv
- bureau.csv
- bureau_balance.csv
- credit_card_balance.csv
- HomeCredit_columns_description.csv
- installments_payments.csv
- POS_CASH_balance.csv
- previous_application.csv
- sample_submission.csv

Voici le diagramme entité-association des données (diagramme ERD) :

![mappage_dataset](https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png)


# <span style='background:blue'>Fichiers du dépôt</span>

- Dossier **Data drift** : Notebook, script python et rapport HTML pour le suivi du drift de données

- Dossier **data_heroku** : Dataset pour le déploiement de l'API sur Heroku

- Dossier **Berthe_Pierrick_4_dossier_code_022024** : dossier comportant les fichiers suivants :
    - data : dossier contenant les datasets
    - mlflow_model : dossier contenant les modèles MLFlow
    - notebooks : dossier contenant les notebooks de nettoyage et de modélisation
    - tests_unitaires : dossier contenant les tests unitaires
    - Fichier explicatif dossier Github : PDF expliquant le contenu du dossier

- **Procfile** : fichier pour le déploiement de l'API sur Heroku
- 
- **runtime.txt** : fichier pour le déploiement de l'API sur Heroku
- 
- **api.py** : Script python pour le déploiement de l'API sur Heroku

- **dashboard.py** : Script python pour le test d'un dashboard Streamlit simpliste

- **Berthe_Pierrick_7_presentation_02024.pdf** : Présentation des résultats


# <span style='background:blue'>Auteur</span>

Pierrick BERTHE<br>
Février 2024
