# Projet Bayésien

- fait par : **Ziad LYAGOUBI** et **Jean-Charles MCHANGAMA**

## Prérequis

- Python 3.8 ou supérieur

- création d'un environnement virtuel
    - Sur Windows:
        ```bash
        python -m venv env
        ```
    - Sur Linux/Mac:
        ```bash
        python3 -m venv env
        ```
- activation de l'environnement virtuel
    - Sur Windows:
        ```bash
        .\env\Scripts\activate
        ```
    - Sur Linux/Mac:
        ```bash
        source env/bin/activate
        ```

- installation des librairies nécessaires
    ```bash
    pip install -r requirements.txt
    ```

##  Dataset
- télécharger le dataset depuis Kaggle :
    + [IMDB Dataset of Top 1000 Movies and TV Shows](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows)
- extraire le fichier CSV et le placer dans le dossier `data/`

## Lancement des notebooks
- sélectionner le kernel de l'environnement virtuel dans Jupyter
- lancer le notebook
    + en ligne de commande, sur un terminal :
        ```bash
        jupyter <chemin vers le notebook choisi>
        ```
    + ou depuis le notebook Jupyter, en cliquant sur "Run" pour exécuter les cellules du notebook

## Lancement de l'API

- s'assurer que l'environnement virtuel est activé (voir section "Prérequis")

- s'assurer que les librairies nécessaires pour l'API sont téléchargées :
    ```bash
    pip install -r api/requirements.txt
    ```

- s'assurer que les modèles exportés sont présents dans le dossier `model/`

- lancer l'API avec FastAPI sur le port 8000 :
    ```bash
    fastapi run api/main.py --reload --port 8000
    ```

## Lancement de l'interface graphique (Streamlit)

- s'assurer que l'environnement virtuel est activé (voir section "Prérequis")

- lancer l'interface avec Streamlit sur le port 8501 :
    ```bash
    streamlit run ./main.py --server.port 8501
    ```