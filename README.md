# Projet Bayésien

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
- télécharger le dataset depuis Kaggle : [IMDB Movies Dataset](https://www.kaggle.com/datasets/ashishjangra27/imdb-movies-dataset)
- extraire le fichier `movies.csv` et le placer dans le dossier `data/`

## Lancement
- sélectionner le kernel de l'environnement virtuel dans Jupyter
- lancer le notebook
    + en ligne de commande, sur un terminal :
        ```bash
        jupyter <chemin vers le notebook choisi>
        ```
    + ou depuis le notebook Jupyter, en cliquant sur "Run" pour exécuter les cellules du notebook