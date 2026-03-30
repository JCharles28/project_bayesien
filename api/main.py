import pandas as pd
import joblib, os, json
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# Chemins
PATH_MODEL_NAIVE_BAYES = "model/best_50_naive_bayes_model.pkl"
PATH_MODEL_DECISION_TREE = "model/best_40_decision_tree_model.pkl" 
PATH_MODEL_XGBOOST = "model/best_50_xgboost_model.pkl" 

PATH_DIRECTOR_TO_ENCODED = "data/director_to_encoded.json"
PATH_CERTIFICATE_TO_ENCODED = "data/certificate_to_encoded.json"


# fonction : charger le modèle depuis un fichier au format .pkl
async def load_model(path: str):
    model = joblib.load(path)
    return model

# fonction : prédire le genre d'un film à partir de ses caractéristiques
# donnees d'entrée pour la prédiction : year, runtim, imdb_rating, meta_score, votes, gross, director_encoded, certificate_encoded
# exemple : {"year": 2010, "runtime": 120, "imdb_rating": 7.5, "meta_score": 80, "votes": 100000, "gross": 50000000, "director_encoded": 10, "certificate_encoded": 2}
async def predict(model, input_data: dict):
    # Convertir les données d'entrée en DataFrame
    # Spécifier l'ordre des colonnes explicitement (important pour sklearn)
    columns_order = ["Released_Year", "Runtime", "IMDB_Rating", "Meta_score", "No_of_Votes", "Gross", "DirectorID", "CertificateID"]
    input_df = pd.DataFrame([input_data], columns=columns_order)
    
    # Faire la prédiction
    prediction = model.predict(input_df)[0]
    confidence = float(model.predict_proba(input_df).max())  # Confiance de la prédiction
    
    # Convertir les types numpy en types Python natifs pour la sérialisation JSON
    prediction = int(prediction) if isinstance(prediction, (np.integer, np.int64)) else prediction
    
    return {"prediction": prediction, "confidence": confidence}

def transform_log(x):
    return np.log(x + 1)  # Ajouter 1 pour éviter le log(0)

# ---- MODELE PYDANTIC POUR LES DONNEES D'ENTREE ----

class PredictionInput(BaseModel):
    year: int
    runtime: int
    imdb_rating: float
    meta_score: int
    votes: int
    gross: int
    director_encoded: int
    certificate_encoded: int

# ---- DONNEES ENTRANTES COMPLEXES ----

director_to_encoded = json.load(open(PATH_DIRECTOR_TO_ENCODED, "r"))
certificate_to_encoded = json.load(open(PATH_CERTIFICATE_TO_ENCODED, "r"))



# --- API ---
API = FastAPI(
    title="CinemAI API",
    description="""
    L'API de CinemAI permet de faire des prédictions de genre de film à partir de caractéristiques d'entrée en ayant à disposition plusieurs modèles de machine learning.
    """
)

@API.get("/")
def read_root():
    return {"Hello": "World"}

# Point d'accès → vérifier la santé/état de l'API
@API.get("/health")
def check_health():
    return {"status": "L'API est fonctionnelle et prête à être utilisée."}

# Point d'accès → prédiction avec modèle Naive Bayes
@API.post("/predict/naive_bayes")
async def predict_naive_bayes(data: PredictionInput):
    # chargement du modèle
    model = await load_model(PATH_MODEL_NAIVE_BAYES)
    
    # transformation des données d'entrée (log pour votes et gross)
    votes_log = float(transform_log(data.votes))
    gross_log = float(transform_log(data.gross))
    
    # préparation des données d'entrée avec les 8 colonnes
    input_data = {
        "Released_Year": float(data.year),
        "Runtime": float(data.runtime),
        "IMDB_Rating": float(data.imdb_rating),
        "Meta_score": float(data.meta_score),
        "No_of_Votes": votes_log,
        "Gross": gross_log,
        "DirectorID": float(data.director_encoded),
        "CertificateID": float(data.certificate_encoded)
    }
    
    result = await predict(model, input_data)
    return result

# Point d'accès → prédiction avec modèle Decision Tree
@API.post("/predict/decision_tree")
async def predict_decision_tree(data: PredictionInput):
    model = await load_model(PATH_MODEL_DECISION_TREE)
    
    votes_log = float(transform_log(data.votes))
    gross_log = float(transform_log(data.gross))
    
    input_data = {
        "Released_Year": float(data.year),
        "Runtime": float(data.runtime),
        "IMDB_Rating": float(data.imdb_rating),
        "Meta_score": float(data.meta_score),
        "No_of_Votes": votes_log,
        "Gross": gross_log,
        "DirectorID": float(data.director_encoded),
        "CertificateID": float(data.certificate_encoded)
    }
    
    result = await predict(model, input_data)
    return result

# Point d'accès → prédiction avec modèle XGBoost
@API.post("/predict/xgboost")
async def predict_xgboost(data: PredictionInput):
    model = await load_model(PATH_MODEL_XGBOOST)
    
    votes_log = float(transform_log(data.votes))
    gross_log = float(transform_log(data.gross))
    
    input_data = {
        "Released_Year": float(data.year),
        "Runtime": float(data.runtime),
        "IMDB_Rating": float(data.imdb_rating),
        "Meta_score": float(data.meta_score),
        "No_of_Votes": votes_log,
        "Gross": gross_log,
        "DirectorID": float(data.director_encoded),
        "CertificateID": float(data.certificate_encoded)
    }
    
    result = await predict(model, input_data)
    return result