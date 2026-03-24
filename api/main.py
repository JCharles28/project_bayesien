import pandas as pd
import joblib, os
from fastapi import FastAPI

# Chemins vers modèles
PATH_MODEL_NAIVE_BAYES = "../model/best_48_naive_bayes_model.pkl"
PATH_MODEL_DECISION_TREE = "../model/best_40_decision_tree_model.pkl"

# fonction : charger le modèle depuis un fichier au format .pkl
async def load_model(path: str):
    model = joblib.load(path)
    return model

# fonction : prédire le genre d'un film à partir de ses caractéristiques
# donnees d'entrée pour la prédiction : year, runtim, imdb_rating, meta_score, votes, gross, director_encoded, certificate_encoded
# exemple : {"year": 2010, "runtime": 120, "imdb_rating": 7.5, "meta_score": 80, "votes": 100000, "gross": 50000000, "director_encoded": 10, "certificate_encoded": 2}
async def predict(model, input_data: dict):
    # Convertir les données d'entrée en DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Faire la prédiction
    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df).max()  # Confiance de la prédiction
    
    return {"prediction": prediction, "confidence": confidence}


# --- API ---
API = FastAPI()

@API.get("/")
def read_root():
    return {"Hello": "World"}

# Point d'accès → vérifier la santé/état de l'API
@API.get("/health")
def read_health():
    return {"status": "healthy"}

# Point d'accès → prédiction avec modèle Naive Bayes
@API.get("/predict/naive_bayes")
async def predict_endpoint(year: int, runtime: int, imdb_rating: float, meta_score: int, votes: int, gross: int, director_encoded: int, certificate_encoded: int):
    # chargement du modèle
    model = await load_model(PATH_MODEL_NAIVE_BAYES)
    
    # préparation des données d'entrée
    input_data = {
        "year": float(year),
        "runtime": float(runtime),
        "imdb_rating": float(imdb_rating),
        "meta_score": float(meta_score),
        "votes": float(votes),
        "gross": float(gross),
        "director_encoded": float(director_encoded),
        "certificate_encoded": float(certificate_encoded)
        }
    
    # Faire la prédiction
    result = await predict(model, input_data)
    
    return result

# Point d'accès → prédiction avec modèle Decision Tree
@API.get("/predict/tree_decision")
async def predict_endpoint(year: int, runtime: int, imdb_rating: float, meta_score: int, votes: int, gross: int, director_encoded: int, certificate_encoded: int):
    # chargement du modèle
    model = await load_model(PATH_MODEL_DECISION_TREE)
    
    # préparation des données d'entrée
    input_data = {
        "year": float(year),
        "runtime": float(runtime),
        "imdb_rating": float(imdb_rating),
        "meta_score": float(meta_score),
        "votes": float(votes),
        "gross": float(gross),
        "director_encoded": float(director_encoded),
        "certificate_encoded": float(certificate_encoded)
        }
    
    # Faire la prédiction
    result = await predict(model, input_data)
    
    return result