import streamlit as st
import requests
import pandas as pd
from typing import Dict, Tuple

# ============================================================================
# Configuration de la page
# ============================================================================
st.set_page_config(
    page_title="Prédiction Multi-Modèles",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎬 Prédiction Multi-Modèles de Genres Cinématographiques")
st.markdown("---")

# ============================================================================
# Barre latérale - Configuration des endpoints
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("### Connectez vos endpoints API")
    
    # Champs pour les URLs des endpoints
    endpoint_bayes = st.text_input(
        "Endpoint Naive Bayes",
        value="http://localhost:8000/predict/naive-bayes",
        placeholder="http://localhost:8000/predict/naive-bayes"
    )
    
    endpoint_tree = st.text_input(
        "Endpoint Decision Tree",
        value="http://localhost:8000/predict/decision-tree",
        placeholder="http://localhost:8000/predict/decision-tree"
    )
    
    st.markdown("---")
    st.markdown("""
    **Format attendu pour les requêtes:**
    ```json
    {
        "year": int,
        "runtime": int,
        "imdb_rating": float,
        "meta_score": int,
        "votes": int,
        "gross": int,
        "director_encoded": int,
        "certificate_encoded": int
    }
    ```
    """)

# ============================================================================
# Formulaire principal
# ============================================================================
st.markdown("### 📋 Paramètres du Film")

col1, col2, col3, col4 = st.columns(4)

with col1:
    year = st.number_input("Année", min_value=1900, max_value=2100, value=2020, step=1)
    imdb_rating = st.number_input("Note IMDb", min_value=0.0, max_value=10.0, value=7.5, step=0.1)

with col2:
    runtime = st.number_input("Durée (min)", min_value=0, max_value=500, value=120, step=1)
    meta_score = st.number_input("Meta Score", min_value=0, max_value=100, value=70, step=1)

with col3:
    votes = st.number_input("Votes IMDb", min_value=0, max_value=2000000, value=100000, step=1000)
    director_encoded = st.number_input("Réalisateur (Code)", min_value=0, max_value=10000, value=1, step=1)

with col4:
    gross = st.number_input("Revenus Bruts", min_value=0, max_value=3000000000, value=100000000, step=1000000)
    certificate_encoded = st.number_input("Certificat (Code)", min_value=0, max_value=10000, value=1, step=1)

# ============================================================================
# Construction des données
# ============================================================================
data = {
    "year": int(year),
    "runtime": int(runtime),
    "imdb_rating": float(imdb_rating),
    "meta_score": int(meta_score),
    "votes": int(votes),
    "gross": int(gross),
    "director_encoded": int(director_encoded),
    "certificate_encoded": int(certificate_encoded)
}

# ============================================================================
# Fonction pour appeler un endpoint
# ============================================================================
def call_endpoint(url: str, data: Dict) -> Tuple[bool, str, Dict]:
    """
    Appelle un endpoint API et retourne (succès, message, résultats)
    """
    try:
        response = requests.post(url, json=data, timeout=5)
        
        if response.status_code == 200:
            return True, "✅ Succès", response.json()
        else:
            return False, f"❌ Erreur {response.status_code}: {response.text}", {}
    
    except requests.exceptions.Timeout:
        return False, "⏱️ Timeout: Endpoint ne répond pas (5s)", {}
    except requests.exceptions.ConnectionError:
        return False, "🔌 Erreur de connexion: Vérifiez l'URL", {}
    except requests.exceptions.RequestException as e:
        return False, f"❌ Erreur: {str(e)}", {}
    except Exception as e:
        return False, f"❌ Erreur inattendue: {str(e)}", {}

# ============================================================================
# Bouton de prédiction
# ============================================================================
col_button1, col_button2, col_button3 = st.columns([1, 1, 2])

with col_button1:
    predict_button = st.button("🚀 Prédire", use_container_width=True, type="primary")

with col_button2:
    reset_button = st.button("🔄 Réinitialiser", use_container_width=True)

# ============================================================================
# Exécution des prédictions
# ============================================================================
if predict_button:
    st.markdown("---")
    st.markdown("### 📊 Résultats des Prédictions")
    
    # Appels parallèles (simulation)
    col_bayes, col_tree = st.columns(2)
    
    # Résultats Naive Bayes
    with col_bayes:
        st.markdown("#### 🤖 Naive Bayes")
        with st.spinner("Chargement..."):
            success, message, result = call_endpoint(endpoint_bayes, data)
        
        if success:
            st.success(message)
            
            # Affichage structuré des résultats
            try:
                # Adaptez les clés selon votre API
                if "genre_pred" in result:
                    st.metric("Genre Prédit", result.get("genre_pred", "N/A"))
                if "confidence" in result:
                    st.metric("Confiance", f"{result.get('confidence', 'N/A'):.2f}%")
                
                # Affichage brut des données si nécessaire
                with st.expander("📋 Détails complets"):
                    st.json(result)
            except Exception as e:
                st.error(f"Erreur lors de l'affichage: {str(e)}")
                st.json(result)
        else:
            st.error(message)
    
    # Résultats Decision Tree
    with col_tree:
        st.markdown("#### 🌳 Decision Tree")
        with st.spinner("Chargement..."):
            success, message, result = call_endpoint(endpoint_tree, data)
        
        if success:
            st.success(message)
            
            # Affichage structuré des résultats
            try:
                if "genre_pred" in result:
                    st.metric("Genre Prédit", result.get("genre_pred", "N/A"))
                if "confidence" in result:
                    st.metric("Confiance", f"{result.get('confidence', 'N/A'):.2f}%")
                
                # Affichage brut des données si nécessaire
                with st.expander("📋 Détails complets"):
                    st.json(result)
            except Exception as e:
                st.error(f"Erreur lors de l'affichage: {str(e)}")
                st.json(result)
        else:
            st.error(message)

# ============================================================================
# Information supplémentaire
# ============================================================================
st.markdown("---")
with st.expander("ℹ️ À propos"):
    st.markdown("""
    **Application de Prédiction Multi-Modèles**
    
    Cette interface permet de tester vos modèles de ML en appelant vos endpoints API.
    
    - **Naive Bayes**: Classification probabiliste
    - **Decision Tree**: Classification par arbre de décision
    
    Les 8 paramètres d'entrée sont les mêmes pour les deux modèles.
    """)
