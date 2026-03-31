import streamlit as st
import requests, json, re, unicodedata, html
import pandas as pd
from typing import Dict, Tuple

# ============================================================================
# Données de correspondance pour les encodages
# ============================================================================
PATH_DIRECTOR_TO_ENCODED = "data/director_to_encoded.json"

PATH_CERTIFICATE_TO_ENCODED = "data/certificate_to_encoded.json"

PATH_ENCODED_TO_GENRE = "data/encoded_to_genre.json"

with open(PATH_DIRECTOR_TO_ENCODED, "r") as f:
    dict_director_to_encoded = json.load(f)

with open(PATH_CERTIFICATE_TO_ENCODED, "r") as f:
    dict_certificate_to_encoded = json.load(f)
    
with open(PATH_ENCODED_TO_GENRE, "r") as f:
    dict_encoded_to_genre = json.load(f)
    
_MOJIBAKE_RE = re.compile(
    r"[\xc0-\xc3\xc5\xc6\xc8-\xcf\xd0-\xd6\xd8-\xdd\xe0-\xef\xf0-\xf6\xf8-\xfd]"
    r"[\x80-\xbf]",
    re.UNICODE,
)

_OBVIOUS_MARKERS = frozenset([
    "Ã", "Â", "â€", "Ã©", "Ã¨", "Ã ", "Ã¢", "Ã®", "Ã´", "Ã»",
    "Ã‡", "Ã±", "Ã¼", "Ã¶", "Ã„", "Ã–", "Ãœ",
    "â€™", "â€œ", "â€\x9c", "â€\x9d",
    "Â«", "Â»", "Â°", "Â·",
    "ï»¿", "â", "¤", "¿", "½", "¼", "¾",
    "\ufffd"])

_SOURCE_ENCODINGS = ["latin1", "cp1252", "cp850", "iso-8859-15", "mac_roman"]
MAX_PASSES = 3

def _has_mojibake(s: str) -> bool:
    if any(marker in s for marker in _OBVIOUS_MARKERS):
        return True
    return bool(_MOJIBAKE_RE.search(s))


def _unicode_score(s: str) -> float:
    if not s:
        return 1.0
    good = sum(
        1 for c in s
        if unicodedata.category(c) in {
            "Ll", "Lu", "Lt", "Lm", "Lo",
            "Nd", "Nl", "No",
            "Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps",
            "Zs", "Sm", "Sc", "So",
        }
    )
    return good / len(s)


def _try_repair(s: str) -> str | None:
    """
    Retourne la meilleure réparation trouvée, ou None si aucune n'est meilleure.
    Logique assouplie : on accepte dès que le score ne régresse pas
    ET qu'il n'y a pas de caractère de remplacement U+FFFD introduit.
    """
    best = None
    best_score = _unicode_score(s)

    for source_enc in _SOURCE_ENCODINGS:
        try:
            candidate = s.encode(source_enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue

        if candidate == s:
            continue

        # Rejeter si la réparation introduit des caractères de remplacement
        if "\ufffd" in candidate:
            continue

        score = _unicode_score(candidate)

        # ✅ Seuil assoupli : accepter si score ≥ original (sans marge arbitraire)
        # car Ã/©/¡ sont valides → les deux côtés ont un score similaire
        if score >= best_score:
            best = candidate
            best_score = score

    return best


def fix_mojibake_text(x):
    if not isinstance(x, str):
        return x

    s = x.strip()
    if not s:
        return s

    # 1. Supprimer le BOM UTF-8
    s = s.lstrip("\ufeff")

    # 2. Décoder les entités HTML (&eacute; → é)
    s_html = html.unescape(s)
    if s_html != s and "\ufffd" not in s_html:
        s = s_html

    # 3. Réparation itérative (multi-passes pour double mojibake)
    for _ in range(MAX_PASSES):
        if not _has_mojibake(s):
            break
        repaired = _try_repair(s)
        if repaired is None:
            break
        s = repaired

    # 4. Normalisation NFC
    s = unicodedata.normalize("NFC", s)

    return s
    
list_directors = list(dict_director_to_encoded.keys())
list_directors = [fix_mojibake_text(director) for director in list_directors]

list_certificates = list(dict_certificate_to_encoded.keys())



# ============================================================================
# Configuration de la page
# ============================================================================
st.set_page_config(
    page_title="CinemIA",
    page_icon="resources/logo/cinemIA_logo.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# mettre le logo avec le titre à côté
col_left, col_center, col_right = st.columns([3, 1, 3])
with col_center:
    st.image("resources/logo/cinemIA_logo.svg", width=150)
    
    
# st.title("CinemIA")
st.markdown("---")


# ============================================================================
# Barre latérale - Configuration des endpoints
# ============================================================================
with st.sidebar:
    with st.expander("ℹ️ À propos"):
        # Centrer l'image
        col_left, col_center, col_right = st.columns([1, 1, 1])
        with col_center:
            st.image("resources/logo/cinemIA_logo.svg", width=100)
        
        st.markdown("""
            
        **CinemIA**, une application de prédiction de genre de film basée sur des modèles de machine learning.
        
        - **Naive Bayes**: Classification probabiliste
        - **Decision Tree**: Classification par arbre de décision
        - **XGBoost**: Classification par boosting de gradient
        
        Les paramètres d'entrée sont les mêmes pour les deux modèles.
        
        """)
        with st.expander(" Données d'entrée attendues"):
            st.markdown("""
            - **Année de sortie** du film
            - **Durée** du film en minutes
            - **Note IMDb** : note moyenne des utilisateurs IMDb
            - **Meta Score** : score agrégé par des professionnels de la critique cinématographique
            - **Nombre de votes** sur IMDb
            - **Revenus bruts** du film
            - **Réalisateur** du film
            - **Classification d'âge** pour film
                + 18+ : déconseillé aux moins de 18 ans
                + 13+ : déconseillé aux moins de 13 ans
                + pg : nécessite l'accord parental
                + u : tous publics
                + nr : non classé
            
            """)
    
    # st.markdown("---")
    
    st.header("⚙️ Configuration")
    st.markdown("### Connectez vos endpoints API")
    
    endpoint = st.text_input(
        "Endpoint de Prédiction",
        value="http://localhost:8000",
        placeholder="http://localhost:8000"
    )
    
    if endpoint:
        # checker la santé de l'API
        try:
            response = requests.get(f"{endpoint}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ API connectée et opérationnelle!")
            else:
                st.error(f"❌ Erreur de connexion: {response.status_code}")
        except requests.exceptions.Timeout:
            st.error("⏱️ Timeout: L'API ne répond pas (5s)")
    
    # Champs pour les URLs des endpoints
    with st.expander("ℹ️ Endpoints détaillés"):
        endpoint_bayes = st.text_input(
            "Endpoint Naive Bayes",
            value="http://localhost:8000/predict/naive_bayes",
            placeholder="/predict/naive-bayes"
        )
        
        endpoint_tree = st.text_input(
            "Endpoint Decision Tree",
            value="http://localhost:8000/predict/decision_tree",
            placeholder="/predict/decision-tree"
        )
        
        endpoint_xgboost = st.text_input(
            "Endpoint XGBoost",
            value="http://localhost:8000/predict/xgboost",
            placeholder="/predict/xgboost"
        )
        
    # st.markdown("""
    # **Format attendu pour les données d'entrée:**
    # ```json
    # {
    #     "year": int,
    #     "runtime": int,
    #     "imdb_rating": float,
    #     "meta_score": int,
    #     "votes": int,
    #     "gross": int,
    #     "director_encoded": int,
    #     "certificate_encoded": int
    # }
    # ```
    # """)

# ============================================================================
# Formulaire principal
# ============================================================================
st.markdown("### Paramètres du film")

# Section 1: Informations générales
with st.container():
    st.markdown("##### 🎬 Informations Générales")
    col1, col2 = st.columns(2)
    
    with col1:
        year = st.slider(
            "📅 Année de sortie",
            min_value=1900,
            max_value=2100,
            value=2020,
            help="L'année de sortie du film"
        )
    
    with col2:
        runtime = st.slider(
            "⏱️ Durée (min)",
            min_value=0,
            max_value=500,
            value=120,
            step=1,
            help="La durée du film en minutes"
        )

st.divider()

# Section 2: Critiques et scores
with st.container():
    st.markdown("##### ⭐ Critiques & Scores")
    col1, col2 = st.columns(2)
    
    with col1:
        imdb_rating = st.slider(
            "🎯 Note IMDb",
            min_value=0.0,
            max_value=10.0,
            value=7.5,
            step=0.1,
            help="Note IMDb sur 10"
        )
    
    with col2:
        meta_score = st.slider(
            "🏆 Meta Score",
            min_value=0,
            max_value=100,
            value=70,
            step=1,
            help="Score Metacritic sur 100"
        )

st.divider()

# Section 3: Engagement et revenus
with st.container():
    st.markdown("##### 💰 Engagement & Finances")
    col1, col2 = st.columns(2)
    
    with col1:
        votes = st.number_input(
            "👥 Votes IMDb",
            min_value=0,
            max_value=2000000,
            value=100000,
            step=1000,
            help="Nombre de votes sur IMDb"
        )
    
    with col2:
        gross = st.number_input(
            "💵 Revenus Bruts",
            min_value=0,
            max_value=3000000000,
            value=100000000,
            step=10000,
            help="Revenus bruts du film"
        )

st.divider()

# Section 4: Détails
with st.container():
    st.markdown("##### 📋 Détails")
    col1, col2 = st.columns(2)
    
    with col1:
        director_encoded = st.selectbox(
            "👤 Réalisateur",
            options=list_directors,
            help="Sélectionnez le réalisateur du film"
        )
    
    with col2:
        certificate_encoded = st.selectbox(
            "🎞️ Classification d'âge",
            options=list_certificates,
            help="Sélectionnez le certificat de classification"
        )

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
    "director_encoded": int(dict_director_to_encoded.get(director_encoded, 0)),
    "certificate_encoded": int(dict_certificate_to_encoded.get(certificate_encoded, 0))
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
            print(data)
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
st.markdown("---")
col_button1, col_button2 = st.columns([2, 2])

with col_button1:
    predict_button = st.button("🚀 Prédire", use_container_width=True, type="primary")

with col_button2:
    reset_button = st.button("🔄 Réinitialiser", use_container_width=True)

# ============================================================================
# Exécution des prédictions
# ============================================================================
if predict_button:
    st.markdown("---")
    st.markdown("### 📊 Résultats")
    
    # Appels parallèles (simulation)
    col_bayes, col_tree, col_xgboost = st.columns(3)
    
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
                if "prediction" in result:
                    if isinstance(result.get("prediction", None), int):
                        genre_decoded = dict_encoded_to_genre.get(str(result.get("prediction")), "N/A")
                        st.metric("Genre Prédit", genre_decoded)
                if "confidence" in result:
                    if isinstance(result.get("confidence", None), (int, float)):
                        pourcentage_confiance = float(result.get("confidence", 0)) * 100
                        if pourcentage_confiance < 45:
                            st.metric(f"Confiance faible, {pourcentage_confiance:.2f}%")
                        else:
                            st.metric("Confiance", f"{pourcentage_confiance:.2f}%")
                    else:
                        st.metric("Confiance", "N/A")
                
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
                if "prediction" in result:
                    if isinstance(result.get("prediction", None), int):
                        genre_decoded = dict_encoded_to_genre.get(str(result.get("prediction")), "N/A")
                        st.metric("Genre Prédit", genre_decoded)
                if "confidence" in result:
                    if isinstance(result.get("confidence", None), (int, float)):
                        pourcentage_confiance = float(result.get("confidence", 0)) * 100
                        if pourcentage_confiance < 45:
                            st.metric(f"Confiance faible, {pourcentage_confiance:.2f}%")
                        else:
                            st.metric("Confiance", f"{pourcentage_confiance:.2f}%")
                    else:
                        st.metric("Confiance", "N/A")
                
                # Affichage brut des données si nécessaire
                with st.expander("📋 Détails complets"):
                    st.json(result)
            except Exception as e:
                st.error(f"Erreur lors de l'affichage: {str(e)}")
                st.json(result)
        else:
            st.error(message)

    # Résultats XGBoost
    with col_xgboost:
        st.markdown("#### ⚡ XGBoost")
        with st.spinner("Chargement..."):
            success, message, result = call_endpoint(endpoint_xgboost, data)
        
        if success:
            st.success(message)
            
            # Affichage structuré des résultats
            try:
                if "prediction" in result:
                    if isinstance(result.get("prediction", None), int):
                        genre_decoded = dict_encoded_to_genre.get(str(result.get("prediction")), "N/A")
                        st.metric("Genre Prédit", genre_decoded)
                if "confidence" in result:
                    if isinstance(result.get("confidence", None), (int, float)):
                        pourcentage_confiance = float(result.get("confidence", 0)) * 100
                        if pourcentage_confiance < 45:
                            st.metric(f"Confiance faible, {pourcentage_confiance:.2f}%")
                        else:
                            st.metric("Confiance", f"{pourcentage_confiance:.2f}%")
                    else:
                        st.metric("Confiance", "N/A")
                
                # Affichage brut des données si nécessaire
                with st.expander("📋 Détails complets"):
                    st.json(result)
            except Exception as e:
                st.error(f"Erreur lors de l'affichage: {str(e)}")
                st.json(result)
        else:
            st.error(message)