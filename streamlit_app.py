import streamlit as st
import joblib
import numpy as np
import re
import os

# Safe TF import
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    pass

# Page Config
st.set_page_config(page_title="IMDB Duygu Analizi (v2)", layout="wide")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'ml_sentiment_model.pkl')
DL_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'dl_sentiment_model.h5')
VOCAB_PATH = os.path.join(BASE_DIR, 'models', 'dl_vocab.pkl')

# --- NLTK resources for stopword removal and stemming ---
import nltk
try:
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
    _stop_words = set(stopwords.words('english'))
    _stemmer = SnowballStemmer('english')
    NLTK_AVAILABLE = True
except Exception:
    try:
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        from nltk.stem import SnowballStemmer
        _stop_words = set(stopwords.words('english'))
        _stemmer = SnowballStemmer('english')
        NLTK_AVAILABLE = True
    except Exception:
        NLTK_AVAILABLE = False

# Text Cleaning Function - matches training pipeline
def clean_text(text):
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r'<.*?>', ' ', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special chars
    text = text.lower().strip()
    # Stopword removal (matches training)
    if NLTK_AVAILABLE:
        words = text.split()
        words = [w for w in words if w not in _stop_words]
        # Stemming (matches training)
        words = [_stemmer.stem(w) for w in words]
        text = " ".join(words)
    return text

# Load Models (Cached)
@st.cache_resource
def load_assets():
    ml_model = None
    dl_model = None
    vectorizer = None
    
    # Load ML
    if os.path.exists(ML_MODEL_PATH):
        try:
            ml_model = joblib.load(ML_MODEL_PATH)
        except Exception as e:
            st.error(f"ML Model yüklenemedi: {e}")
            
    # Load DL + Vocab
    if TF_AVAILABLE and os.path.exists(DL_MODEL_PATH) and os.path.exists(VOCAB_PATH):
        try:
            dl_model = tf.keras.models.load_model(DL_MODEL_PATH)
            
            # Reconstruct Vectorizer
            vocab = joblib.load(VOCAB_PATH)
            # DİNAMİK DÜZELTME: Kelime listesi boyutuna göre max_tokens ayarlanır
            MAX_TOKENS = len(vocab) 
            OUTPUT_LEN = 500   # Matches training
            
            vectorizer = layers.TextVectorization(
                max_tokens=MAX_TOKENS,
                output_mode='int',
                output_sequence_length=OUTPUT_LEN,
                vocabulary=vocab 
            )
            
        except Exception as e:
            st.error(f"DL Varlıkları yüklenemedi: {e}")
            
    return ml_model, dl_model, vectorizer

ml_pipeline, dl_model, vectorizer = load_assets()

# UI
st.title("🎬 IMDB Film Yorumu Duygu Analizi (v2)")
st.write("Makine Öğrenmesi ve Derin Öğrenme modelleri ile yorum analizi.")

# Input
user_input = st.text_area("Yorumunuzu İngilizce olarak giriniz:", height=150, placeholder="The movie was absolutely fantastic...")

if st.button("Analiz Et"):
    if not user_input or user_input.strip() == "":
        st.warning("Lütfen bir metin girin.")
    else:
        # Preprocess
        cleaned_text = clean_text(user_input)
        
        col1, col2 = st.columns(2)
        
        # --- ML Prediction ---
        with col1:
            st.subheader("🤖 Makine Öğrenmesi (Logistic Reg)")
            if ml_pipeline:
                try:
                    input_data = [str(cleaned_text)]
                    prediction = ml_pipeline.predict(input_data)[0]
                    proba = ml_pipeline.predict_proba(input_data)[0]
                    
                    sentiment = "Pozitif 😊" if prediction == 1 else "Negatif 😠"
                    confidence = proba[prediction]
                    
                    st.success(f"Sonuç: **{sentiment}**")
                    st.progress(float(confidence))
                    st.caption(f"Güven Skoru: {confidence:.4f}")
                except Exception as e:
                    st.error(f"Hata (ML): {e}")
            else:
                st.warning("ML Model bulunamadı.")

        # --- DL Prediction ---
        with col2:
            st.subheader("🧠 Derin Öğrenme (Keras Embedding)")
            if dl_model and vectorizer:
                try:
                    # 1. Vectorize Input (Text -> Int Sequence)
                    input_seq = vectorizer([str(cleaned_text)]).numpy()
                    
                    # 2. Predict (Input is Int Sequence)
                    pred_prob = dl_model.predict(input_seq, verbose=0)[0][0]
                    
                    if pred_prob > 0.5:
                        sentiment_dl = "Pozitif 😊"
                        confidence_dl = pred_prob
                    else:
                        sentiment_dl = "Negatif 😠"
                        confidence_dl = 1 - pred_prob
                    
                    st.success(f"Sonuç: **{sentiment_dl}**")
                    st.progress(float(confidence_dl))
                    st.caption(f"Güven Skoru: {confidence_dl:.4f} (Ham: {pred_prob:.4f})")
                except Exception as e:
                    st.error(f"Hata (DL): {e}")
            else:
                st.warning("DL Model veya Vocab bulunamadı.")

        st.divider()
        st.write("### İşlenen Metin (Cleaned)")
        st.code(cleaned_text)
