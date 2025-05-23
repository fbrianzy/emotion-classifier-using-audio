import streamlit as st
import os
import numpy as np
import base64
import soundfile as sf
import librosa
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import tempfile

# Konfigurasi halaman
st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f0f0f0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .success-text {
        color: #4CAF50;
        font-weight: bold;
    }
    .info-text {
        color: #2196F3;
        font-style: italic;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    .result-section {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #c8e6c9;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #f0f0f0;
        color: #000000;
        font-size: 0.8rem;
    }
    .stAudio {
        margin-top: 1rem;
        margin-bottom: 1rem;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpg;base64,{encoded_string});
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Path ke model
MODEL_RF_PATH = "C:\\Project_UAS_PMD\\Model\\model_rf_83.pkl"
MODEL_RF_TUNED_PATH = "C:\\Project_UAS_PMD\\Model\\model_rf_tuned_81.pkl"

MODEL_GNB_PATH = "C:\\Project_UAS_PMD\\Model\\model_gnb_57.pkl"
MODEL_GNB_TUNED_PATH = "C:\\Project_UAS_PMD\\Model\\model_gnb_tuned_50.pkl"

MODEL_SVM_PATH = "C:\\Project_UAS_PMD\\Model\\model_svm_33.pkl"
MODEL_SVM_TUNED_PATH = "C:\\Project_UAS_PMD\\Model\\model_svm_tuned_68.pkl"

MODEL_XGB_PATH = "C:\\Project_UAS_PMD\\Model\\model_xgb_78.pkl"
MODEL_XGB_TUNED_PATH = "C:\\Project_UAS_PMD\\Model\\model_xgb_tuned_81.pkl"

@st.cache_resource
def load_model(model_path):
    """Memuat model dari file"""
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def extract_features(audio, sr):
    """Ekstrak fitur audio untuk prediksi"""
    features = []
    feature_names = []
    try:
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        # Basic features
        features.append(np.mean(audio))
        feature_names.append('Mean Amplitude')
        
        features.append(np.std(audio))
        feature_names.append('Standard Deviation')
        
        features.append(np.max(audio))
        feature_names.append('Maximum Amplitude')
        
        features.append(np.min(audio))
        feature_names.append('Minimum Amplitude')
        
        features.append(np.median(audio))
        feature_names.append('Median Amplitude')
        
        features.append(np.mean(np.abs(audio)))
        feature_names.append('Mean Absolute Amplitude')
        
        features.append(np.sqrt(np.mean(audio**2)))
        feature_names.append('Root Mean Square')
        
        features.append(np.sum(audio < 0))
        feature_names.append('Zero Crossings')
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        for i in range(mfccs.shape[0]):
            features.append(np.mean(mfccs[i, :]))
            feature_names.append(f'MFCC{i+1} Mean')
        
        for i in range(mfccs.shape[0]):
            features.append(np.std(mfccs[i, :]))
            feature_names.append(f'MFCC{i+1} Std')
        
        # Store feature names in a global variable to access later
        st.session_state['feature_names'] = feature_names
        
        return np.array(features).reshape(1, -1)
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def save_uploaded_wav(uploaded_file):
    """Menyimpan file WAV yang diunggah ke file temporary"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving WAV file: {e}")
        return None

def visualize_audio(audio, sr):
    """Membuat visualisasi waveform dan spektogram"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5))
    
    # Plot waveform
    time_axis = np.arange(0, len(audio)) / sr
    ax1.plot(time_axis, audio, color='#1E88E5')
    ax1.set_title('Waveform')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    
    # Plot spektogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax2, cmap='viridis')
    ax2.set_title('Spectrogram')
    ax2.set_xlabel('Time (s)')
    
    fig.tight_layout()
    return fig

def display_feature_importance(model, features):
    """Menampilkan fitur penting dari model"""
    try:
        importances = model.feature_importances_

        # Get feature names from session state if available
        if 'feature_names' in st.session_state:
            feature_names = st.session_state['feature_names']
        else:
            # Fallback to generic names if feature names not available
            n_features = len(importances)
            feature_names = [f"Feature_{i+1}" for i in range(n_features)]
        
        # Make sure we have the right number of feature names
        if len(feature_names) != len(importances):
            st.warning(f"Feature name count mismatch: {len(feature_names)} names vs {len(importances)} features")
            feature_names = [f"Feature_{i+1}" for i in range(len(importances))]
        
        # Create dataframe with feature names and importances
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        # Plot top 10 features
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_df['Feature'][:10][::-1], feature_df['Importance'][:10][::-1], color='#2196F3')
        ax.set_title('Top 10 Feature Importance')
        ax.set_xlabel('Importance')
        plt.tight_layout()

        return fig, feature_df
    except Exception as e:
        st.error(f"Error calculating feature importance: {e}")
        return None, None


def map_prediction_to_label(prediction):
    """Memetakan hasil prediksi ke label yang bermakna"""
    # Mapping dari indeks numerik (0,1,2,3,4) ke label emosi
    label_map = {
        0: "disappointed",
        1: "disgust",
        2: "happy",
        3: "neutral",
        4: "surprise"
    }
    try:
        return label_map.get(prediction, f"Unknown ({prediction})")
    except:
        return f"Unknown ({prediction})"

def main():
    if 'feature_names' not in st.session_state:
        st.session_state['feature_names'] = []

    st.markdown("<h1 class='main-header'>Audio Signal Emotion Classifier</h1>", unsafe_allow_html=True)

    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Upload WAV File</h2>", unsafe_allow_html=True)

        # Sidebar
    with st.sidebar:
        # st.image("https://img.freepik.com/free-vector/sound-wave-with-imitation-sound-audio-identification-technology_78370-866.jpg", use_container_width=True)
        st.markdown("### About")
        st.info("This Signal Emotion Classifier uses a Random Forest model to classify emotions from audio signals. "
                 "It extracts features from the audio and predicts the emotion based on the trained model.")
        
        st.markdown("### Instructions")
        st.markdown("""
        1. Upload a WAV audio file (.wav only)
        2. Wait for the audio to be processed
        3. View the classification result
        """)
        
        st.markdown("### Model Information")
        st.markdown("""
        - **Model Type**: Random Forest, Gaussian Naive Bayes
        - **Training Data**: Audio files with labeled emotions
        - **Classes**: Disappointed, Disgust, Happy, Neutral, Surprise
        - **Input Features**: Audio signal features
        - **Output**: Emotion classification
        - **Feature Extraction**:
            - Mean, Std, Max, Min, Median, Abs Mean, RMS, Zero Crossings
            - MFCC (Mel-frequency cepstral coefficients)
        - **Model Performance**:
            - Random Forest: 83% accuracy
            - Random Forest Tuned: 81% accuracy
            - Gaussian Naive Bayes: 57% accuracy
            - Gaussian Naive Bayes Tuned: 50% accuracy
            - SVM: 33% accuracy
            - SVM Tuned: 68% accuracy
            - XGBoost: 78% accuracy
            - XGBoost Tuned: 81% accuracy
        """)

    uploaded_file = st.file_uploader("Choose a WAV audio file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        with st.spinner('Processing audio file...'):
            progress_bar = st.progress(0)

            wav_path = save_uploaded_wav(uploaded_file)
            progress_bar.progress(20)

            if wav_path:
                try:
                    audio, sr = sf.read(wav_path)
                    progress_bar.progress(40)

                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1)

                    features = extract_features(audio, sr)
                    progress_bar.progress(100)

                    if features is not None:
                        # Model & Parameter Options
                        st.markdown("### 🔧 Model Selection")
                        model_choice = st.selectbox("Choose Model:", ["Random Forest", "Naive Bayes", "SVM", "XGBoost"])
                        param_choice = st.selectbox("Choose Parameter Set:", ["Default", "Tuned"])

                        # Mapping path
                        model_paths = {
                            "Random Forest": {
                                "Default": MODEL_RF_PATH,
                                "Tuned": MODEL_RF_TUNED_PATH
                            },
                            "Naive Bayes": {
                                "Default": MODEL_GNB_PATH,
                                "Tuned": MODEL_GNB_TUNED_PATH
                            },
                            "SVM": {
                                "Default": MODEL_SVM_PATH,
                                "Tuned": MODEL_SVM_TUNED_PATH
                            },
                            "XGBoost": {
                                "Default": MODEL_XGB_PATH,
                                "Tuned": MODEL_XGB_TUNED_PATH
                            }
                        }

                        selected_model_path = model_paths[model_choice][param_choice]
                        model = load_model(selected_model_path)

                        if model:
                            prediction = model.predict(features)[0]
                            label = map_prediction_to_label(prediction)

                            st.markdown("<div class='result-section'>", unsafe_allow_html=True)
                            st.markdown("<h2 class='sub-header'>Classification Result</h2>", unsafe_allow_html=True)
                            st.markdown(f"<p class='success-text'>🎯 {model_choice} ({param_choice}) Prediction: {label}</p>", unsafe_allow_html=True)

                            # Visualisasi audio
                            st.markdown("### Audio Visualization")
                            audio_fig = visualize_audio(audio, sr)
                            st.pyplot(audio_fig)

                            # Feature importance jika tersedia
                            if hasattr(model, 'feature_importances_'):
                                st.markdown("### Feature Importance")
                                importance_fig, feature_df = display_feature_importance(model, features)
                                if importance_fig:
                                    col1, col2 = st.columns([2, 1])
                                    with col1:
                                        st.pyplot(importance_fig)
                                    with col2:
                                        st.dataframe(feature_df)

                            st.markdown("</div>", unsafe_allow_html=True)

                        else:
                            st.error("Failed to load selected model.")

                except Exception as e:
                    st.error(f"Error processing audio: {e}")
                finally:
                    if wav_path and os.path.exists(wav_path):
                        os.unlink(wav_path)
            else:
                st.error("Failed to save WAV file")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='footer'>© 2025 Audio Classification System by S1 Sains Data Team </div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()