import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data (only once)
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

download_nltk_data()

# Load model and vectorizer
@st.cache_resource
def load_model():
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# Initialize NLTK components
@st.cache_resource
def init_nltk():
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

# Text preprocessing function
def clean_text(text, stop_words, lemmatizer):
    """
    Clean and preprocess text using NLP techniques
    """
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and process
    words = text.split()
    
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

# Prediction function
def predict_sentiment(text, model, vectorizer, stop_words, lemmatizer):
    """
    Predict sentiment for given text
    """
    # Clean text
    cleaned = clean_text(text, stop_words, lemmatizer)
    
    # Vectorize
    vectorized = vectorizer.transform([cleaned])
    
    # Predict
    prediction = model.predict(vectorized)[0]
    
    # Get probability if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(vectorized)[0]
        confidence = dict(zip(model.classes_, probabilities))
    else:
        confidence = {prediction: 1.0}
    
    return prediction, confidence, cleaned

# Page Configuration
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide"
)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'index'
if 'tweet_text' not in st.session_state:
    st.session_state.tweet_text = ''
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# Navigation
def go_to_predict():
    st.session_state.page = 'predict'

def go_to_index():
    st.session_state.page = 'index'
    st.session_state.prediction_result = None

# Load resources
model, vectorizer = load_model()
stop_words, lemmatizer = init_nltk()

# ==================== INDEX PAGE ====================
if st.session_state.page == 'index':
    
    # Header
    st.title("🐦 Twitter Sentiment Analysis")
    st.markdown("### Analyze the sentiment of tweets using NLP and Machine Learning")
    
    st.markdown("---")
    
    # Main content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("## 📝 Enter Tweet Text")
        st.markdown("Type or paste the tweet you want to analyze:")
        
        # Text input
        tweet_input = st.text_area(
            label="Tweet Text",
            placeholder="Example: I love this product! It's amazing!",
            height=150,
            key="tweet_input",
            label_visibility="collapsed"
        )
        
        st.markdown("")  # Spacing
        
        # Analyze button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            analyze_button = st.button(
                "🔍 Analyze Sentiment",
                type="primary",
                use_container_width=True
            )
        
        # Process when button clicked
        if analyze_button:
            if tweet_input.strip():
                st.session_state.tweet_text = tweet_input
                
                # Perform prediction
                prediction, confidence, cleaned = predict_sentiment(
                    tweet_input, model, vectorizer, stop_words, lemmatizer
                )
                
                st.session_state.prediction_result = {
                    'original': tweet_input,
                    'cleaned': cleaned,
                    'sentiment': prediction,
                    'confidence': confidence
                }
                
                # Navigate to predict page
                go_to_predict()
                st.rerun()
            else:
                st.error("⚠️ Please enter some text to analyze!")
    
    # Footer info
    st.markdown("---")
    st.markdown("### ℹ️ About")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("**📊 Models Used**")
        st.markdown("- Naive Bayes")
        st.markdown("- Logistic Regression")
        st.markdown("- Random Forest")
    
    with col_b:
        st.markdown("**🔧 NLP Techniques**")
        st.markdown("- Stopwords Removal")
        st.markdown("- Lemmatization")
        st.markdown("- TF-IDF Vectorization")
    
    with col_c:
        st.markdown("**🎯 Sentiment Classes**")
        st.markdown("- Positive")
        st.markdown("- Negative")
        st.markdown("- Neutral")

# ==================== PREDICT PAGE ====================
elif st.session_state.page == 'predict':
    
    # Header
    st.title("🎯 Sentiment Prediction Results")
    
    # Back button
    if st.button("← Back to Home"):
        go_to_index()
        st.rerun()
    
    st.markdown("---")
    
    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        
        # Display original tweet
        st.markdown("## 📄 Original Tweet")
        st.info(result['original'])
        
        st.markdown("")
        
        # Display prediction
        sentiment = result['sentiment']
        confidence = result['confidence']
        max_confidence = max(confidence.values())
        
        # Sentiment colors
        sentiment_colors = {
            'Positive': '#28a745',
            'Negative': '#dc3545',
            'Neutral': '#ffc107',
            'Irrelevant': '#6c757d'
        }
        
        color = sentiment_colors.get(sentiment, '#6c757d')
        
        # Main prediction display
        st.markdown("## 🎯 Predicted Sentiment")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(
                f"""
                <div style="
                    background-color: {color}; 
                    padding: 30px; 
                    border-radius: 10px; 
                    text-align: center;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                ">
                    <h1 style="color: white; margin: 0; font-size: 3em;">{sentiment}</h1>
                    <p style="color: white; margin: 10px 0 0 0; font-size: 1.2em;">
                        Confidence: {max_confidence*100:.2f}%
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("")
        st.markdown("")
        
        # Confidence scores for all classes
        st.markdown("## 📊 Confidence Scores")
        
        # Sort by confidence
        sorted_confidence = sorted(confidence.items(), key=lambda x: x[1], reverse=True)
        
        for sent, conf in sorted_confidence:
            col_label, col_bar = st.columns([1, 4])
            
            with col_label:
                st.markdown(f"**{sent}**")
            
            with col_bar:
                st.progress(conf)
                st.caption(f"{conf*100:.2f}%")
        
        st.markdown("")
        st.markdown("---")
        
        # Show preprocessing details
        with st.expander("🔍 View Preprocessing Details"):
            st.markdown("### Original Text:")
            st.code(result['original'])
            
            st.markdown("### Cleaned Text (after preprocessing):")
            st.code(result['cleaned'])
            
            st.markdown("### Preprocessing Steps Applied:")
            st.markdown("""
            1. ✅ Converted to lowercase
            2. ✅ Removed URLs
            3. ✅ Removed mentions (@username)
            4. ✅ Removed hashtags (#hashtag)
            5. ✅ Removed special characters and numbers
            6. ✅ Removed stopwords (using NLTK)
            7. ✅ Applied lemmatization (using NLTK)
            """)
        
        # Try another button
        st.markdown("")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            if st.button("🔄 Analyze Another Tweet", type="primary", use_container_width=True):
                go_to_index()
                st.rerun()
    
    else:
        st.warning("No prediction result found. Please go back and analyze a tweet.")
        if st.button("Go to Home"):
            go_to_index()
            st.rerun()
