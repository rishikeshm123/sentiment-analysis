# Twitter Sentiment Analysis - Streamlit App

## 🚀 Quick Start

### 1. Prerequisites

Make sure you have the following files in the same directory:
- `app.py` - Main Streamlit application
- `sentiment_model.pkl` - Trained model (from notebook)
- `vectorizer.pkl` - TF-IDF vectorizer (from notebook)
- `requirements_streamlit.txt` - Dependencies

### 2. Install Dependencies

```bash
pip install -r requirements_streamlit.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📱 How to Use

### Page 1: Index (Input Page)
1. Enter or paste tweet text in the text area
2. Click "🔍 Analyze Sentiment" button
3. App will automatically navigate to prediction page

### Page 2: Predict (Results Page)
1. View the predicted sentiment (Positive/Negative/Neutral/Irrelevant)
2. See confidence scores for all sentiment classes
3. Check preprocessing details (optional)
4. Click "🔄 Analyze Another Tweet" to go back

## 🔧 Features

### Text Preprocessing (Automatic)
The app automatically applies these preprocessing steps in the background:
- ✅ Lowercase conversion
- ✅ URL removal
- ✅ Mentions removal (@username)
- ✅ Hashtags removal (#hashtag)
- ✅ Special characters & numbers removal
- ✅ Stopwords removal (NLTK)
- ✅ Lemmatization (NLTK)

### Sentiment Prediction
- Uses the trained Random Forest model
- Shows confidence percentages
- Color-coded results:
  - 🟢 Green for Positive
  - 🔴 Red for Negative
  - 🟡 Yellow for Neutral
  - ⚫ Gray for Irrelevant

### User-Friendly Interface
- Simple 2-page design
- Easy navigation
- Responsive layout
- Beautiful visualizations

## 📊 Example Tweets to Try

**Positive:**
```
I absolutely love this product! It works perfectly and exceeded my expectations!
```

**Negative:**
```
Worst service ever. Very disappointed with the quality and customer support.
```

**Neutral:**
```
The weather is nice today. Going for a walk in the park.
```

## 🛠️ Technical Details

### Libraries Used
- **Streamlit** - Web application framework
- **scikit-learn** - Model inference
- **NLTK** - Text preprocessing
- **pickle** - Model loading

### Model Information
- The app loads the best performing model from your training
- Typically Random Forest or Logistic Regression
- Uses TF-IDF features (5000 features)

## 📁 File Structure

```
project/
│
├── app.py                      # Main Streamlit app
├── sentiment_model.pkl         # Trained model
├── vectorizer.pkl              # TF-IDF vectorizer
├── requirements_streamlit.txt  # Dependencies
└── README_STREAMLIT.md        # This file
```

## 🐛 Troubleshooting

### Issue: "FileNotFoundError: sentiment_model.pkl"
**Solution:** Make sure you have run the Jupyter notebook and saved the model files in the same directory as `app.py`

### Issue: NLTK download errors
**Solution:** The app automatically downloads NLTK data on first run. If issues persist, manually run:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### Issue: "ModuleNotFoundError"
**Solution:** Install all requirements:
```bash
pip install -r requirements_streamlit.txt
```

## 🎨 Customization

### Change Colors
Edit the `sentiment_colors` dictionary in `app.py`:
```python
sentiment_colors = {
    'Positive': '#28a745',  # Green
    'Negative': '#dc3545',  # Red
    'Neutral': '#ffc107',   # Yellow
    'Irrelevant': '#6c757d' # Gray
}
```

### Add More Features
- You can add charts/graphs on the predict page
- Add history of predictions
- Export results to CSV
- Add batch prediction capability

## 📄 License

Educational project for sentiment analysis demonstration.

## 👨‍💻 Support

For issues:
1. Check that all `.pkl` files are present
2. Verify NLTK data is downloaded
3. Ensure all dependencies are installed

---

**Enjoy analyzing sentiments! 🎉**
