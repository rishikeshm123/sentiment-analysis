# Twitter Sentiment Analysis - Complete Project Guide

## 📋 Project Structure

```
twitter-sentiment-analysis/
│
├── Part 1: Model Training (Jupyter Notebook)
│   ├── twitter_sentiment_analysis.ipynb
│   ├── twitter_training.csv
│   ├── twitter_validation.csv
│   └── requirements.txt
│
├── Part 2: Streamlit App
│   ├── app.py
│   ├── sentiment_model.pkl (generated from notebook)
│   ├── vectorizer.pkl (generated from notebook)
│
└── Documentation
    ├── README.md
    └── README_STREAMLIT.md
```

## 🚀 Complete Setup Guide

### Step 1: Model Training (Jupyter Notebook)

1. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib scikit-learn nltk
   ```

2. **Prepare datasets:**
   - Place `twitter_training.csv` and `twitter_validation.csv` in the same folder
   
3. **Run the notebook:**
   ```bash
   jupyter notebook twitter_sentiment_analysis.ipynb
   ```

4. **Run all cells** - This will:
   - Load and preprocess data
   - Train 3 models (Naive Bayes, Logistic Regression, Random Forest)
   - Evaluate on validation set
   - Save `sentiment_model.pkl` and `vectorizer.pkl`

### Step 2: Streamlit App

1. **Verify you have the pickle files:**
   - `sentiment_model.pkl` ✅
   - `vectorizer.pkl` ✅

2. **Install Streamlit dependencies:**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

4. **Open browser:**
   - App opens at `http://localhost:8501`

## 📱 How to Use the App

### Index Page (Home)
1. Enter tweet text in the text area
2. Click "Analyze Sentiment"
3. App processes and shows results

### Predict Page (Results)
1. View predicted sentiment
2. See confidence scores
3. Check preprocessing details
4. Analyze another tweet

## 🎯 Features

### Notebook (Part 1)
- ✅ Loads 74,682 training + 1,758 validation tweets
- ✅ EDA with visualizations
- ✅ NLTK preprocessing (stopwords, lemmatization)
- ✅ TF-IDF vectorization
- ✅ 3 model comparison
- ✅ Confusion matrix
- ✅ Best model selection & saving

### Streamlit App (Part 2)
- ✅ Clean, modern UI
- ✅ 2-page navigation (Index → Predict)
- ✅ Automatic text preprocessing
- ✅ Real-time sentiment prediction
- ✅ Confidence scores visualization
- ✅ Color-coded results
- ✅ Preprocessing details viewer

## 🔧 Technical Stack

### Part 1 (Training)
- **Python 3.8+**
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **matplotlib** - Visualizations
- **scikit-learn** - ML models
- **nltk** - NLP preprocessing

### Part 2 (Deployment)
- **Streamlit** - Web framework
- **pickle** - Model loading
- **NLTK** - Text preprocessing
- **scikit-learn** - Inference

## 📊 Model Performance

After running the notebook, you'll see:
- Training accuracy for each model
- Validation accuracy for each model
- Best model selected automatically
- Confusion matrix visualization

## 🎨 App Screenshots

### Index Page
- Large text input area
- Analyze button
- About section with project details

### Predict Page
- Original tweet display
- Large sentiment prediction (color-coded)
- Confidence percentage
- Progress bars for all classes
- Preprocessing details (expandable)

## 💡 Example Tweets to Test

```
Positive: "I love this product! It's amazing and works perfectly!"

Negative: "Worst experience ever. Very disappointed with the service."

Neutral: "The weather is nice today. Going for a walk."

Irrelevant: "Check out this link: http://example.com"
```

## 🐛 Common Issues & Solutions

### Issue 1: Model file not found
```
FileNotFoundError: sentiment_model.pkl
```
**Solution:** Run the Jupyter notebook first to generate the pickle files

### Issue 2: NLTK download errors
```
LookupError: Resource not found
```
**Solution:** The app auto-downloads NLTK data, but if needed:
```python
import nltk
nltk.download('all')
```

### Issue 3: Port already in use
```
Address already in use
```
**Solution:** Run Streamlit on different port:
```bash
streamlit run app.py --server.port 8502
```

## 📝 Project Workflow

1. **Data Collection** ✅ (Kaggle datasets)
2. **EDA** ✅ (Notebook)
3. **Preprocessing** ✅ (NLTK in notebook)
4. **Model Training** ✅ (3 algorithms)
5. **Model Evaluation** ✅ (Validation set)
6. **Model Saving** ✅ (Pickle files)
7. **App Development** ✅ (Streamlit)
8. **Deployment** ✅ (Local/Cloud)

## 🚀 Next Steps (Optional)

### Deploy to Cloud
- **Streamlit Cloud** (Free, easiest)
- **Heroku** (Free tier available)
- **AWS/GCP** (More control)

### Enhance Features
- Add tweet history
- Batch prediction
- Export to CSV
- Charts and analytics
- User authentication

### Improve Model
- Try deep learning (LSTM, BERT)
- Add more features
- Ensemble methods
- Hyperparameter tuning

## 📚 Documentation Files

1. **README.md** - Notebook documentation
2. **README_STREAMLIT.md** - App documentation
3. **QUICKSTART.md** - This file

## ✅ Checklist

Before running the app, ensure:
- [ ] Python 3.8+ installed
- [ ] All dependencies installed
- [ ] Datasets downloaded
- [ ] Notebook executed successfully
- [ ] `sentiment_model.pkl` generated
- [ ] `vectorizer.pkl` generated
- [ ] Streamlit installed

## 🎓 Learning Outcomes

After completing this project, you'll understand:
- Text preprocessing with NLTK
- TF-IDF vectorization
- Multiple ML algorithms (NB, LR, RF)
- Model evaluation and selection
- Streamlit app development
- End-to-end ML project workflow

---

**Ready to analyze sentiments! 🎉**

For questions or issues, check the troubleshooting section or review the README files.
