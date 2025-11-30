# Twitter Sentiment Analysis - NLP Project

## 🎯 Project Overview
Sentiment analysis on Twitter data using NLP techniques and Machine Learning.

## 🚀 Quick Start

### 1. Install Libraries
```bash
Install from requirements.txt
```

### 2. Download Dataset (Already provided in data folder)
- Visit: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis
- Download the CSV file
- Place it in the same folder as the notebook

### 3. Run Notebook
```bash
jupyter notebook twitter_sentiment_nlp.ipynb
```

## 📝 What It Does

### 1. **Data Loading & EDA**
   - Load Twitter dataset
   - Visualize sentiment distribution
   - Analyze sample tweets

### 2. **NLP Preprocessing** (using NLTK)
   - Lowercase conversion
   - Remove URLs, mentions, hashtags
   - Remove special characters
   - **Stopwords removal** (NLTK)
   - **Lemmatization** (NLTK)

### 3. **Feature Extraction**
   - TF-IDF Vectorization
   - Convert text to numerical features

### 4. **Model Training** (3 models)
   - Naive Bayes
   - Logistic Regression
   - Random Forest Classifier

### 5. **Model Comparison**
   - Compare accuracy
   - Select best model
   - Save for deployment

## 📊 Output Files

After running the notebook, you'll get:
- `sentiment_model.pkl` - Best performing model
- `vectorizer.pkl` - TF-IDF vectorizer

These files are ready to use in your Streamlit app!

## 📚 Libraries Used

- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **matplotlib** - Visualizations
- **scikit-learn** - Machine Learning models
- **nltk** - NLP preprocessing (stopwords, lemmatization)

## 🔍 NLP Preprocessing Steps

1. **Lowercasing**: Convert all text to lowercase
2. **URL Removal**: Remove http/https links
3. **Mention Removal**: Remove @mentions
4. **Hashtag Removal**: Remove #hashtags
5. **Special Character Removal**: Keep only letters
6. **Stopwords Removal**: Remove common words (using NLTK)
7. **Lemmatization**: Convert words to base form (using NLTK)

## 🎯 Next Steps

After completing this notebook:
1. ✅ Trained models saved as `.pkl` files
2. 🚀 Run Streamlit app for predictions

