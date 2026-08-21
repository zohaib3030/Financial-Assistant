# Financial Assistant

An AI-powered personal finance assistant built with Python and Streamlit. It automatically categorizes daily expenses using Machine Learning and generates personalized money-saving strategies using Generative AI.

## Features

* Natural-language expense input
* ML-based expense categorization
* Category-wise spending analysis
* Spending visualization
* Personalized AI saving strategies
* Budget recommendations

## Tech Stack

* Python
* Streamlit
* Pandas
* Scikit-learn
* TF-IDF
* Logistic Regression
* Groq API / Llama

## How It Works

```text
Expense Input
     ↓
ML Categorization
     ↓
Spending Analysis
     ↓
Generative AI
     ↓
Saving Strategies
```

## API Configuration

For Groq-powered recommendations, configure your API key using Streamlit Secrets:

```toml
GROQ_API_KEY = "your_api_key"
```

Never commit API keys to GitHub.

## Disclaimer

This project provides general financial guidance for educational purposes and is not a substitute for professional financial advice.

## Author

**Zohaib Sharif**
