# 🎬 Movie Review Sentiment Analyser with Explainable AI

A real-time sentiment analysis system that predicts whether a movie review is *POSITIVE* or *NEGATIVE*, with explainable AI showing exactly which words influenced the prediction.

## 🔴 Live Demo
*[Try it here → http://18.132.228.208:8503](http://18.132.228.208:8503)*

## 🧠 What This Project Does
- Analyses movie reviews and predicts sentiment with confidence scores
- Explains WHY the model made each decision using word importance (SHAP-style)
- Interactive Streamlit dashboard anyone can use
- Trained on 50,000 IMDB movie reviews

## 📊 Model Performance
| Metric | Score |
|--------|-------|
| Test Accuracy | 85.67% |
| Precision (Positive) | 86% |
| Precision (Negative) | 85% |
| Recall (Positive) | 85% |
| Recall (Negative) | 86% |

## 🔍 Explainable AI
Instead of just showing predictions, the model explains its reasoning:
- 🟢 Green bars = words pushing towards POSITIVE
- 🔴 Red bars = words pushing towards NEGATIVE
- Longer bars = stronger influence

This is critical for regulatory compliance (GDPR, FCA) where customers have the right to know WHY an AI made a decision.

## 🛠 Tech Stack
- Python - Core language
- TensorFlow/Keras - Deep learning model
- SHAP-style Explainability - Word importance analysis
- Streamlit - Interactive web dashboard
- AWS EC2 - Cloud deployment
- NumPy/Matplotlib - Data processing and visualisation

## 🏗 Model Architecture
- Embedding Layer (10,000 words x 32 dimensions)
- Global Average Pooling
- Dense(32, ReLU) + L2 Regularisation + Dropout(0.5)
- Dense(16, ReLU) + L2 Regularisation + Dropout(0.5)
- Dense(1, Sigmoid)

## 🔧 Anti-Overfitting Techniques
- L2 Regularisation (penalty = 0.001)
- Dropout (50%)
- Early Stopping (patience = 3)
- Reduced model capacity

## 🚀 Run Locally
pip install tensorflow streamlit matplotlib numpy
streamlit run app.py

## 👨‍💻 Author
Pavan Krishna - Machine Learning Engineer

## 🌐 Other Live Projects
- [Customer Churn Predictor](http://18.132.228.208:8501) - 93.8% accuracy
- [Credit Card Fraud Detection](http://18.132.228.208:8502) - AUC-ROC 0.974
