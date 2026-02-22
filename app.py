import streamlit as st
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('sentiment_model.keras')
    with open('word_index.json', 'r') as f:
        word_index = json.load(f)
    return model, word_index

model, word_index = load_model()
max_length = 200

def predict_sentiment(text):
    tokens = text.lower().split()
    encoded = [word_index.get(word, 2) + 3 for word in tokens]
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        [encoded], maxlen=max_length, padding='post', truncating='post'
    )
    prediction = model.predict(padded, verbose=0)[0][0]
    return prediction, padded

def get_shap_explanation(text, padded, prediction):
    tokens = text.lower().split()
    base_pred = prediction
    word_impacts = []
    for i in range(min(len(tokens), max_length)):
        modified = padded.copy()
        modified[0][i] = 0
        new_pred = model.predict(modified, verbose=0)[0][0]
        impact = base_pred - new_pred
        word_impacts.append((tokens[i], impact))
    word_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    return word_impacts[:10]

st.set_page_config(page_title="Sentiment Analyser", page_icon="🎬", layout="centered")
st.title("🎬 Movie Review Sentiment Analyser")
st.markdown("### With Explainable AI (Word Importance)")
st.markdown("---")

user_review = st.text_area("Enter a movie review:", height=120,
    placeholder="e.g., This movie was absolutely amazing with brilliant acting...")

if st.button("🔍 Analyse Sentiment", type="primary"):
    if user_review.strip() == "":
        st.warning("Please enter a review first!")
    else:
        with st.spinner("Analysing..."):
            prediction, padded = predict_sentiment(user_review)
            col1, col2 = st.columns(2)
            if prediction >= 0.5:
                confidence = prediction
                col1.metric("Sentiment", "✅ POSITIVE")
            else:
                confidence = 1 - prediction
                col1.metric("Sentiment", "❌ NEGATIVE")
            col2.metric("Confidence", f"{confidence:.1%}")
            st.markdown("---")
            st.subheader("🔍 Why this prediction?")
            st.caption("Words that most influenced the model's decision:")
            top_words = get_shap_explanation(user_review, padded, prediction)
            words = [pair[0] for pair in top_words]
            values = [pair[1] for pair in top_words]
            colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in values]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(range(len(words)), values, color=colors)
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words, fontsize=11)
            ax.set_xlabel('Impact on Prediction', fontsize=11)
            ax.set_title('Word Importance (Green = Positive, Red = Negative)', fontsize=12)
            ax.axvline(x=0, color='black', linewidth=0.5)
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("""
            *How to read this chart:*
            - 🟢 *Green bars* = words pushing towards POSITIVE
            - 🔴 *Red bars* = words pushing towards NEGATIVE
            - *Longer bar* = stronger influence on the prediction
            """)

st.markdown("---")
st.markdown("Built by *Pavan Krishna* | [GitHub](https://github.com/pavankrishna021-rgb) | Model: TensorFlow + Explainable AI")
