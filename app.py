
import streamlit as st
import re
import numpy as np
import gensim
import tensorflow as tf

st.set_page_config(page_title="Arabic Text Classifier", page_icon="🧠")


@st.cache_resource
def load_models():
    cnn = tf.keras.models.load_model("cnn_model.h5")
    w2v = gensim.models.KeyedVectors.load("word2vec.kv")
    return cnn, w2v

cnn_model, w2v_model = load_models()


def clean_text(text):
    chars = r'[٠١٢٣٤٥٦٧٨٩0123456789؟$|.!_،,@!#%^&*();<>":`\'/\\]'
    
    doc = re.sub(r'[^\w]+', ' ', text)
    doc = re.sub(r'[a-zA-Z]', '', doc)
    doc = re.sub(chars, '', doc)
    doc = re.sub(r'\s+', ' ', doc)
    doc = " ".join([word for word in doc.split() if len(word) > 2])
    
    return doc.strip()


def sentence_to_seq(sentence, max_len=916, dim=100):
    seq = [
        w2v_model[word] if word in w2v_model else np.zeros(dim)
        for word in sentence.split()
    ]

    if len(seq) < max_len:
        seq += [np.zeros(dim)] * (max_len - len(seq))
    else:
        seq = seq[:max_len]

    return np.array(seq)


def process_text(text):
    cleaned = clean_text(text)
    seq = sentence_to_seq(cleaned)
    return np.expand_dims(seq, axis=0)


# -------------------------------
# UI
# -------------------------------
st.title("🧠 Arabic Text Classification")
st.write("أدخل النص وسيتم تصنيفه باستخدام CNN + Word2Vec")

user_input = st.text_area("من ماذا تعاني؟")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("⚠️ الرجاء إدخال نص")
    else:
        with st.spinner("جاري التحليل..."):
            processed = process_text(user_input)
            prediction = cnn_model.predict(processed)
            predicted_class = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

        # Example class labels (EDIT these)
        classes = ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5"]

        st.success("✅ تم الإدخال بنجاح")
        st.write("🔮 التوقع:", classes[predicted_class])
        st.write("📊 نسبة الثقة:", f"{confidence*100:.2f}%")
