
import streamlit as st
import re
import numpy as np
import gensim
import tensorflow as tf

# -------------------------------
# Load models
# -------------------------------
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("cnn_model.h5")

@st.cache_resource
def load_word2vec_model():
    return gensim.models.KeyedVectors.load("word2vec.kv")

cnn_model = load_cnn_model()
w2v_model = load_word2vec_model()


# -------------------------------
# Text Processing
# -------------------------------
def sentence_to_seq(sentence, w2v_model, max_len, dim):
    seq = [w2v_model[word] if word in w2v_model else np.zeros(dim)
           for word in sentence.split()]
    
    if len(seq) < max_len:
        seq += [np.zeros(dim)] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
        
    return np.array(seq)


def process_text(text, w2v_model, max_len=916, dim=100):
    chars = '[٠١٢٣٤٥٦٧٨٩0123456789[؟|$|.|!_،,@!#%^&*();<>":``.//\',\']'
    
    doc = re.sub(r'[^\w]+', ' ', text)
    doc = re.sub(r'[a-zA-Z]', '', doc)
    doc = re.sub(chars, '', doc)
    doc = re.sub(r'\s+', ' ', doc)
    doc = " ".join([word for word in doc.split() if len(word) > 2])
    doc = doc.replace('\n', ' ')

    seq = sentence_to_seq(doc, w2v_model, max_len, dim)
    return np.expand_dims(seq, axis=0)  # Add batch dimension


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Arabic Text Classification App")

user_input = st.text_input("من ماذا تعاني؟")

if user_input:
    processed = process_text(user_input, w2v_model)
    prediction = cnn_model.predict(processed)
    predicted_class = np.argmax(prediction, axis=1)[0]

    st.success("✅ تم الإدخال بنجاح")
    st.write("🔮 التوقع (رقم الفئة):", predicted_class)
