
import streamlit as st
import re
import numpy as np
import gensim
import tensorflow as tf

st.set_page_config(page_title="Arabic Text Classifier", page_icon="🧠")


@st.cache_resource
def load_models():
    cnn = tf.keras.models.load_model("cnn_model.h5")
    w2v = gensim.models.KeyedVectors.load("word2vec_model.kv")
    return cnn, w2v

cnn_model, w2v_model = load_models()
class_names = [
    'Dental medicine',
    'Dermatology',
    'Ear, Nose & Throat Problems',
    'General Medicine',
    'Ophthalmology & Eye Diseases'
]
def clean_text(doc):
    # if text is None:
    #   text = ""
    chars = '[٠١٢٣٤٥٦٧٨٩0123456789[؟|$|.|!_،,@!#%^&*();<>":``.//\',\']'
    doc = re.sub(r'[^\w]+', ' ', doc)
    doc = re.sub(r'[a-zA-Z]', r'', doc)
    doc = re.sub(chars, r'', str(doc))
    doc = re.sub(r'\s+', r' ', doc, flags=re.I)  # remove multiple spaces with single space
    doc = " ".join([word for word in doc.split() if len(word) > 2])
    doc = doc.replace('\n', ' ')
    doc = re.sub(r'[إأآا]', 'ا', doc)
    doc = doc.replace('ة','ه').replace('ى','ي').replace('ؤ','و').replace('ئ','ي')
    doc = re.sub("ـ", "", doc)
    doc = re.sub("[ًٌٍَُِّْ]", "", doc)

    return doc


# def sentence_to_seq(sentence, max_len=916, dim=100):
#     seq = [
#         w2v_model[word] if word in w2v_model else np.zeros(dim)
#         for word in sentence.split()
#     ]

#     if len(seq) < max_len:
#         seq += [np.zeros(dim)] * (max_len - len(seq))
#     else:
#         seq = seq[:max_len]

#     return np.array(seq)
MAX_SEQUENCE_LENGTH = 916
VECTOR_SIZE = w2v_model.vector_size
def vectorize_text(tokens, w2v_model, max_len=916, vector_size=100):

    vectors = []
    
    for word in tokens:
        if word in w2v_model.key_to_index:  # use .key_to_index for KeyedVectors
            vectors.append(w2v_model[word])
        else:
            vectors.append(np.zeros(vector_size))  # OOV words → zeros
    
    # Pad or truncate sequence
    if len(vectors) < max_len:
        padding = [np.zeros(vector_size)] * (max_len - len(vectors))
        vectors.extend(padding)
    else:
        vectors = vectors[:max_len]
    
    return np.array(vectors)

def process_text(text):
    cleaned = clean_text(text)
    seq = vectorize_text(cleaned, w2v_model)
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
            # Preprocessing
          # CNN expects batch dimension
            processed = process_text(user_input)
            # Prediction
            prediction = cnn_model.predict(processed)
            predicted_index = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            
            # Map index to class name
            predicted_class = class_names[predicted_index]

        st.success("✅ تم الإدخال بنجاح")
        st.write("🔮 التوقع:", predicted_class)
        st.write("📊 نسبة الثقة:", f"{confidence*100:.2f}%")
