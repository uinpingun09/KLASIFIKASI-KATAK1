import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np

# 1. Judul Halaman Web
st.title("🐸 Deteksi Jenis Katak dengan AI")
st.write("Unggah foto katak, dan AI akan menebak jenisnya!")

# 2. Load Model dan Labels
@st.cache_resource
def load_my_model():
    # Langsung panggil nama filenya sebagai string
    model = keras.models.load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        class_names = f.readlines()
    return model, class_names

try:
    model, class_names = load_my_model()

    # 3. Fitur Upload Gambar
    uploaded_file = st.file_uploader("Pilih gambar katak...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diunggah", use_container_width=True)
        
        st.write("🔍 Memproses dan menganalisis...")

        size = (224, 224)
        image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image_resized)

        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        st.success(f"**Hasil Prediksi:** {class_name}")
        st.info(f"**Tingkat Kepercayaan AI:** {confidence_score * 100:.2f}%")

except Exception as e:
    st.error(f"Gagal memuat model: {e}")
