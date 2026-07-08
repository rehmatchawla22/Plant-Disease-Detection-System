import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json
import pandas as pd
from treatment import treatment
import os
import random

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# -----------------------------
# Background Gradient + CSS
# -----------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        to right,
        #e8f5e9,
        #f1f8e9
    );
}

.prediction-card {
    background-color: white;
    border: 3px solid #2E8B57;
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
model = tf.keras.models.load_model(
    "models/multicrop_mobilenetv2.keras"
)
base_model = model.get_layer("mobilenetv2_1.00_224")

print("\n\nLAST 30 LAYERS OF MOBILENETV2:\n")
for layer in base_model.layers[-30:]:
    print(layer.name)

# -----------------------------
# Load Class Names
# -----------------------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#2E8B57;'>
    🌿 Multi-Crop Plant Disease Detection System
    </h1>

    <p style='text-align:center; font-size:18px;'>
    Upload a leaf image and let AI identify the disease.
    </p>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🌿 Plant Disease Detection")

st.sidebar.write("### Model")
st.sidebar.success("MobileNetV2")

st.sidebar.write("### Number of Classes")
st.sidebar.success(len(class_names))

st.sidebar.write("## Supported Crops")

crops=[

"Apple",
"Blueberry",
"Cherry",
"Corn",
"Grape",
"Orange",
"Peach",
"Pepper",
"Potato",
"Raspberry",
"Soybean",
"Squash",
"Strawberry",
" Tomato"
]

for crop in crops:
    st.sidebar.write(f"•{crop}")


# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    img = Image.open(uploaded_file)

    # Resize
    resized_img = img.resize((224, 224))

    # Convert to array
    img_array = image.img_to_array(resized_img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Prediction
    prediction = model.predict(
        img_array,
        verbose=0
    )

    predicted_class = np.argmax(
        prediction
    )

    confidence = np.max(
        prediction
    ) * 100

    # Get class name
    full_name = class_names[
        predicted_class
    ]

    parts = full_name.split("___")

    crop_name = parts[0]
    disease_name = parts[1]

    disease_name = disease_name.replace(
        "_",
        " "
    )

    crop_name = crop_name.replace(
        "(",
        ""
    )

    crop_name = crop_name.replace(
        ")",
        ""
    )

    # -----------------------------
    # Disease Information
    # -----------------------------
    if "healthy" in disease_name.lower():

        info = f"""
        The {crop_name} leaf appears healthy.

        No visible disease symptoms were detected.
        """

    else:

        info = f"""
        Disease detected in {crop_name} crop.

        Disease Name: {disease_name}

        Please consult agricultural experts or refer to treatment guidelines for proper disease management.
        """

    # -----------------------------
    # Layout
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:

        st.image(
            img,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="prediction-card">

            <h2>
            🌿 Prediction
            </h2>

            <p style="font-size:22px;">
            🍃 <b>Crop:</b> {crop_name}
            </p>

            <p style="font-size:22px;">
            🦠 <b>Disease:</b> {disease_name}
            </p>

            <p style="font-size:22px;">
            📈 <b>Confidence:</b> {confidence:.2f}%
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(
            int(confidence)
        )

        st.info(
            info
        )

        # -----------------------------
        # Treatment Suggestions
        # -----------------------------
        if "healthy" in disease_name.lower():

            st.success(
                treatment["healthy"]
            )

        else:

            st.warning(
                treatment["default"]
            )

        # -----------------------------
        # Compare Images
        # -----------------------------
        example_folder = os.path.join(
            "examples",
            full_name
        )

        if os.path.exists(example_folder):

            image_files = [

                file

                for file in os.listdir(
                    example_folder
                )

                if file.lower().endswith(
                    (
                        ".jpg",
                        ".jpeg",
                        ".png"
                    )
                )
            ]

            if len(image_files) > 0:

                random_image = random.choice(
                    image_files
                )

                example_image = os.path.join(
                    example_folder,
                    random_image
                )

                st.markdown("---")

                st.subheader(
                    "🔍 Compare Images"
                )

                compare_col1, compare_col2 = st.columns(2)

                with compare_col1:

                    st.image(
                        img,
                        caption="📤 Uploaded Image",
                        use_container_width=True
                    )

                with compare_col2:

                    st.image(
                        example_image,
                        caption=f"🖼️ Example of {disease_name}",
                        use_container_width=True
                    )

    # -----------------------------
    # Top 5 Predictions (Bar Chart)
    # -----------------------------
    st.subheader("📊 Top 5 Predictions")

    top_indices = np.argsort(
        prediction[0]
    )[::-1][:5]

    names = []
    probs = []

    for idx in top_indices:

        name = class_names[idx]

        name = name.replace(
            "___",
            " : "
        )

        name = name.replace(
            "_",
            " "
        )

        names.append(name)

        probs.append(
            prediction[0][idx] * 100
        )

    df = pd.DataFrame({
        "Class": names,
        "Probability (%)": probs
    })

    st.bar_chart(
        df.set_index("Class")
    )