import streamlit as st
from google.cloud import aiplatform
import os
from PIL import Image
import base64
import requests
import json
# Step 1: Setup credentials and init Vertex AI



# Write service account to a temp file from secrets
creds = st.secrets["gcp_service_account"]
with open("temp_creds.json", "w") as f:
    json.dump(creds, f)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "temp_creds.json"

# Vertex AI setup
PROJECT_ID = creds["project_id"]
REGION = "us-central1"
ENDPOINT_ID = "2755100161788084224"

aiplatform.init(project=PROJECT_ID, location=REGION)

endpoint = aiplatform.Endpoint(
    endpoint_name=f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}"
)

# Streamlit UI
st.title("CIFAR-10 AutoML Image Classifier")

uploaded_file = st.file_uploader("Upload a CIFAR-10 image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width =True)

    image_bytes = uploaded_file.read()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    # Prediction instance format for AutoML
    instance = {"content": encoded_image}

    # Predict
    prediction = endpoint.predict(instances=[instance])

    # Show results
    if prediction.predictions:
        st.subheader("Prediction Results")
        for i, label in enumerate(prediction.predictions[0]["displayNames"]):
            score = prediction.predictions[0]["confidences"][i]
            st.write(f"{label}: {score * 100:.2f}%")
    else:
        st.warning("No prediction returned. Please try again.")

