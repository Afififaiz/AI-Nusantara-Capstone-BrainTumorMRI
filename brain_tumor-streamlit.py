import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import os
import gdown
import zipfile

st.set_page_config(
    page_title="NeuroAI",
    page_icon="🧠")

# Define your CNNModel here without dropout and with the last 10 layers unfrozen
class CNNModel(nn.Module):
    def __init__(self, num_classes=4):
        super(CNNModel, self).__init__()
        
        # Load the pre-trained ResNet-50 model
        self.resnet = models.resnet50(pretrained=True)
        
        # Unfreeze the last 3 layers
        layers_to_unfreeze = 3
        for param in list(self.resnet.parameters())[-layers_to_unfreeze:]:
            param.requires_grad = True
        
        # Replace the last fully connected layer with a new one with `num_classes` outputs
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Google Drive file ID
GDRIVE_FILE_ID = '1rgnAFKBux3TTGUqu19ulJzhtxbmAAxTP'

# Download the model directly from Google Drive
@st.cache_resource  # Cache the download and loading of the model
def load_model_from_gdrive():
    url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
    output = 'temp_model.pth.tar'  # Temporary file name for the downloaded model
    gdown.download(url, output, quiet=False)
    
    model = CNNModel(num_classes=4)  # Instantiate your model
    
    # Load the state dictionary into the model
    model.load_state_dict(torch.load(output, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    
    os.remove(output)  # Remove the temporary model file after loading
    return model

# Define the transformations for the image
def transform_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")  # Convert image to RGB if it's not already
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


# Prediction function with confidence score
def predict(image, model):
    class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No tumor', 'Pituitary Tumor']
    image = transform_image(image)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return class_names[predicted.item()], confidence.item()


# Description function
def get_description(predicted_class):
    descriptions = {
        "Glioma Tumor": """
        **Glioma**: Gliomas are a type of tumor that occurs in the brain and spinal cord. They originate in the glial cells, which support nerve cells.
        - **Location**: Gliomas can occur anywhere in the brain or spinal cord, but are most common in the cerebral hemispheres.
        - **How They Form**: Gliomas form when glial cells, which provide support and insulation to nerve cells, begin to divide uncontrollably.
        - **Causes**: The exact cause of gliomas is unknown, but genetic factors and exposure to radiation are considered risk factors.
        - **Detection**: Gliomas are usually detected through MRI scans. Symptoms may include headaches, seizures, or neurological deficits.
        """,
        "Meningioma Tumor": """
        **Meningioma**: Meningiomas are typically benign tumors that arise from the meninges, the membranes that surround the brain and spinal cord.
        - **Location**: Meningiomas often form on the surface of the brain, near the skull, or at the base of the brain.
        - **How They Form**: These tumors develop from the meninges and can compress the brain as they grow, causing various symptoms.
        - **Causes**: The exact cause is unknown, but genetic mutations and exposure to radiation have been linked to meningioma development.
        - **Detection**: Meningiomas are often detected via MRI or CT scans, particularly when they cause symptoms such as headaches or changes in vision.
        """,
        "No tumor": """
        **No Tumor**: The MRI scan shows no evidence of a tumor.
        - **Location**: No abnormal growth detected in the brain.
        - **How They Form**: Since no tumor is present, there is no abnormal cell growth.
        - **Causes**: The patient may have symptoms related to other non-tumorous conditions.
        - **Detection**: Continuous monitoring and additional tests may be required to confirm the absence of a tumor or to diagnose other conditions.
        """,
        "Pituitary Tumor": """
        **Pituitary Tumor**: These tumors occur in the pituitary gland, which is located at the base of the brain. They can affect hormone levels and cause various symptoms.
        - **Location**: The pituitary gland is located in a small cavity in the skull, just below the brain, and behind the bridge of the nose.
        - **How They Form**: Pituitary tumors form from abnormal growth of cells in the pituitary gland, leading to hormonal imbalances.
        - **Causes**: Pituitary tumors may be caused by genetic mutations or hormonal factors.
        - **Detection**: MRI is the primary tool for detecting pituitary tumors, particularly when hormone levels are abnormal or when there are symptoms like headaches or vision problems.
        """
    }
    return descriptions.get(predicted_class, "No description available for this class.")


# Homepage
def homepage():
    st.image(r"ss\home.png", use_column_width=True)

    st.markdown("""
        <style>
        .video-container {
            position: relative;
            padding-bottom: 56.25%; /* 16:9 aspect ratio */
            height: 0;
            overflow: hidden;
            max-width: 100%;
            background: #000;
            border-radius: 15px; /* Rounded corners */
            margin: 0 auto;
        }

        .video-container iframe {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;  /* Full width */
            height: 100%; /* Full height */
            border-radius: 15px; /* Rounded corners for the iframe */
        }
                
                /* Style for the button */
        .stButton > button {
            background-color: #4CAF50; /* Green background */
            color: white; /* White text */
            padding: 10px 24px; /* Padding */
            font-size: 16px; /* Font size */
            border-radius: 8px; /* Rounded corners */
            border: none; /* Remove border */
            cursor: pointer; /* Pointer cursor on hover */
        }

        /* Style for the button on hover */
        .stButton > button:hover {
            background-color: #45a049; /* Darker green on hover */
        }
        </style>
    """, unsafe_allow_html=True)

    # Embed YouTube video with the correct aspect ratio and rounded corners
    st.markdown("""
        <div class="video-container">
            <iframe 
                src="https://www.youtube.com/embed/cSeXJKSQpiI" 
                frameborder="0" 
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen>
            </iframe>
        </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.write("")
    st.image(r"ss\home2.png", use_column_width=True)

       # Use session state for page navigation
    if st.button("Make Predictions"):
        st.session_state.page = "Prediction"


# Main Prediction Page
def prediction_page():
    st.image(r"ss\ai.png", use_column_width=True)
    st.write("Upload an image to classify. You can have the image through the button below. Download it and predict")

    if st.button("Go Download a sample"):
        st.session_state.page = "Dataset"

    # Load the model
    model = load_model_from_gdrive()

    # Uploading the image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image after resizing it to 224x224 pixels
        image = Image.open(uploaded_file)
        image = image.resize((224, 224))  # Resize the image to 224x224 pixels
        image = image.convert("RGB")  # Ensure the image is in RGB format
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Display loading message
        with st.spinner('Classifying...'):
            predicted_class, confidence = predict(image, model)
        
        # Show the predicted class with confidence
        st.write(f"Predicted Class: **{predicted_class}** ({confidence*100:.2f}% confidence)")

        # Display additional information about the predicted class
        description = get_description(predicted_class)
        st.markdown(description)

        # Store prediction in session state
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        st.session_state['history'].append({'image_name': uploaded_file.name, 'prediction': predicted_class})

# History Page
def history_page():
    st.image(r"ss\his.png", use_column_width=True)

    # Check if there's any history
    if 'history' in st.session_state and st.session_state['history']:
        for record in st.session_state['history']:
            st.write(f"Image: **{record['image_name']}** - Predicted Class: **{record['prediction']}**")
    else:
        st.write("No history yet. Make some predictions first!")

# Direct download link from Google Drive
GDRIVE_FILE_ID_PIC = '1xd_Ad3I_Nv0oNzkrlg3bHxmXITk9vm0f'
GDRIVE_URL = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID_PIC}&export=download'

# Download and extract the zip file from Google Drive
@st.cache_resource
def download_and_extract_zip():
    output = 'dataset.zip'  # Temporary file name for the downloaded zip
    gdown.download(GDRIVE_URL, output, quiet=False)
    
    # Extract the zip file
    extract_dir = 'extracted_dataset'
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    os.remove(output)  # Remove the temporary zip file after extraction
    return extract_dir

# Function to list categories and images
def list_categories_and_images(extract_dir):
    categories = [f for f in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, f))]
    selected_category = st.selectbox("Choose a category", categories)
    
    if selected_category:
        image_dir = os.path.join(extract_dir, selected_category)
        images = os.listdir(image_dir)
        selected_image = st.selectbox("Choose an image", images)
        
        if selected_image:
            image_path = os.path.join(image_dir, selected_image)
            image = Image.open(image_path)
            st.image(image, caption=f"{selected_category} - {selected_image}", use_column_width=True)
            
            # Provide download link
            with open(image_path, "rb") as file:
                st.download_button(
                    label="Download Image",
                    data=file,
                    file_name=selected_image,
                    mime="image/png"
                )

def dataset_page():
    st.image(r"ss\data.png", use_column_width=True)
    extract_dir = download_and_extract_zip()
    list_categories_and_images(extract_dir)


# Multi-Page Setup
def main():
    if 'page' not in st.session_state:
        st.session_state.page = "Home"

    st.sidebar.image(r"ss\ailogo.png", use_column_width=True)

    st.sidebar.title("Navigation")

    page = st.sidebar.radio("Go to", ["Home", "Prediction", "History", "Dataset"], index=["Home", "Prediction", "History", "Dataset"].index(st.session_state.page))

    if page != st.session_state.page:
        st.session_state.page = page

    if st.session_state.page == "Home":
        homepage()
    elif st.session_state.page == "Prediction":
        prediction_page()
    elif st.session_state.page == "History":
        history_page()
    elif st.session_state.page == "Dataset":
        dataset_page()

    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")

# Add copyright notice at the bottom of the sidebar
    st.sidebar.markdown("""
        <style>
        .sidebar .sidebar-content {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 100%;
        }
        .sidebar .copyright {
            text-align: center;
            font-size: 12px;
            padding: 20px 0;
        }
        </style>
        <div class="copyright">
            &copy; 2024 (Afifi Faiz) 
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

 # make design color
 # add more information
