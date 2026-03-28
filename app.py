import streamlit as st
import joblib
import numpy as np
from datetime import datetime

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="HealthAI Diagnostic Tool",
    page_icon="⚕️",
    layout="centered"
)

# --- 2. CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background-color: #1e88e5;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover { background-color: #1565c0; }
    .report-card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border-top: 5px solid #1e88e5;
        margin-top: 20px;
    }
    .diagnosis-header { color: #555; font-size: 0.9em; margin-bottom: 0; }
    .diagnosis-result { color: #d32f2f; margin-top: 0; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load('final_model.pkl')
    symptoms_list = list(joblib.load('symptoms_list.pkl'))
    return model, symptoms_list

try:
    model, symptoms_list = load_assets()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("Project Info")
    st.write("This AI assistant uses a **Random Forest Classifier** trained on medical datasets.")
    st.divider()
    st.caption("Developed by: Pranav")
    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d')}")

# --- 5. MAIN CONTENT ---
st.title("🩺 Smart Medical Assistant")
st.markdown("Select your symptoms below to generate a professional diagnostic report.")

selected_symptoms = st.multiselect(
    "Search Symptoms:",
    options=symptoms_list,
    placeholder="e.g. itching, shivering, joint_pain"
)

st.divider()

# --- 6. PREDICTION & FORMATTING LOGIC ---
if st.button("Generate Diagnostic Report"):
    if selected_symptoms:
        # Create input vector
        input_vector = np.zeros(len(symptoms_list))
        for s in selected_symptoms:
            idx = symptoms_list.index(s)
            input_vector[idx] = 1
        
        # Model Prediction
        prediction = model.predict(input_vector.reshape(1, -1))
        disease = prediction[0][0]
        med_raw = prediction[0][1]

        # --- DATA SANITIZER (Fixes Vertical Text & Brackets) ---
        if isinstance(med_raw, str):
            clean_str = med_raw.replace("[", "").replace("]", "").replace("'", "")
            med_list = [m.strip() for m in clean_str.split(',')]
        else:
            med_list = med_raw

        # Formatted string for UI display
        formatted_meds_ui = "\n".join(med_list)

        # Display Result Card
        st.markdown(f"""
        <div class="report-card">
            <p class="diagnosis-header">PREDICTED CONDITION</p>
            <h2 class="diagnosis-result">{disease}</h2>
            <hr>
            <p class="diagnosis-header">RECOMMENDED MEDICATION</p>
            <h3 style="color: #2e7d32; margin-top: 0; white-space: pre-wrap; line-height: 1.6;">{formatted_meds_ui}</h3>
        </div>
        """, unsafe_allow_html=True)

        # --- 7. DOWNLOAD BUTTON (Arranged & Aligned) ---
        # Create a tab-indented list for the text file
        report_meds_list = "\n\t- ".join(med_list)

        report_content = f"""
MEDICAL ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
-------------------------------------------

SYMPTOMS SELECTED:
{', '.join(selected_symptoms)}

ANALYSIS:
Predicted Disease: {disease}

INITIAL MEDICATION:
\t- {report_meds_list}

-------------------------------------------
DISCLAIMER: This report is AI-generated for educational purposes. 
Please consult a medical professional.
-------------------------------------------
"""

        st.download_button(
            label="📥 Download This Report (.txt)",
            data=report_content,
            file_name=f"Diagnosis_Report_{disease.replace(' ', '_')}.txt",
            mime="text/plain"
        )
        
    else:
        st.warning("Please select at least one symptom to analyze.")

# Footer
st.markdown("<br><center><small>Healthcare AI Dashboard | Python & Streamlit</small></center>", unsafe_allow_html=True)