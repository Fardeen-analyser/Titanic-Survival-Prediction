import streamlit as st
import pickle
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Load the trained SVM model safely
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("⚠️ 'model.pkl' not found. Please ensure it is in the correct directory.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

model = load_model()

# 3. Custom CSS for a Professional look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f8;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# 4. App Header
st.title("🚢 Titanic Survival Prediction Dashboard")
st.markdown("""
    Welcome! This dashboard uses a trained **Support Vector Machine (SVM)** model to predict whether a passenger would have survived the tragic Titanic voyage based on their core demographics.
""")
st.write("---")

# 5. Sidebar Inputs for Passenger Data
st.sidebar.header("📋 Passenger Details")

# Passenger Class
pclass = st.sidebar.selectbox(
    "Passenger Class (Pclass)",
    options=[1, 2, 3],
    format_func=lambda x: f"Class {x} (Luxury)" if x == 1 else (f"Class {x} (Middle)" if x == 2 else f"Class {x} (Economy)")
)

# Gender
sex_input = st.sidebar.radio("Gender", options=["Female", "Male"])
sex = 0 if sex_input == "Female" else 1  # Standard encoding (0 for Female, 1 for Male)

# Companions
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.sidebar.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)

# Age
age = st.sidebar.slider("Age", min_value=0, max_value=100, value=28, step=1)

# 6. Prediction Logic
st.subheader("🔮 Prediction Results")

if model is not None:
    # Prepare the feature array using exactly the 5 selected features.
    # CRITICAL: Verify this matches the exact sequence of columns you passed to your model.fit()
    features = np.array([[pclass, sex, sibsp, parch, age]])
    
    # Add an action button
    if st.button("Analyze Survival Probability", type="primary"):
        with st.spinner("Calculating probabilities..."):
            prediction = model.predict(features)
            
            # Check if model supports probability estimates
            has_proba = hasattr(model, "predict_proba")
            if has_proba:
                probabilities = model.predict_proba(features)[0]
                survival_prob = probabilities[1] * 100
                perish_prob = probabilities[0] * 100

            st.write("") # Spacer
            
            if prediction[0] == 1:
                st.success("### 🎉 Result: Passenger Likely Survived!")
                if has_proba:
                    st.metric(label="Survival Confidence", value=f"{survival_prob:.1f}%")
                    st.progress(survival_prob / 100)
            else:
                st.error("### 😓 Result: Passenger Likely Did Not Survive.")
                if has_proba:
                    st.metric(label="Perish Probability", value=f"{perish_prob:.1f}%")
                    st.progress(perish_prob / 100)
                    
            # Updated Summary Box
            st.info(f"""
            **Passenger Summary:** A {age}-year-old **{sex_input.lower()}** traveling in **Class {pclass}** with **{sibsp}** sibling(s)/spouse(s) and **{parch}** parent(s)/child(ren).
            """)
else:
    st.warning("Please resolve the model loading error to run predictions.")

# 7. Footer
st.write("---")
st.caption("Built with Streamlit & Scikit-Learn | SVM Model Dashboard")
