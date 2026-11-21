
import streamlit as st
import pandas as pd
import joblib

# --------------------------------
# 1. Load the trained machine learning model
# --------------------------------

model_data = joblib.load("bankruptcy_model.pkl")

model = model_data["model"]
features = model_data["features"]

# --------------------------------
# 2. Configure Streamlit page
# --------------------------------

st.set_page_config(
    page_title="Bankruptcy Prevention System",
    page_icon="📊",
    layout="centered"
)

# --------------------------------
# 3. Application title
# --------------------------------

st.title("🏦 Bankruptcy Prevention System")

st.write(
    "This application predicts whether a business "
    "belongs to the Bankruptcy or Non-bankruptcy class."
)

st.divider()

# --------------------------------
# 4. Business risk factor inputs
# --------------------------------

st.subheader("Enter Business Risk Factors")

risk_options = [0, 0.5, 1]

risk_labels = {
    0: "0 - Low",
    0.5: "0.5 - Medium",
    1: "1 - High"
}

input_values = {}

for feature in features:

    input_values[feature] = st.selectbox(
        feature.replace("_", " ").title(),
        options=risk_options,
        format_func=lambda x: risk_labels[x]
    )

st.divider()

# --------------------------------
# 5. Prediction button
# --------------------------------

if st.button("Predict Bankruptcy", type="primary"):

    # Convert input into DataFrame
    input_data = pd.DataFrame(
        [input_values],
        columns=features
    )

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.error("⚠️ Bankruptcy")

        st.write(
            "The model classifies this business "
            "as Bankruptcy."
        )

    else:

        st.success("✅ Non-bankruptcy")

        st.write(
            "The model classifies this business "
            "as Non-bankruptcy."
        )

# --------------------------------
# 6. Footer
# --------------------------------

st.divider()

st.caption(
    "Bankruptcy Prevention using Machine Learning | "
    "Educational Project"
)
