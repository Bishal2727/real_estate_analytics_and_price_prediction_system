import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Price Predictor", page_icon="💰", layout="wide")



@st.cache_resource
def load_df():
    with open('df.pkl', 'rb') as file:
        return pickle.load(file)


@st.cache_resource
def load_pipeline():
    with open('pipeline.pkl', 'rb') as file:
        return pickle.load(file)


df = load_df()
pipeline = load_pipeline()

st.title('💰 Price Predictor')
st.header('Enter your inputs')

col1, col2 = st.columns(2)

with col1:
    property_type = st.selectbox('Property Type', ['flat', 'house'])
    sector = st.selectbox('Sector', sorted(df['sector'].unique().tolist()))
    bedrooms = float(st.selectbox('Number of Bedroom', sorted(df['bedRoom'].unique().tolist())))
    bathroom = float(st.selectbox('Number of Bathrooms', sorted(df['bathroom'].unique().tolist())))
    balcony = st.selectbox('Balconies', sorted(df['balcony'].unique().tolist()))
    property_age = st.selectbox('Property Age', sorted(df['agePossession'].unique().tolist()))

with col2:

    built_up_area = float(
        st.number_input(
            'Built Up Area (sqft)',
            min_value=100.0,
            max_value=20000.0,
            value=1000.0,
            step=50.0,
            help="Enter the carpet/built-up area in square feet. Must be at least 100 sqft."
        )
    )
    servant_room = float(st.selectbox('Servant Room', [0.0, 1.0]))
    store_room = float(st.selectbox('Store Room', [0.0, 1.0]))
    furnishing_type = st.selectbox('Furnishing Type', sorted(df['furnishing_type'].unique().tolist()))
    luxury_category = st.selectbox('Luxury Category', sorted(df['luxury_category'].unique().tolist()))
    floor_category = st.selectbox('Floor Category', sorted(df['floor_category'].unique().tolist()))

# Warn if area looks like an outlier relative to training data (extrapolation risk)
train_area_min, train_area_max = df['built_up_area'].quantile(0.01), df['built_up_area'].quantile(0.99)
if built_up_area < train_area_min or built_up_area > train_area_max:
    st.warning(
        f"⚠️ The entered area ({built_up_area:.0f} sqft) is outside the typical range seen in "
        f"training data ({train_area_min:.0f}–{train_area_max:.0f} sqft). The prediction may be less reliable."
    )

predict_clicked = st.button('Predict', type='primary')

if predict_clicked:
    # form a dataframe
    data = [[property_type, sector, bedrooms, bathroom, balcony, property_age,
             built_up_area, servant_room, store_room, furnishing_type,
             luxury_category, floor_category]]
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
               'agePossession', 'built_up_area', 'servant room', 'store room',
               'furnishing_type', 'luxury_category', 'floor_category']

    one_df = pd.DataFrame(data, columns=columns)

    # predict (model trained on log1p(price); invert with expm1)
    base_price = np.expm1(pipeline.predict(one_df))[0]


    MARGIN_PCT = 0.10  # +/- 10%
    low = base_price * (1 - MARGIN_PCT)
    high = base_price * (1 + MARGIN_PCT)

    st.success(
        f"### Estimated price: **₹{low:.2f} Cr – ₹{high:.2f} Cr**"
    )
    st.caption(f"Point estimate: ₹{base_price:.2f} Cr · Band: ±{int(MARGIN_PCT * 100)}%")

    st.divider()

    # ---- SHAP explainability ----
    st.subheader("🔍 Why this price?")
    st.caption("How each feature pushed the prediction up (red) or down (blue) from the average.")

    try:
        # Get the fitted preprocessor + model out of the pipeline.
        # Adjust the step names below ('preprocessor', 'model') if your pipeline
        # uses different step names -- check with: pipeline.named_steps
        preprocessor = pipeline.named_steps.get('preprocessor') or pipeline.steps[0][1]
        model = pipeline.named_steps.get('model') or pipeline.steps[-1][1]

        # Transform the single input row the same way the pipeline does internally
        transformed_input = preprocessor.transform(one_df)

        # Get feature names after transformation (works for ColumnTransformer w/ get_feature_names_out)
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = [f"feature_{i}" for i in range(transformed_input.shape[1])]

        # Dense array if sparse
        if hasattr(transformed_input, "toarray"):
            transformed_input = transformed_input.toarray()

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed_input)

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(
            shap_values, transformed_input,
            feature_names=feature_names,
            plot_type="bar", show=False, max_display=10
        )
        st.pyplot(fig, clear_figure=True)

        with st.expander("See detailed SHAP waterfall"):
            fig2 = plt.figure(figsize=(8, 5))
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value,
                shap_values[0],
                feature_names=feature_names,
                max_display=10,
                show=False
            )
            st.pyplot(fig2, clear_figure=True)

    except Exception as e:
        st.info(
            "Feature-level explanation isn't available for this model/pipeline structure. "
            f"({e})"
        )
