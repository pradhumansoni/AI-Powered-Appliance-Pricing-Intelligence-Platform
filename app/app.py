import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Ensure sys.path is correctly set for imports from src/
import sys
root = Path.cwd()

while root != root.parent:
    if (root / "src").exists():
        break
    root = root.parent

sys.path.insert(0, str(root))

from src.bi_layer import _preprocess_user_input, build_feature_dataframe, predict_price, get_prediction_range, get_shap_explanation, interpret_brand_premium
from src.recommender import ApplianceRecommender

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Smart Buy: AI Price & Deal Assistant", layout="wide")

st.title("🛒 Smart Buy: AI Price & Deal Assistant")
st.markdown("Analyze the best deals or explore fair market prices for home appliances.")

# --- 2. LOADING THE MODELS ---
@st.cache_resource
def load_resources():
    pipeline_path = Path(__file__).parent.parent / "models/saved_models/multi_category_xgb_pipeline.pkl"
    try:
        pipeline = joblib.load(pipeline_path)
        recommender = ApplianceRecommender() 
        return pipeline, recommender
    except Exception as e:
        st.error(f"❌ Failed to load model or recommender: {e}")
        st.stop()

try:
    price_pipeline, recommender_engine = load_resources()
    st.sidebar.success("✅ Models loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Critical error during app initialization: {e}")
    st.stop()

# --- 3. USER INPUT SECTION ---
st.header("🔍 Appliance Specifications")

col1, col2 = st.columns(2)

user_friendly_input = {}

with col1:
    user_friendly_input['category'] = st.selectbox("Select Category", ["AC", "Refrigerator", "Washing Machine"])
    user_friendly_input['brand_name'] = st.text_input("Enter Brand Name (e.g., LG, Samsung, Whirlpool)", "LG")
    
    if user_friendly_input['category'] == "AC":
        user_friendly_input['capacity_value'] = st.number_input("Capacity in Tons", min_value=0.5, max_value=3.0, value=1.5, step=0.5)
    elif user_friendly_input['category'] == "Refrigerator":
        user_friendly_input['capacity_value'] = st.number_input("Capacity in Litres", min_value=100, max_value=1000, value=250, step=10)
    else:  # Washing Machine
        user_friendly_input['capacity_value'] = st.number_input("Capacity in Kg", min_value=5, max_value=20, value=7, step=1)

with col2:
    user_friendly_input['star_rating'] = st.slider("Star Rating", 1, 5, 3)
    user_friendly_input['has_inverter'] = st.checkbox("Inverter Technology", value=True) # General for all categories

    # --- CATEGORY-SPECIFIC CHECKBOXES ---
    if user_friendly_input['category'] == "AC":
        user_friendly_input['ac_split'] = st.checkbox("Split AC", value=True)
        # Add more AC features here if needed, matching names in X_train_schema.md
        user_friendly_input['ac_pm25_filter'] = st.checkbox("PM 2.5 Filter", value=False)
        user_friendly_input['ac_hepa_filter'] = st.checkbox("HEPA Filter", value=False)
        user_friendly_input['ac_auto_clean'] = st.checkbox("Auto Clean", value=False)
        user_friendly_input['ac_hot_and_cold'] = st.checkbox("Hot & Cold", value=False)
        user_friendly_input['ac_copper_condenser'] = st.checkbox("Copper Condenser", value=True)
        user_friendly_input['ac_Dehumidification'] = st.checkbox("Dehumidification", value=True)
        user_friendly_input['ac_Turbo Mode'] = st.checkbox("Turbo Mode", value=True)
        user_friendly_input['ac_Self Diagnosis'] = st.checkbox("Self Diagnosis", value=False)

    elif user_friendly_input['category'] == "Refrigerator":
        user_friendly_input['ref_frost_free'] = st.checkbox("Frost Free", value=True)
        # Add more Refrigerator features here
        user_friendly_input['ref_multi_door'] = st.checkbox("Multi-Door", value=False)
        user_friendly_input['ref_french_door'] = st.checkbox("French Door", value=False)
        user_friendly_input['ref_double_door'] = st.checkbox("Double Door", value=True) # Assuming typical for many
        user_friendly_input['ref_convertible'] = st.checkbox("Convertible Freezer", value=False)
        user_friendly_input['ref_door_alarm'] = st.checkbox("Door Alarm", value=True)
        user_friendly_input['ref_door_lock'] = st.checkbox("Door Lock", value=False)
        user_friendly_input['ref_dispenser'] = st.checkbox("Water/Ice Dispenser", value=False)
        user_friendly_input['ref_door_display'] = st.checkbox("Door Display", value=True)

    else:  # Washing Machine
        user_friendly_input['wm_fully_automatic'] = st.checkbox("Fully Automatic", value=True)
        # Add more Washing Machine features here
        user_friendly_input['wm_with_dryer'] = st.checkbox("Washer Dryer Combo", value=False)
        user_friendly_input['wm_front_load'] = st.checkbox("Front Load", value=True) # Assuming common
        user_friendly_input['wm_top_load'] = st.checkbox("Top Load", value=False)
        user_friendly_input['wm_inbuilt_heater'] = st.checkbox("Inbuilt Heater", value=True)
        user_friendly_input['wm_quick_wash'] = st.checkbox("Quick Wash", value=True)
        user_friendly_input['wm_child_lock'] = st.checkbox("Child Lock", value=True)
        user_friendly_input['wm_shock_proof'] = st.checkbox("Shock Proof", value=False)
        user_friendly_input['wm_display'] = st.checkbox("Digital Display", value=True)


# --- 4. MODE SELECTION ---
st.divider()
st.header("🎯 Choose Your Goal")
mode = st.radio(
    "What would you like to do?",
    ["Explore Fair Prices", "Analyze a Deal"],
    horizontal=True
)

st.divider() # Separator before results

# --- 5. EXECUTION BUTTON ---
submit_button = st.button("Get Insights!")

if submit_button:
    # --- PREPROCESSING USER INPUT ---
    # Convert user-friendly input to model-compatible features
    processed_features = _preprocess_user_input(user_friendly_input)
    # Build the full DataFrame with defaults for missing features
    features_df = build_feature_dataframe(processed_features)

    if mode == "Explore Fair Prices":
        st.subheader(f"💡 Fair Price Estimation for your {user_friendly_input['category']}")

        # Get fair price and range
        predicted_price, _ = predict_price(features_df)
        lower_bound, upper_bound = get_prediction_range(processed_features) # Pass processed_features for consistency

        st.metric(
            label=f"Estimated Fair Price", 
            value=f"₹ {predicted_price:,.0f}"
        )
        st.write(f"*(Expected range: ₹{lower_bound:,.0f} - ₹{upper_bound:,.0f})*")

        # Get SHAP explanation and Brand Premium
        shap_explanation = get_shap_explanation(features_df)
        brand_premium_text = interpret_brand_premium(shap_explanation)
        
        st.markdown(f"**Brand Impact:** {brand_premium_text}")

        st.subheader("📊 Key Price Drivers (SHAP Explanation)")
        # Display top SHAP features
        shap_df = pd.DataFrame(shap_explanation['top_features'], columns=['Feature', 'Contribution'])
        st.dataframe(shap_df.style.format({"Contribution": "{:,.2f}"}), use_container_width=True)
        st.caption("Positive contribution increases price, negative contribution decreases price.")

        st.subheader("🌟 Smart Alternatives for Better Value")
        # Get recommendations using the recommender engine
        recommendations = recommender_engine.get_recommendations(user_friendly_input, n=3)
        
        if recommendations:
            recs_df = pd.DataFrame(recommendations)
            
            # Select relevant columns based on category
            if user_friendly_input['category'] == "AC":
                display_cols = ['brand_name', 'capacity_ac_tons', 'star_rating', 'actual_price', 'fair_price', 'value_score']
                display_names = ['Brand', 'Capacity (Tons)', 'Stars', 'Market Price', 'Fair Price', 'Value Score']
            elif user_friendly_input['category'] == "Refrigerator":
                display_cols = ['brand_name', 'capacity_ref_litres', 'star_rating', 'actual_price', 'fair_price', 'value_score', 'ref_frost_free', 'ref_double_door']
                display_names = ['Brand', 'Capacity (Litres)', 'Stars', 'Market Price', 'Fair Price', 'Value Score', 'Frost Free', 'Double Door']
            else: # Washing Machine
                display_cols = ['brand_name', 'capacity_wm_kg', 'star_rating', 'actual_price', 'fair_price', 'value_score', 'wm_fully_automatic', 'wm_front_load']
                display_names = ['Brand', 'Capacity (Kg)', 'Stars', 'Market Price', 'Fair Price', 'Value Score', 'Fully Automatic', 'Front Load']
            
            recs_df_display = recs_df[display_cols].copy()
            recs_df_display.columns = display_names
            
            for col in ['Market Price', 'Fair Price']:
                if col in recs_df_display.columns:
                    recs_df_display[col] = recs_df_display[col].apply(lambda x: f"₹ {x:,.0f}")
            
            recs_df_display['Value Score'] = recs_df_display['Value Score'].apply(lambda x: f"{x:.2f}")

            st.dataframe(recs_df_display, use_container_width=True, hide_row_index=True)
            st.caption("Higher 'Value Score' indicates a better deal (Fair Price > Market Price).")
        else:
            st.info("No alternative recommendations found with similar specifications.")

    else: # Analyze a Deal Mode - Logic for Stage 3 will go here
        st.info("Switch to 'Analyze a Deal' mode to compare with an observed price. (Logic coming in Stage 3)")