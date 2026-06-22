import streamlit as st

st.set_page_config(
    page_title="Welcome to VESTA",
    page_icon="🏠",
    layout="wide"
)

st.title("VESTA: A Real Estate Analytics Platform")
st.markdown("### Data-driven insights for the Gurgaon property market")

st.markdown("""
This app helps you explore, understand, and predict residential property prices 
across Gurgaon sectors using a dataset of real listings.
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**💰 Price Predictor**\n\nEstimate a flat or house price from its specifications and features.")
with col2:
    st.info("**📊 Analytics**\n\nExplore price trends, area distributions, and sector-wise geomaps.")
with col3:
    st.info("**🏘️ Recommender**\n\nFind similar apartments and nearby landmarks with keywords.")

