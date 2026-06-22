import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Recommend Apartments", layout="wide")


# ---- Cached loaders ----
@st.cache_resource
def load_location_df():
    return pickle.load(open('datasets/location_distance.pkl', 'rb'))


@st.cache_resource
def load_cosine_sims():
    cosine_sim1 = pickle.load(open('datasets/cosine_sim1.pkl', 'rb'))
    cosine_sim2 = pickle.load(open('datasets/cosine_sim2.pkl', 'rb'))
    cosine_sim3 = pickle.load(open('datasets/cosine_sim3.pkl', 'rb'))
    return cosine_sim1, cosine_sim2, cosine_sim3


location_df = load_location_df()
cosine_sim1, cosine_sim2, cosine_sim3 = load_cosine_sims()


def recommend_properties_with_scores(property_name, top_n=5):
    cosine_sim_matrix = 0.5 * cosine_sim1 + 0.8 * cosine_sim2 + 1 * cosine_sim3

    sim_scores = list(enumerate(cosine_sim_matrix[location_df.index.get_loc(property_name)]))
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_indices = [i[0] for i in sorted_scores[1:top_n + 1]]
    top_scores = [i[1] for i in sorted_scores[1:top_n + 1]]
    top_properties = location_df.index[top_indices].tolist()

    recommendations_df = pd.DataFrame({
        'PropertyName': top_properties,
        'SimilarityScore': top_scores
    })

    return recommendations_df


st.title('Recommend Apartments')

# ---- Section 1: Search by location + radius ----
st.header('Search Apartments Near a Location')

col1, col2 = st.columns(2)
with col1:
    selected_location = st.selectbox('Location', sorted(location_df.columns.to_list()))
with col2:
    radius = st.number_input('Radius in Kms', min_value=0.5, value=2.0, step=0.5)

if st.button('Search'):
    result_ser = location_df[location_df[selected_location] < radius * 1000][selected_location].sort_values()

    if result_ser.empty:
        st.info(f"No apartments found within {radius} km of {selected_location}.")
    else:
        st.write(f"**{len(result_ser)} apartments found within {radius} km of {selected_location}:**")
        result_table = pd.DataFrame({
            'Apartment': result_ser.index,
            'Distance (km)': (result_ser.values / 1000).round(2)
        })
        st.dataframe(result_table, use_container_width=True, hide_index=True)

st.divider()

# ---- Section 2: Similar apartment recommender ----
st.header('Find Similar Apartments')
selected_apartment = st.selectbox('Select an apartment', sorted(location_df.index.to_list()))

top_n = st.slider('Number of recommendations', min_value=3, max_value=10, value=5)

if st.button('Recommend'):
    recommendation_df = recommend_properties_with_scores(selected_apartment, top_n=top_n)

    st.write(f"**Apartments most similar to {selected_apartment}:**")

    # Show as styled progress-bar-like score for readability
    display_df = recommendation_df.copy()
    display_df['SimilarityScore'] = display_df['SimilarityScore'].round(3)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "SimilarityScore": st.column_config.ProgressColumn(
                "Similarity",
                min_value=0,
                max_value=float(display_df['SimilarityScore'].max()) or 1.0,
                format="%.3f"
            )
        }
    )

    st.download_button(
        "Download as CSV",
        data=recommendation_df.to_csv(index=False),
        file_name=f"recommendations_{selected_apartment.replace(' ', '_')}.csv",
        mime="text/csv"
    )