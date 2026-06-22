import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")


# ---- Cached loaders ----
@st.cache_data
def load_data():
    return pd.read_csv('datasets/data_viz1.csv')


@st.cache_resource
def load_feature_text():
    return pickle.load(open('datasets/feature_text.pkl', 'rb'))


st.title('📊 Analytics')

new_df = load_data()
feature_text = load_feature_text()

group_df = new_df.groupby('sector')[['price', 'price_per_sqft', 'built_up_area', 'latitude', 'longitude']].mean()

st.header('Sector Price per Sqft Geomap')
fig = px.scatter_mapbox(
    group_df, lat="latitude", lon="longitude", color="price_per_sqft", size='built_up_area',
    color_continuous_scale=px.colors.cyclical.IceFire, zoom=10,
    mapbox_style="open-street-map", width=1200, height=700, hover_name=group_df.index
)
st.plotly_chart(fig, use_container_width=True)

st.header('Features Wordcloud')

wordcloud = WordCloud(
    width=800, height=800,
    background_color='black',
    stopwords=set(['s']),
    min_font_size=10
).generate(feature_text)

fig_wc, ax_wc = plt.subplots(figsize=(8, 8))
ax_wc.imshow(wordcloud, interpolation='bilinear')
ax_wc.axis("off")
fig_wc.tight_layout(pad=0)
st.pyplot(fig_wc, clear_figure=True)

st.header('Area Vs Price')

property_type = st.selectbox('Select Property Type', ['flat', 'house'])

filtered_df = new_df[new_df['property_type'] == property_type]
fig1 = px.scatter(filtered_df, x="built_up_area", y="price", color="bedRoom", title="Area Vs Price")
st.plotly_chart(fig1, use_container_width=True)

st.header('BHK Pie Chart')

sector_options = new_df['sector'].unique().tolist()
sector_options.insert(0, 'overall')

selected_sector = st.selectbox('Select Sector', sector_options)

if selected_sector == 'overall':
    fig2 = px.pie(new_df, names='bedRoom')
else:
    fig2 = px.pie(new_df[new_df['sector'] == selected_sector], names='bedRoom')

st.plotly_chart(fig2, use_container_width=True)

st.header('Side by Side BHK price comparison')

fig3 = px.box(new_df[new_df['bedRoom'] <= 4], x='bedRoom', y='price', title='BHK Price Range')
st.plotly_chart(fig3, use_container_width=True)

st.header('Side by Side Distplot for property type')

fig4, ax4 = plt.subplots(figsize=(10, 4))
sns.histplot(new_df[new_df['property_type'] == 'house']['price'], label='house', kde=True, ax=ax4, stat="density")
sns.histplot(new_df[new_df['property_type'] == 'flat']['price'], label='flat', kde=True, ax=ax4, stat="density")
ax4.legend()
st.pyplot(fig4, clear_figure=True)

st.divider()


st.header('Luxury Score vs Price')

fig5 = px.scatter(
    new_df, x='luxury_score', y='price', color='sector',
    size='built_up_area', hover_name='society',
    title='Luxury Score vs Price (bubble size = built-up area)',
    opacity=0.7
)
fig5.update_layout(showlegend=False)
st.plotly_chart(fig5, use_container_width=True)

st.header('Price by Property Age')

age_order = ['Under Construction', 'New Property', 'Relatively New', 'Moderately Old', 'Old Property']
age_present = [a for a in age_order if a in new_df['agePossession'].unique()]

fig6 = px.box(
    new_df, x='agePossession', y='price', color='agePossession',
    category_orders={'agePossession': age_present},
    title='Price Distribution by Age/Possession Status'
)
fig6.update_layout(showlegend=False)
st.plotly_chart(fig6, use_container_width=True)

st.header('Price per Sqft by Floor')


# Bucket floors for a cleaner trend line (raw floor 0-51 is too granular/noisy)
floor_df = new_df.copy()
floor_df['floor_bucket'] = pd.cut(
    floor_df['floorNum'],
    bins=[-1, 3, 7, 12, 20, 100],
    labels=['Ground-3', '4-7', '8-12', '13-20', '21+']
)
floor_avg = floor_df.groupby('floor_bucket', observed=True)['price_per_sqft'].mean().reset_index()

fig7 = px.bar(
    floor_avg, x='floor_bucket', y='price_per_sqft',
    title='Average Price/Sqft by Floor Range',
    labels={'floor_bucket': 'Floor Range', 'price_per_sqft': 'Avg Price per Sqft (₹)'},
    color='price_per_sqft', color_continuous_scale='Blues'
)
st.plotly_chart(fig7, use_container_width=True)

st.header('Furnishing Type vs Price')

furnish_map = {0.0: 'Unfurnished', 1.0: 'Semi-Furnished', 2.0: 'Furnished'}
furnish_df = new_df.copy()
furnish_df['furnishing_label'] = furnish_df['furnishing_type'].map(furnish_map)

fig8 = px.box(
    furnish_df, x='furnishing_label', y='price', color='furnishing_label',
    category_orders={'furnishing_label': ['Unfurnished', 'Semi-Furnished', 'Furnished']},
    title='Price by Furnishing Type'
)
fig8.update_layout(showlegend=False)
st.plotly_chart(fig8, use_container_width=True)

st.header('Amenity Impact on Price')
amenity_cols = ['study room', 'servant room', 'store room', 'pooja room', 'others']
amenity_impact = []
for col in amenity_cols:
    with_amenity = new_df[new_df[col] == 1]['price'].mean()
    without_amenity = new_df[new_df[col] == 0]['price'].mean()
    amenity_impact.append({
        'Amenity': col.title(),
        'With': with_amenity,
        'Without': without_amenity,
        'Premium (Cr)': with_amenity - without_amenity
    })

amenity_df = pd.DataFrame(amenity_impact).sort_values('Premium (Cr)', ascending=False)

fig9 = px.bar(
    amenity_df, x='Amenity', y='Premium (Cr)',
    title='Average Price Premium by Amenity',
    color='Premium (Cr)', color_continuous_scale='Greens'
)
st.plotly_chart(fig9, use_container_width=True)

with st.expander("See raw average prices with/without each amenity"):
    st.dataframe(
        amenity_df.style.format({'With': '{:.2f}', 'Without': '{:.2f}', 'Premium (Cr)': '{:.2f}'}),
        use_container_width=True, hide_index=True
    )

st.header('Top Societies by Average Price')


min_listings = st.slider('Minimum listings per society', min_value=3, max_value=20, value=5)

society_stats = (
    new_df[new_df['society'] != 'independent']
    .groupby('society')
    .agg(avg_price=('price', 'mean'), listings=('price', 'count'), avg_price_per_sqft=('price_per_sqft', 'mean'))
    .query('listings >= @min_listings')
    .sort_values('avg_price', ascending=False)
    .head(15)
    .reset_index()
)

fig10 = px.bar(
    society_stats, x='avg_price', y='society', orientation='h',
    title=f'Top 15 Societies by Average Price (min {min_listings} listings)',
    labels={'avg_price': 'Average Price (Cr)', 'society': 'Society'},
    color='avg_price_per_sqft', color_continuous_scale='Purples',
    hover_data=['listings', 'avg_price_per_sqft']
)
fig10.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig10, use_container_width=True)