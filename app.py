import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

# Define the function to fetch seismic data
def fetch_seismic_data(start_time, end_time):
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        'format': 'geojson',
        'starttime': start_time,
        'endtime': end_time
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        st.error(f"Failed to get data: {response.content}")
        return None
    data = response.json()
    earthquakes = data['features']
    earthquake_list = []
    for earthquake in earthquakes:
        properties = earthquake['properties']
        geometry = earthquake['geometry']
        mag = properties['mag']
        place = properties['place']
        time = properties['time']
        time_converted = datetime.utcfromtimestamp(time / 1000).strftime('%Y-%m-%d %H:%M:%S')
        lon = geometry['coordinates'][0]
        lat = geometry['coordinates'][1]
        earthquake_list.append([mag, place, time_converted, lat, lon])
    df = pd.DataFrame(earthquake_list, columns=['magnitude', 'Place', 'date_time', 'latitude', 'longitude'])
    df['magnitude'] = df['magnitude'].apply(lambda x: max(0, x))
    return df

# Define the function to bin data
def bin_data(df, lat_bins, lon_bins):
    df['latbin'] = pd.cut(df['latitude'], bins=np.linspace(min(df['latitude']), max(df['latitude']), lat_bins))
    df['lonbin'] = pd.cut(df['longitude'], bins=np.linspace(min(df['longitude']), max(df['longitude']), lon_bins))
    binned_df = df.groupby(['latbin', 'lonbin'], observed=False).agg(
        latitude=pd.NamedAgg(column='latitude', aggfunc='median'),
        longitude=pd.NamedAgg(column='longitude', aggfunc='median'),
        magnitude=pd.NamedAgg(column='magnitude', aggfunc='median'),
        most_recent_date=pd.NamedAgg(column='date_time', aggfunc='max')
    ).reset_index()
    return binned_df

# Define the function to plot earthquake data
def simple_plot_earthquake_data(df, start_time, end_time):
    df = df.dropna(subset=['magnitude'])
    custom_color_scale = ['#2C0F06', '#67260F', '#7B3413', '#E6B663', '#F0E0C6']
    title = "___"
    fig = px.scatter_geo(df,
                         lat='latitude',
                         lon='longitude',
                         color='magnitude',
                         title=title,
                         color_continuous_scale=custom_color_scale,
                         size='magnitude',
                         size_max=10,
                         projection='orthographic',
                         hover_data=['most_recent_date'])
    st.plotly_chart(fig)

def summarize_df_for_chat(df):
    if df.empty:
        return "The dataset is currently empty."

    # Basic statistics
    total_events = len(df)
    min_magnitude = df['magnitude'].min()
    max_magnitude = df['magnitude'].max()
    avg_magnitude = df['magnitude'].mean()

    # Geographical spread
    min_latitude, max_latitude = df['latitude'].min(), df['latitude'].max()
    min_longitude, max_longitude = df['longitude'].min(), df['longitude'].max()

    # Trend analysis over time - simple approach considering linear trends
    df_sorted = df.sort_values(by='date_time')
    df.loc[:, 'date'] = pd.to_datetime(df['date_time']).dt.date
    daily_counts = df.groupby('date').size()
    trend_direction = "increasing" if daily_counts.iloc[-1] - daily_counts.iloc[0] > 0 else "decreasing"
    
    # Magnitude distribution
    magnitude_distribution = df['magnitude'].value_counts(bins=5).sort_index()

    # Assembling the summary
    summary = (
        f"The dataset contains {total_events} seismic events. "
        f"Magnitudes range from {min_magnitude} to {max_magnitude}, with an average magnitude of {avg_magnitude:.2f}. "
        f"Geographically, the events span from latitude {min_latitude} to {max_latitude} and longitude {min_longitude} to {max_longitude}. "
        f"The number of daily seismic events is {trend_direction}. "
        "Magnitude distribution: "
    )

    # Adding magnitude distribution to the summary
    for interval, count in magnitude_distribution.items():
        summary += f"\n - {interval}: {count} events"

    return summary

llm = OpenAI(api_token=st.secrets["openai_api_key"])

# Initialize chat and data
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    df = None  # Placeholder for your dataframe

# Streamlit UI
st.title("SeismicAI")

st.expander("About this app", expanded=True)
st.write(
    "This app uses the U.S. Geological Survey (USGS) Earthquake Hazards Program API to fetch seismic data. "
    "The data is then analyzed using PandasAI to answer questions and provide insights. "
    "You can ask questions about the data in the chat window and visualize the earthquake data on the map."
    "Note: SeismicAI does not forecast earthquakes. It is strictly for educational purposes."
)

# Calculate 1 week ago from today
one_week_ago = datetime.now() - timedelta(days=7)

# Set today's date
today = datetime.now()

col1, col2 = st.columns(2)
with col1:
    start_time = st.date_input("Start Date", value=one_week_ago)

with col2:
    end_time = st.date_input("End Date", value=today)

if start_time and end_time:
    df = fetch_seismic_data(start_time.strftime('%Y-%m-%dT%H:%M:%S'), end_time.strftime('%Y-%m-%dT%H:%M:%S'))
    # Initiate pandasai instance
    query_engine = SmartDataframe(df, config={"llm": llm})

    if df is not None:
        filtered_df = df[df['magnitude'] > 4]
        binned_df = bin_data(filtered_df, 50, 50)
        simple_plot_earthquake_data(binned_df, start_time.strftime('%Y-%m-%d'), end_time.strftime('%Y-%m-%d'))
    
# Update chat history with dataframe summary for the first interaction
if not st.session_state.chat_history:
    df_summary = summarize_df_for_chat(df) if df is not None else "Data is not available."

st.expander("General Summary", expanded=True)
st.write(df_summary)

# Chat input
prompt = st.text_area("Ask me about the seismic data...")

# Generate output
if st.button("Ask a question"):
    if prompt:
        # call pandas_ai.run(), passing dataframe and prompt
        with st.spinner("Generating response..."):
            st.caption("Example prompts: 'What is the average magnitude of the earthquakes?', 'What are the top 10 places with largest earthquakes?', 'What country had the most earthquakes?', 'Chart the number of earthquakes by places', 'What is the distribution of magnitudes?', 'Where did the most recent earthquake occur?'")
            st.write(query_engine.chat(prompt))
    else:
        st.warning("Please enter a prompt.")

st.caption("Data source: U.S. Geological Survey (USGS) Earthquake Hazards Program")
