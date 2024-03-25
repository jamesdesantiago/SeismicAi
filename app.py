import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np
from openai import OpenAI

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
    unique_places = df['Place'].nunique()
    
    # Magnitude analysis
    min_magnitude, max_magnitude, mean_magnitude, std_dev_magnitude = df['magnitude'].min(), df['magnitude'].max(), df['magnitude'].mean(), df['magnitude'].std()
    magnitude_quantiles = df['magnitude'].quantile([0.25, 0.5, 0.75]).to_dict()
    
    # Example places to mention
    example_places = df['Place'].drop_duplicates().sample(min(5, unique_places)).tolist()
    example_places_str = ", ".join(example_places)
    
    # Temporal trends analysis
    df_sorted = df.sort_values(by='date_time')
    df.loc[:, 'date'] = pd.to_datetime(df['date_time']).dt.date
    start_date, end_date = df['date'].min(), df['date'].max()
    
    # Monthly distribution of events
    monthly_counts = df.groupby(df['date'].dt.to_period("M")).size()
    most_active_month = monthly_counts.idxmax().strftime('%B %Y')
    least_active_month = monthly_counts.idxmin().strftime('%B %Y')
    
    # Geographic diversity
    northernmost, southernmost = df.loc[df['latitude'].idxmax(), 'Place'], df.loc[df['latitude'].idxmin(), 'Place']
    easternmost, westernmost = df.loc[df['longitude'].idxmax(), 'Place'], df.loc[df['longitude'].idxmin(), 'Place']
    
    # Assembling the summary
    summary = (
        f"The dataset contains a total of {total_events} seismic events spanning from {start_date} to {end_date}, "
        f"across {unique_places} unique locations, including {example_places_str}. "
        f"\nMagnitude analysis reveals a range from {min_magnitude} to {max_magnitude} with a mean magnitude of {mean_magnitude:.2f} and a standard deviation of {std_dev_magnitude:.2f}. "
        f"Quantile distribution is as follows: 25% quantile at {magnitude_quantiles[0.25]}, 50% (median) at {magnitude_quantiles[0.5]}, and 75% quantile at {magnitude_quantiles[0.75]}."
        f"\nTemporal analysis indicates varied activity over time, with {most_active_month} being the month with the highest number of seismic events, and {least_active_month} the least. "
        f"\nGeographically, the events showcase significant diversity, occurring from the northernmost location of {northernmost} to the southernmost point of {southernmost}, "
        f"and from the easternmost location of {easternmost} to the westernmost point of {westernmost}."
    )

    return summary

client = OpenAI(api_key=st.secrets["openai_api_key"])

# Initialize chat and data
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    df = None  # Placeholder for your dataframe

# Streamlit UI
st.title("Seismic AI")

col1, col2 = st.columns(2)
with col1:
    start_time = st.date_input("Start Date", value=pd.to_datetime("2023-09-01"))

with col2:
    end_time = st.date_input("End Date", value=pd.to_datetime("2023-09-13"))

if start_time and end_time:
    df = fetch_seismic_data(start_time.strftime('%Y-%m-%dT%H:%M:%S'), end_time.strftime('%Y-%m-%dT%H:%M:%S'))

    if df is not None:
        filtered_df = df[df['magnitude'] > 4]
        binned_df = bin_data(filtered_df, 50, 50)
        simple_plot_earthquake_data(binned_df, start_time.strftime('%Y-%m-%d'), end_time.strftime('%Y-%m-%d'))

with st.sidebar:
    # Update chat history with dataframe summary for the first interaction
    if not st.session_state.chat_history:
        df_summary = summarize_df_for_chat(df) if df is not None else "Data is not available."
        st.session_state.chat_history.append({"role": "system", "content": df_summary})

    # Display chat history
    for message in st.session_state.chat_history:
        st.chat_message(message["role"]).write(message["content"])

    # Chat input
    user_input = st.chat_input("Ask me anything about the seismic data...")

    if user_input:
        # Update chat history with user input
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Call OpenAI API with the current chat history including the dataframe summary
        # Assume 'client' is already initialized with your OpenAI API key
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.chat_history]
        )
        
        # Extract response and update chat history
        ai_response = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})