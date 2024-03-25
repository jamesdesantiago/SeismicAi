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
    binned_df = df.groupby(['latbin', 'lonbin']).agg(
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
    title = f'Global Seismic Activity from {start_time} to {end_time}'
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

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    client = OpenAI(api_key=st.secrets['openai_api_key'])
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
