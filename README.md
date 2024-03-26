# SeismicAI: Earthquake Data Visualization & Analysis App

SeismicAI is an interactive web application built with Streamlit, designed to fetch, analyze, and visualize seismic data from the U.S. Geological Survey (USGS) Earthquake Hazards Program API. It provides users with insights into earthquake data, allowing them to visualize seismic events on a map and interact with the data through a simple chat interface. The app is intended for educational and informational purposes and does not forecast earthquakes.

## Features

- Fetch seismic data from the USGS API based on user-selected date ranges.
- Data analysis and insights, including statistical summaries and trend analysis.
- Visualization of earthquake data on a map, highlighting events with a magnitude greater than 4.0.
- A chat interface that allows users to ask questions about the seismic data, with examples provided.
- Customizable date input for data fetching, enabling users to explore seismic events within specific time frames.

## How to Use

1. **Select Date Range**: Use the sidebar to input the start and end dates for fetching seismic data. The default range is the past seven days.
2. **View Map Visualization**: The main area displays a map visualization of earthquakes with a magnitude greater than 4.0, based on the selected date range.
3. **Read General Summary**: Expand the "General Summary" section to view statistical insights into the fetched seismic data.
4. **Ask Questions**: Use the "Seismic Chat" section in the sidebar to ask specific questions about the seismic data. You can select an example query or type your own question and press "Ask" to submit.

## Technical Details

- **Python Libraries**: The app utilizes `requests` for API requests, `pandas` and `pandasai` for data manipulation, `plotly.express` for data visualization, and `datetime` and `numpy` for handling dates and numerical operations.
- **Data Source**: Seismic data is fetched from the [USGS Earthquake Hazards Program API](https://earthquake.usgs.gov/fdsnws/event/1/).
- **Chat Interface**: Implemented using the `pandasai` library's `SmartDataframe` class and OpenAI's API for natural language processing and interaction.

## Installation and Setup

To run SeismicAI locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/jamesdesantiago/SeismicAI.git
    cd SeismicAI
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your OpenAI API key:
    - Obtain an API key from [OpenAI](https://openai.com/).
    - Create a file named `.streamlit/secrets.toml` and add your OpenAI API key:
        ```toml
        openai_api_key = "your_api_key_here"
        ```

4. Run the app:
    ```bash
    streamlit run app.py
    ```

5. The app will start running locally. Open the provided URL in a web browser to use SeismicAI.

## Contribution Guidelines

Contributions to SeismicAI are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Implement your changes.
4. Submit a pull request with a clear description of your changes.

## License

SeismicAI is released under the Apache-2.0 License. See the `LICENSE` file for more details.

## Credits

SeismicAI was created by [James De Santiago](https://github.com/jamesdesantiago). Data is sourced from the U.S. Geological Survey (USGS) Earthquake Hazards Program.

Please note that this documentation is a generic template and should be customized to fit the specific features and functionalities of your version of SeismicAI.
