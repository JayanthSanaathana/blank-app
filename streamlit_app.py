import streamlit as st
import yfinance as yf
from datetime import date
import pandas as pd
import plotly.graph_objs as go
from prophet import Prophet

from prophet.plot import plot_plotly

# Define constants
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Streamlit App Title
st.title('Stock Forecast App')

# Dropdown for stock selection
stocks = ('RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

# Slider for selecting prediction period
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache_data
def load_data(ticker):
    """Load stock data from Yahoo Finance and clean column names."""
    try:
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)

        # Flatten multi-level columns and clean up column names
        data.columns = [col[0].lower().replace(' ', '_') if isinstance(col, tuple) else col.lower().replace(' ', '_')
                        for col in data.columns]

        # Remove ticker suffixes like "_infy.ns" from column names
        data.columns = [col.split('_')[0] if '_' in col else col for col in data.columns]

        return data
    except Exception as e:
        st.error(f"Failed to load data for {ticker}. Error: {e}")
        return pd.DataFrame()


# Load data
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

if data.empty:
    st.error("No data available for the selected stock and period.")
else:
    st.subheader('Raw data')
    st.write(data.tail())


    # Plot raw data
    def plot_raw_data():
        """Plot the raw stock data."""
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['date'], y=data['open'], name="stock_open"))
            fig.add_trace(go.Scatter(x=data['date'], y=data['close'], name="stock_close"))
            fig.layout.update(
                title_text='Time Series data with Rangeslider',
                xaxis_rangeslider_visible=True,
                xaxis_title="Date",
                yaxis_title="Stock Price"
            )
            st.plotly_chart(fig)
        except KeyError as e:
            st.error(f"Error in plotting data. Missing column: {e}")


    plot_raw_data()



# Prepare data for Prophet
if 'date' in data.columns and 'close' in data.columns:
    df_train = data[['date', 'close']].copy()
    df_train = df_train.rename(columns={"date": "ds", "close": "y"})
else:
    st.error("Required columns 'Date' and 'Close' are missing from the data. Check the dataset.")
    st.stop()

# Debugging step: Check df_train after renaming
st.write("Debug: df_train after renaming columns", df_train.head())

# Ensure 'y' column is numeric and handle missing or invalid values
df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')  # Convert invalid values to NaN
df_train = df_train.dropna(subset=['y'])  # Drop rows with NaN values in 'y'

# Ensure 'ds' column is a datetime object
df_train['ds'] = pd.to_datetime(df_train['ds'], errors='coerce')  # Convert invalid dates to NaT
df_train = df_train.dropna(subset=['ds'])  # Drop rows with invalid dates
df_train['ds'] = pd.to_datetime(df_train['ds'], errors='coerce').dt.tz_localize(None)

# Debugging step: Check the 'ds' column after removing timezone
st.write("Debug: df_train after removing timezone", df_train.head())


m = Prophet()
m.fit(df_train)

# Make future predictions
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

# Plot forecast
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)


st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)