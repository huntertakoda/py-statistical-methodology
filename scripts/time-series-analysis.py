import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# load the preprocessed dataset

file_path = "C:/puredata/statistical_methodology_preprocessed.csv"
data = pd.read_csv(file_path)

# convert 'age' to a categorical year-like format (for demonstration)

data['Year'] = data['age'] + 2000  # pseudo-year data for analysis

# prepare the data for time series analysis

time_series_data = data.groupby('Year')['mental_health_score'].mean()

# visualize the time series data

plt.figure(figsize=(10, 5))
plt.plot(time_series_data, label='Mental Health Score Over Time')
plt.title('Time Series of Mental Health Scores')
plt.xlabel('Year')
plt.ylabel('Mental Health Score')
plt.legend()
plt.grid()
plt.savefig("C:/puredata/mental_health_time_series_plot.png")
plt.show()

# decompose the time series

decomposition = seasonal_decompose(time_series_data, model='additive', period=5)
decomposition.plot()
plt.savefig("C:/puredata/mental_health_time_series_decomposition.png")
plt.show()

# fit an ARIMA model

arima_model = ARIMA(time_series_data, order=(1, 1, 1))
arima_result = arima_model.fit()

# print and save ARIMA model summary

arima_summary = arima_result.summary()
print(arima_summary)
output_path = "C:/puredata/statistical_methodology_arima_summary.txt"
with open(output_path, "w") as f:
    f.write(str(arima_summary))

# forecast future values

last_year = int(time_series_data.index[-1])  # ensure integer 4 range
forecast = arima_result.forecast(steps=5)
forecast_df = pd.DataFrame({
    "Year": range(last_year + 1, last_year + 6),
    "Forecasted Mental Health Score": forecast
})
forecast_path = "C:/puredata/statistical_methodology_arima_forecast.csv"
forecast_df.to_csv(forecast_path, index=False)

print(f"Time series analysis complete. Results saved to {output_path} and forecast saved to {forecast_path}.")
