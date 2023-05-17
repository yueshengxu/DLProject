import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pred = np.load("pred.npy").squeeze()
true = np.load("true.npy").squeeze()

df = pd.read_csv('maml.csv')

def calculate_hourly_error(pred, true, hour):
    # Convert hour to the corresponding index in the arrays
    index = hour * 12  # Assuming each hour has 12 data points (5-minute resolution)

    # Select the data for the specified hour
    pred_hour = pred[index:index + 12]  # Assuming each hour has 12 data points (5-minute resolution)
    true_hour = true[index:index + 12]  # Assuming each hour has 12 data points (5-minute resolution)

    # Calculate the absolute error
    error = np.abs(pred_hour - true_hour)

    # Calculate the average absolute error
    avg_error = np.mean(error)

    return avg_error

maml = df.mean(axis=0)
print(maml)
hours = np.arange(24)  # Array representing the 24 hours of the day
errors = [calculate_hourly_error(pred, true, hour) for hour in hours]

# Plotting the hourly errors
plt.plot(hours, errors, label="STGCN")
plt.plot(hours, maml, label="MAML+STGCN")
plt.xlabel('Hour of the day')
plt.ylabel('Absolute Error')
plt.title('Hourly Error')
plt.grid(True)
plt.legend()
plt.show()