import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file from the Results folder
file_path = 'Results/Results_data_1_4ghosts.csv'  # Replace with your actual file name
data = pd.read_csv(file_path)

# Calculate the moving average (you can adjust the window size as needed)
window_size = 10
data['moving_average'] = data['Episode Scores'].rolling(window=window_size).mean()

# Plot the scores with respect to the episode number
plt.figure(figsize=(10, 6))
plt.plot(data.index + 1, data['Episode Scores'], label='Episode Scores')
plt.plot(data.index + 1, data['moving_average'], label=f'{window_size}-Episode Moving Average', linewidth=2)
plt.xlabel('Episode Number')
plt.ylabel('Episode Scores')
plt.title('Episode Scores vs Episode Number')
plt.legend()
plt.grid(True)
plt.show()

# Calculate the moving average (you can adjust the window size as needed)
window_size = 10
data['moving_average'] = data['Episode Scores'].rolling(window=window_size).mean()

# Plot the scores with respect to the episode number
plt.figure(figsize=(10, 6))
plt.plot(data.index + 1, data['Episode Scores'], label='Episode Scores')
plt.plot(data.index + 1, data['moving_average'], label=f'{window_size}-Episode Moving Average', linewidth=2)
plt.xlabel('Episode Number')
plt.ylabel('Episode Scores')
plt.title('Episode Scores vs Episode Number')
plt.legend()
plt.grid(True)
plt.show()
