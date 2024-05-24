import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define the file path
file_path = 'final_scores.csv'  # Update with the correct file path

# Load the data
data = pd.read_csv(file_path)

# Calculate the moving average
window_size = 10
data['Moving_Average'] = data['Score'].rolling(window=window_size).mean()

# Calculate the exploration rate
initial_exploration_rate = 1.0
decay_rate = 0.99
data['Exploration_Rate'] = initial_exploration_rate * (decay_rate ** data['Episode'])

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Score and Moving Average
ax1.plot(data['Episode'], data['Score'], label='Score', alpha=0.5)
ax1.plot(data['Episode'], data['Moving_Average'], label='Moving Average', linestyle='-', linewidth=2)
ax1.set_xlabel('Episode')
ax1.set_ylabel('Score')
ax1.grid(True)
ax1.legend(loc='upper left')

# Plot Exploration Rate on a secondary y-axis
ax2 = ax1.twinx()
ax2.plot(data['Episode'], data['Exploration_Rate'], label='Exploration Rate', color='red', linestyle='--')
ax2.set_ylabel('Exploration Rate')
ax2.legend(loc='upper right')

plt.title('Score and Exploration Rate over Episodes')
plt.show()

