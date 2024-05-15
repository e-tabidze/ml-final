import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load network traffic data
data = pd.read_csv("network_traffic_data.csv")

# Drop non-numeric columns or encode them properly
data_numeric = data.select_dtypes(include=['number'])

# Calculate correlation matrix
corr_matrix = data_numeric.corr()

# Plot heatmap of correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Network Traffic Data')
plt.show()
