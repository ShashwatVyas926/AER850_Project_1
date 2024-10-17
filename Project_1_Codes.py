#Data Processing
import pandas as pd
Project_1_data = 'Project_1_Data.csv'
df = pd.read_csv(Project_1_data)
print(df)

#Data Visualization
import matplotlib.pyplot as plt
import numpy as np

print("\Summary statistics:")
print(df.describe())

df.hist(figsize=(10,7), bins=25, grid=True)
plt.suptitle('Histograms of X, Y, Z')
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(df['X'], df['Y'], df['Z'], c=df['Step'], cmap='viridis', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot of Dataset')
legend = fig.colorbar(scatter, ax=ax, label='Step')
plt.show()

#Correlation Analysis
import seaborn as sns

correlation_matrix = df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Correlation Matrix of X,Y,Z with Target Variable (Step)')
plt.show()