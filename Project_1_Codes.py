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
plt.suptitle('Histograms of X, Y, Z, & Step')
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

#Classification Model Development/Engineering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

X = df.drop(columns=["Step"])
y = df["Step"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled.shape, X_test_scaled.shape, y_train.shape, y_test.shape

# Model 1: Random Forest with GridSearchCV
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(estimator=rf, param_grid=rf_params, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train_scaled, y_train)

# Model 2: Support Vector Classifier (SVC) with GridSearchCV
svc_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
svc = SVC(random_state=42)
svc_grid = GridSearchCV(estimator=svc, param_grid=svc_params, cv=5, scoring='accuracy', n_jobs=-1)
svc_grid.fit(X_train_scaled, y_train)

# Model 3: K-Nearest Neighbors (KNN) with GridSearchCV
knn_params = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
knn = KNeighborsClassifier()
knn_grid = GridSearchCV(estimator=knn, param_grid=knn_params, cv=5, scoring='accuracy', n_jobs=-1)
knn_grid.fit(X_train_scaled, y_train)

print("Best parameters for Random Forest:", rf_grid.best_params_)
print("Best parameters for SVC:", svc_grid.best_params_)
print("Best parameters for KNN:", knn_grid.best_params_)

#Model Performance Analysis
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

rf_pred = rf_grid.best_estimator_.predict(X_test_scaled)
svc_pred = svc_grid.best_estimator_.predict(X_test_scaled)
knn_pred = knn_grid.best_estimator_.predict(X_test_scaled)

def evaluate_model(y_test, y_pred, model_name):
    print(f"Performance of {model_name}:")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    return accuracy, precision, f1

print("Random Forest Classifier:")
evaluate_model(y_test, rf_pred, "Random Forest")
print("\nSupport Vector Classifier (SVC):")
evaluate_model(y_test, svc_pred, "SVC")
print("\nK-Nearest Neighbors (KNN):")
evaluate_model(y_test, knn_pred, "KNN")