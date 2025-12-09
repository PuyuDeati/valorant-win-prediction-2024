import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load data
data = pd.read_csv('VCT_2024.csv')

# Drop irrelevant columns
drop_cols = ['Region', 'Player', 'Team Abbreviated', 'Event', 'CL']
data = data.drop(columns=drop_cols)

# Define numeric columns (including ACS)
numeric_cols = ['ACS', 'K:D', 'KAST', 'ADR', 'KPR', 'APR', 'FKPR', 'FDPR', 'HS%', 'CL%', 'CW', 'CP']

# --- Define Binary Target (Win=1, Loss=0) ---
data['Win'] = (data['ACS'] >= data['ACS'].median()).astype(int)
y = data['Win']
X = data.drop(columns=['Win'])

# --- Preprocessing ---
# 1. Encode categorical variables (Team)
X = pd.get_dummies(X, columns=['Team'], drop_first=True)

# 2. Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')  # Replace NaNs with mean
X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

# 3. Normalize numeric features
scaler = MinMaxScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# 4. Final check for NaNs (should print 0)
print("Remaining NaNs:", X.isna().sum().sum())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Train SVM ---
# Using a pipeline to ensure no data leakage
svm_pipeline = make_pipeline(
    SVC(kernel='rbf', C=1.0, probability=True)
)
svm_pipeline.fit(X_train, y_train)

# Evaluate
y_pred = svm_pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))