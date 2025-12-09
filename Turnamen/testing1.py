import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

# 1. Muat dataset
data = pd.read_csv('VCT_2024.csv')

# 2. Tampilkan informasi awal tentang DataFrame
print("Informasi Data Awal:")
print(data.info())

# 3. Mengisi nilai yang hilang dengan rata-rata kolom
for column in data.columns:
    if data[column].isnull().sum() > 0:  # Cek apakah kolom memiliki nilai hilang
        mean_value = data[column].mean()  # Hitung rata-rata kolom
        data[column].fillna(mean_value, inplace=True)  # Isi nilai yang hilang dengan rata-rata

# Tampilkan informasi setelah pengisian nilai yang hilang
print("\nInformasi Data Setelah Pengisian Nilai yang Hilang:")
print(data.info())

# 4. Preprocessing Data
# Normalisasi kolom numerik
num_cols = ['ACS', 'K:D', 'KAST', 'ADR', 'KPR', 'APR', 'FKPR', 'FDPR', 'HS%', 'CL%', 'CW', 'CP']
scaler = MinMaxScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

# Encoding kolom kategorikal
data = pd.get_dummies(data, columns=['Region', 'Event', 'Player', 'Team Abbreviated', 'Team'], drop_first=True)

# 5. Memisahkan fitur dan target
target_column = 'R'  # Ganti dengan kolom yang sesuai jika perlu
X = data.drop(columns=[target_column])
y = data[target_column]

# 6. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 7. Model Training dan Evaluasi
# SVM Model
svm_model = SVC(kernel='linear', C=1, probability=True)
svm_model.fit(X_train, y_train)

# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30]
}
grid_search = GridSearchCV(rf_model, param_grid, cv=3)
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_

# 8. Evaluasi Model
# Prediksi menggunakan SVM
y_pred_svm = svm_model.predict(X_test)

# Prediksi menggunakan Random Forest
y_pred_rf = best_rf_model.predict(X_test)

# Hitung metrik untuk SVM
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm, average='weighted', zero_division=0)
svm_recall = recall_score(y_test, y_pred_svm, average='weighted')
svm_f1 = f1_score(y_test, y_pred_svm, average='weighted')

# Hitung metrik untuk Random Forest
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)
rf_recall = recall_score(y_test, y_pred_rf, average='weighted')
rf_f1 = f1_score(y_test, y_pred_rf, average='weighted')

# Tampilkan hasil evaluasi
results_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'SVM': [svm_accuracy, svm_precision, svm_recall, svm_f1],
    'Random Forest': [rf_accuracy, rf_precision, rf_recall, rf_f1]
})

print("\nHasil Evaluasi Model:")
print(results_df)