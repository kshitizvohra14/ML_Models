import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("logistic_astrology_dataset.csv")

# Features and target
X = df.drop("Zodiac", axis=1)
y = df["Zodiac"]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Balance classes with SMOTE
smote = SMOTE(random_state=42, sampling_strategy="auto")
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'C': [0.1, 1, 5, 10],
    'solver': ['lbfgs', 'saga'],
    'penalty': ['l2'],
    'max_iter': [500, 1000, 2000]
}

log_reg = LogisticRegression(multi_class="multinomial")
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)

print("Best Parameters:", grid_search.best_params_)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n")
#user input
print("\nEnter your astrology traits (values 1-10):")
adventurousness = int(input("Adventurousness: "))
emotionality = int(input("Emotionality: "))
analytical = int(input("Analytical: "))
sociability = int(input("Sociability: "))
discipline = int(input("Discipline: "))
user_features = [[adventurousness, emotionality, analytical, sociability, discipline]]
user_features_scaled = scaler.transform(user_features)
predicted_zodiac = le.inverse_transform(best_model.predict(user_features_scaled))
print(f"\nPredicted Zodiac Sign for your traits: {predicted_zodiac[0]}")
