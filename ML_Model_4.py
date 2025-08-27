from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# Split data
df=pd.read_csv("random_forest_dataset .csv")
X = df.drop("approved", axis=1)
y = df["approved"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(accuracy, report, conf_matrix)
def predict_approval(age, income, loan_amount, savings, credit_score, years_employed, debt_ratio, num_dependents):
    # Create dataframe with input
    input_data = pd.DataFrame([[age, income, loan_amount, savings, credit_score, years_employed, debt_ratio, num_dependents]], 
                              columns=X.columns)
    
    # Prediction
    prediction = rf.predict(input_data)[0]
    proba = rf.predict_proba(input_data)[0]
    
    return "Approved ✅" if prediction == 1 else "Not Approved ❌", proba

# Example Prediction
result, probabilities = predict_approval(30, 60000, 15000, 5000, 700, 5, 0.2, 2)
print(result, probabilities)
