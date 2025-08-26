import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("decision_tree_dataset.csv")

# Separate features and target
X = df.drop("Play_Tennis", axis=1)
y = df["Play_Tennis"]

# Encode categorical variables
encoder = LabelEncoder()
for col in X.columns:
    X[col] = encoder.fit_transform(X[col])

y = encoder.fit_transform(y)  # target encoding

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)

# Accuracy
print("Training Accuracy:", dtc.score(x_train, y_train))
print("Testing Accuracy:", dtc.score(x_test, y_test))

 