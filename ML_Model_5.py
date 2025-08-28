# Naïve Bayes and SVM Model Implementation
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------ Naïve Bayes ------------------
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

print("🔹 Naïve Bayes Results:")
print("Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred, target_names=iris.target_names))

# ------------------ Support Vector Machine ------------------
svm_model = SVC(kernel='linear')  # you can try 'rbf', 'poly' too
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

print("\n🔹 Support Vector Machine Results:")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred, target_names=iris.target_names))