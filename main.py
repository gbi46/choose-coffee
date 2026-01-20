from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Beispiel-Daten laden
X, y = load_iris(return_X_y=True)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modell
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print(model.score(X_test, y_test))
