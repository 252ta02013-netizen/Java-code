import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("news.csv")

# Expected columns: text and label
X = data["text"]
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Convert text into numbers
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Test model
predictions = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy * 100, "%")

# Check new news
while True:
    news = input("\nEnter news article (or type exit): ")

    if news.lower() == "exit":
        break

    news_vector = vectorizer.transform([news])
    result = model.predict(news_vector)[0]

    print("Prediction:", result)
