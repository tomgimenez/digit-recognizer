from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import time
import joblib

# from visualize_sample import show_sample
# from visualize_confusion_matrix import show_confusion_matrix

# Complete MNIST (70,000 images of 28x28 = 784 pixels)
print('Downloading MNIST')
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

print(f'Shape of X: {X.shape}') # (70000, 784)
print(f'Shape of y: {y.shape}') # (70000,)

# Let's see an example to confirm that's OK
# show_sample(X, y)

# Normalizar: los píxeles van de 0-255, los llevamos a 0-1
X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42, stratify=y
)

print(f'Train: {X_train.shape[0]} samples')
print(f'Test: {X_test.shape[0]} samples')

model = LogisticRegression(
  solver='lbfgs',
  max_iter=200,
  verbose=1
)

print('Training...')
start = time.time()
model.fit(X_train, y_train)
print(f'Training in {time.time() - start:.1f}s')

y_pred = model.predict(X_test)
print(f'\nAccuracy: {accuracy_score(y_test, y_pred):.4f}')
print('\n', classification_report(y_test, y_pred))

# Matrix Visualization
# show_confusion_matrix(y_test, y_pred)

joblib.dump(model, 'digit_model.joblib')
joblib.dump((X_test, y_test), 'test_data.joblib')
print('Model saved')