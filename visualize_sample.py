import matplotlib.pyplot as plt

def show_sample(X, y):
  plt.imshow(X[0].reshape(28, 28), cmap='gray')
  plt.title(f'Label: {y[0]}')
  plt.show()