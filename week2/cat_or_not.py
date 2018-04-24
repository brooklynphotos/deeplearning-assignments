import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  s = sigmoid(x)
  return s * (1-s)

def image2vector(image):
  return image.reshape(image.size, 1)

def normalizeRows(x):
  x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
  return x / x_norm

if __name__ == "__main__":
  print("Ah")
  print(sigmoid(2))
  print(sigmoid_derivative(2))
