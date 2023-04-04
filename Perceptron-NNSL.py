from PIL import Image
import numpy as np
import os

# Define the folder path
smile_path = './data/smile'
not_smile_path = './data/non_smile'
image_size = (64, 64)

def processImgs(folder_path, image_size):
    # Create an empty array to hold the image data
    image_array = np.empty((0, image_size[0] * image_size[1] * 3))
    print('Shape of image array:', image_array.shape)

    # Loop through all the images in the folder
    for filename in os.listdir(folder_path):
        # Load the image file using PIL
        image = Image.open(os.path.join(folder_path, filename))
        
        # Resize the image to the desired size
        image = image.resize(image_size)
        
        # Convert the image to a NumPy array and normalize its pixel values
        image_array_single = np.asarray(image) / 255.0
        
        # Add an extra dimension to the array
        image_array_single = np.expand_dims(image_array_single, axis=0)
        
        # Flatten the image array
        image_array_single = image_array_single.reshape((1, -1))

        # Concatenate the new image array with the existing array of images
        image_array = np.concatenate((image_array, image_array_single), axis=0)

    # return completed array
    return image_array

smiling_images = processImgs(smile_path, image_size)
not_smiling_images = processImgs(not_smile_path, image_size)

print('Shape of image array:', smiling_images.shape)
print('Shape of image array:', not_smiling_images.shape)

# Create the training data
X = np.concatenate((smiling_images, not_smiling_images), axis=0)
y = np.concatenate((np.ones(smiling_images.shape[0]), np.zeros(not_smiling_images.shape[0])))
print(y)
# Shuffle the data
shuffle_idx = np.random.permutation(len(X))
X = X[shuffle_idx]
y = y[shuffle_idx]

# Convert the list of data to a NumPy array
X = np.array(X)
# Split the data into training and testing sets
split_idx = int(len(X) * 0.8)
X_train = X[:split_idx]
y_train = y[:split_idx]
X_test = X[split_idx:]
y_test = y[split_idx:]

# Define the perceptron class
class Perceptron():
    def __init__(self, input_dim):
        self.weights = np.random.randn(input_dim)
    
    def predict(self, X):
        return np.where(np.dot(X, self.weights) > 0, 1, 0)
    
    def train(self, X, y, epochs=10, learning_rate=0.1):
        for epoch in range(epochs):
            for xi, yi in zip(X, y):
                prediction = self.predict(xi)
                self.weights += learning_rate * (yi - prediction) * xi

class NeuralNetwork():
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.hidden_layer = [Perceptron(input_dim) for i in range(hidden_dim)]
        self.output_layer = Perceptron(hidden_dim)
    
    def forward(self, X):
        hidden_outputs = np.array([p.predict(X) for p in self.hidden_layer]).T
        return self.output_layer.predict(hidden_outputs)
    
    def train(self, X, y, epochs=10, learning_rate=0.1):
        for epoch in range(epochs):
            for xi, yi in zip(X, y):
                hidden_outputs = np.array([p.predict(xi) for p in self.hidden_layer]).T
                prediction = self.output_layer.predict(hidden_outputs)
                error = yi - prediction
                for i, p in enumerate(self.hidden_layer):
                    p.weights += learning_rate * error * self.output_layer.weights[i] * xi
                self.output_layer.weights += learning_rate * error * np.array(hidden_outputs)


# Train the perceptron
input_dim = X_train.shape[1]
perceptron = Perceptron(input_dim)
perceptron.train(X_train, y_train)

# Evaluate the perceptron
y_pred_train = perceptron.predict(X_train)
train_accuracy = np.mean(y_pred_train == y_train)
print('Training accuracy: {:.2f}%'.format(train_accuracy * 100))

y_pred_test = perceptron.predict(X_test)
test_accuracy = np.mean(y_pred_test == y_test)
print('Testing accuracy: {:.2f}%'.format(test_accuracy * 100))

# Train the neural network
input_dim = X_train.shape[1]
hidden_dim = 100
output_dim = 1
nn = NeuralNetwork(input_dim, hidden_dim, output_dim)
nn.train(X_train, y_train)

# Evaluate the neural network
y_pred_train = nn.forward(X_train)
train_accuracy = np.mean(y_pred_train == y_train)
print('Training accuracy: {:.2f}%'.format(train_accuracy * 100))

y_pred_test = nn.forward(X_test)
test_accuracy = np.mean(y_pred_test == y_test)
print('Testing accuracy: {:.2f}%'.format(test_accuracy * 100))