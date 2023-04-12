from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# Define the folder path
smilePath = './data/smile'
nonSmilePath = './data/non_smile'
imgSz = (64, 64)

def processImgs(folderPath, imgSz):
    # Create an empty array to hold the image data
    imgs = np.empty((0, imgSz[0] * imgSz[1] * 3))
    print('Shape of image array:', imgs.shape)

    # Loop through all the images in the folder
    for filename in os.listdir(folderPath):
        # Load the image file using PIL
        img = Image.open(os.path.join(folderPath, filename))
        

        
        # Convert the image to a NumPy array and normalize its pixel values
        imgTemp = np.asarray(img) / 255.0

        # Add an extra dimension to the array
        imgTemp = np.expand_dims(imgTemp, axis=0)

        # Flatten the image array
        imgTemp = imgTemp.reshape((1, -1))


        # Concatenate the new image array with the existing array of images
        imgs = np.concatenate((imgs, imgTemp), axis=0)

    # return completed array
    return imgs

smileImgs = processImgs(smilePath, imgSz)
nonSmileImgs = processImgs(nonSmilePath, imgSz)

print('Shape of image array:', smileImgs.shape)
print('Shape of image array:', nonSmileImgs.shape)

# Create the training data
x = np.concatenate((smileImgs, nonSmileImgs), axis=0)
y = np.concatenate((np.ones(smileImgs.shape[0]), np.zeros(nonSmileImgs.shape[0])))

np.random.seed(123)
# Shuffle the data
shuffleIdx = np.random.permutation(len(x))
shuffledSet = x[shuffleIdx]
shuffledLabels = y[shuffleIdx]

# Split the data into training and testing sets
splitIdx = int(len(shuffledSet) * 0.8)
trainSet = shuffledSet[:splitIdx]
trainLabels = shuffledLabels[:splitIdx]
testSet = shuffledSet[splitIdx:]
testLabels = shuffledLabels[splitIdx:]

# Define the perceptron class
class Perceptron():
    def __init__(self, input_dim):
        self.w = np.zeros(input_dim + 1)
    
    def predict(self, x):

        return np.where(np.dot(x, self.w[1:]) + self.w[0] > 0, 1, 0)
    
    def train(self, x, t, epochs=200, eta=0.01):
        accuracyListTrain = []
        accuracyListTest = []

        for epoch in range(epochs):
            for xi, ti in zip(x, t):

                # Find output of perceptron
                o = self.predict(xi)  # output

                self.w[1:] += eta * (ti - o) * xi
                self.w[0] += eta * (ti - o)


            predIteration = self.predict(x)
            trainAccuracy = np.mean(t == predIteration) * 100
            accuracyListTrain.append(trainAccuracy)
            predIteration = self.predict(testSet)
            trainAccuracy = np.mean(testLabels == predIteration) * 100
            accuracyListTest.append(trainAccuracy)
        
        return accuracyListTrain, accuracyListTest


class NeuralNetwork():
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.hidden_layer = [Perceptron(input_dim) for i in range(hidden_dim)]
        self.output_layer = Perceptron(hidden_dim)
    
    def forward(self, X):
        hidden_outputs = np.array([p.predict(X) for p in self.hidden_layer]).T
        return self.output_layer.predict(hidden_outputs)
    
    def train(self, X, y, epochs=1, learning_rate=0.01):
        for epoch in range(epochs):
            for xi, yi in zip(X, y):
                hidden_outputs = np.array([p.predict(xi) for p in self.hidden_layer]).T
                prediction = self.output_layer.predict(hidden_outputs)
                error = yi - prediction
                for i, p in enumerate(self.hidden_layer):
                    p.w += learning_rate * error * self.output_layer.w[i] * xi
                self.output_layer.w += learning_rate * error * hidden_outputs


# Train the perceptron
input_dim = trainSet.shape[1]
perceptron = Perceptron(input_dim)
accuracyListTrain, accuracyListTest = perceptron.train(trainSet, trainLabels)

# Evaluate the perceptron
y_pred_train = perceptron.predict(trainSet) 
train_accuracy = np.mean(y_pred_train == trainLabels)
print('Training accuracy: {:.2f}%'.format(train_accuracy * 100))

y_pred_test = perceptron.predict(testSet)
test_accuracy = np.mean(y_pred_test == testLabels)
print('Testing accuracy: {:.2f}%'.format(test_accuracy * 100))

# Plot the accuracy over epochs
plt.plot(accuracyListTrain, label='Training accuracy')
plt.plot(accuracyListTest, label='Testing accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.show()


# # Train the neural network
# input_dim = trainSet.shape[1]
# hidden_dim = 1
# output_dim = 1
# nn = NeuralNetwork(input_dim, hidden_dim, output_dim)
# nn.train(trainSet, trainLabels)

# # Evaluate the neural network
# y_pred_train = nn.forward(trainSet)
# train_accuracy = np.mean(y_pred_train == trainLabels)
# print('Training accuracy: {:.2f}%'.format(train_accuracy * 100))

# y_pred_test = nn.forward(testSet)
# test_accuracy = np.mean(y_pred_test == testLabels)
# print('Testing accuracy: {:.2f}%'.format(test_accuracy * 100))