from PIL import Image
import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt

# Define the folder path
smilePath = './data/smile'
nonSmilePath = './data/non_smile'
imgSz = (64, 64)

def processImgs(folderPath, imgSz):
    # Create an empty array to hold the image data
    imgs_array = []

    # Loop through all the images in the folder
    for filename in os.listdir(folderPath):
        # Load the image file using PIL
        img = Image.open(os.path.join(folderPath, filename))
        
        # Convert the image to a NumPy array and normalize its pixel values
        imgTemp = np.array(img)

        # Add an extra dimension to the array
        imgTemp = imgTemp.flatten()

        # Concatenate the new image array with the existing array of images
        imgs_array.append(imgTemp)

    # return completed array
    imgs = np.array(imgs_array)
    return imgs

smileImgs = processImgs(smilePath, imgSz)
nonSmileImgs = processImgs(nonSmilePath, imgSz)

print('Shape of image array:', smileImgs.shape)
print('Shape of image array:', nonSmileImgs.shape)

# Create the training data
x = np.concatenate((smileImgs, nonSmileImgs), axis=0)
y = np.concatenate((np.ones(smileImgs.shape[0]), -1*np.ones(nonSmileImgs.shape[0])))

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
class LogisticRegression():
    def __init__(self, input_dim):
        self.w = np.zeros(input_dim + 1)
    
    def predict(self, x):
        sum = np.dot(x, self.w[1:]) + self.w[0]
        sig = self.sigmoid(sum)
        return np.where(sig >= 0.5, 1, -1)
    
    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def train(self, x, t, epochs=200, eta=0.01):
        accuracyListTrain = []
        accuracyListTest = []

        for epoch in range(epochs):
            
            # 1. Calculate the gradient
            m = x.shape[0] # Number of rows
            n = x.shape[1] # Number of dimensions

            gradient = np.zeros(n + 1) # intialize gradient to 0
            for i in range(m):
                numerator = t[i] * x[i]
                denominator = 1 + math.exp(t[i] * (np.dot(self.w[1:], x[i]) + self.w[0]))
                gradient += numerator / denominator
            gradient = -1/m * gradient

            # 2. Update the weights
            self.w -= eta*gradient / np.linalg.norm(gradient)
            
            """
            w_inc = np.zeros(len(self.w))
            for xi, ti in zip(x, t):
                o = self.predict(xi)

                w_inc[1:] += eta * (ti - o) * xi
                w_inc[0] += eta * (ti - o)

            self.w += w_inc
            """


            # Get prediction rates after training
            predIteration = self.predict(x)
            trainAccuracy = np.mean(t == predIteration) * 100
            accuracyListTrain.append(trainAccuracy)
            predIteration = self.predict(testSet)
            trainAccuracy = np.mean(testLabels == predIteration) * 100
            accuracyListTest.append(trainAccuracy)
        
        return accuracyListTrain, accuracyListTest


# Train the logistic regression method
input_dim = trainSet.shape[1]
regr = LogisticRegression(input_dim)
accuracyListTrain, accuracyListTest = regr.train(trainSet, trainLabels)

# Evaluate the logistic regression method
y_regr_train = regr.predict(trainSet) 
train_accuracy = np.mean(y_regr_train == trainLabels)
print('Training accuracy: {:.2f}%'.format(train_accuracy * 100))

y_pred_test = regr.predict(testSet)
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
