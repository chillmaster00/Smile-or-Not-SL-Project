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
# Convert y to one-hot encoded vectors
one_hot_y = np.zeros((y.size, 2))
one_hot_y[np.arange(y.size), y.astype(int)] = 1
y = one_hot_y
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


class neuralNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.wH = np.random.randn(input_size, hidden_size)  #weights for hidden layer
        self.bH = np.zeros((1, hidden_size))                #bias for hidden layer
        self.wO = np.random.randn(hidden_size, output_size) #weights for output layer
        self.bO = np.zeros((1, output_size))                #bias for output layer

    #activation function and derivation
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def sigmoidDerivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    # feed forward thru neural net
    def feedForward(self, X):
        #First calc dot product
        dotP = np.dot(X, self.wH) + self.bH         
        #Put dot product thru activation function (sigmoid)
        #also record the output for future backpropagation
        self.oH = self.sigmoid(dotP)                
        #Repeat for output layer
        dotP = np.dot(self.oH, self.wO) + self.bO   
        o = self.sigmoid(dotP)
        return o
    
    # back propagation 
    def backProp(self, x, t, o, eta):
        # First calulate the backpropagation thru the output layer

        sigmaO = (t - o) * self.sigmoidDerivative(o)
        deltaO = np.dot(self.oH.T, sigmaO)              # dot product w/ weights from prev layer
        
        # Next calculate hidden layer backprop
        sigmaH = np.dot(sigmaO, self.wO.T) * self.sigmoidDerivative(self.oH)
        deltaH = np.dot(x.T, sigmaH)

        #update values
        self.wH += eta * deltaH                 #update weights
        self.bH += eta * np.sum(sigmaH, axis=0) #update biases
        self.wO += eta * deltaO
        self.bO += eta * np.sum(sigmaO, axis=0)
        
    def train(self, x, t, epochs, eta):
        accuracyListTrain = []
        accuracyListTest = []
        for epoch in range(epochs):
            # feed foward to receive outputs
            o = self.feedForward(x)

            # backpropagate those outputs
            self.backProp(x, t, o, eta)

            # document accuracy
            predIteration = self.predict(x)
            trainAccuracy = np.mean(t == predIteration) * 100
            accuracyListTrain.append(trainAccuracy)
            predIteration = self.predict(testSet)
            trainAccuracy = np.mean(testLabels == predIteration) * 100
            accuracyListTest.append(trainAccuracy)
        
        return accuracyListTrain, accuracyListTest

    def predict(self, X):

        o = self.feedForward(X)
        return np.where(o > 0.5, 1, 0) 



# Train the perceptron
inputSize = trainSet.shape[1]
hiddenSize = 100
outputSize = 2
eta = 0.05
epochs = 20000

nn = neuralNet(inputSize, hiddenSize, outputSize)
accuracyListTrain, accuracyListTest = nn.train(trainSet, trainLabels, epochs, eta)

# Evaluate the perceptron
predTrain = nn.predict(trainSet) 
trainAcc = np.mean(predTrain == trainLabels)
print('Training accuracy: {:.2f}%'.format(trainAcc * 100))

predTest = nn.predict(testSet)
testAcc = np.mean(predTest == testLabels)
print('Testing accuracy: {:.2f}%'.format(testAcc * 100))

# Plot the accuracy over epochs
plt.plot(accuracyListTrain, label='Training accuracy')
plt.plot(accuracyListTest, label='Testing accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.show()
