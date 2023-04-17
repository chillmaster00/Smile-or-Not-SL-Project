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
        return x * (1 - x)
    
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
        sigmaH = np.dot(sigmaO, self.wO) * self.sigmoidDerivative(self.oH)
        deltaH = np.dot(x.T, sigmaH)

        #update values
        self.wH += eta * deltaH                 #update weights
        self.bH += eta * np.sum(sigmaH, axis=0) #update biases
        self.wO += eta * deltaO
        self.bO += eta * np.sum(sigmaO, axis=0)
        
    def train(self, x, t, epochs, eta):
        for epoch in range(epochs):
            # feed foward to receive outputs
            o = self.feedForward(x)

            # backpropagate those outputs
            self.backProp(x, t, o, eta)

            if epoch % 1000 == 0:
                # displays MSE every 1000 epochs
                loss = np.mean(np.square(t - o))
                print("Epoch:", epoch, "Loss:", loss)

    def print_weights(self):
        print("Weights of hidden layer:", self.wH)


# Train the perceptron
inputSize = trainSet.shape[1]
hiddenSize = 10
outputSize = 1
eta = 0.05
epochs = 100

nn = neuralNet(inputSize, hiddenSize, outputSize)
nn.train(trainSet, trainLabels, epochs, eta)

# Evaluate the neural network
y_pred_train = nn.feedForward(trainSet)
print(y_pred_train)
train_accuracy = np.mean(y_pred_train == trainLabels)
print('Training accuracy: {:.2f}%'.format(train_accuracy * 100))

y_pred_test = nn.forward(testSet)
test_accuracy = np.mean(y_pred_test == testLabels)
print('Testing accuracy: {:.2f}%'.format(test_accuracy * 100))