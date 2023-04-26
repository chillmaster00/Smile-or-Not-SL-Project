from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
import numpy as np

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


# Define the neural network model
neural_network = MLPClassifier(hidden_layer_sizes=(100), activation='logistic')

# Train the neural network
accuracyListTrain = []
accuracyListTest = []
for epoch in range(100):
    neural_network.partial_fit(trainSet, trainLabels, classes=[0, 1])
    
    predTrain = neural_network.predict(trainSet)
    trainAcc = np.mean(predTrain == trainLabels)
    accuracyListTrain.append(trainAcc * 100)
    
    predTest = neural_network.predict(testSet)
    testAcc = np.mean(predTest == testLabels)
    accuracyListTest.append(testAcc * 100)


# Evaluate the neural network
predTrain = neural_network.predict(trainSet)
trainAcc = np.mean(predTrain == trainLabels)
print('Training accuracy: {:.2f}%'.format(trainAcc * 100))

predTest = neural_network.predict(testSet)
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
