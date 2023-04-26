"""
Plan:
1. Get data (already implemented)
2. Initialize arrays for holding counts of each color-intensity
    0. Number of features n should be 12288 for 64*64(pixels)*3(RGB/pixel)
    1. For each pixel-color, need to hold 256 variables, so shape is (12288, 256)
        1. Index by (pixel-color, intensity-level)
        2. Call it intensity_counts?
    2. For each feature, have total number
        1. Index by (pixel-color)
        2. Call it total_counts?
3. Process data into the count arrays
4. For testing, throw into naive bayes using m-intensity
"""


from re import L
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
    imgs_array = []

    # Loop through all the images in the folder
    for filename in os.listdir(folderPath):
        # Load the image file using PIL
        img = Image.open(os.path.join(folderPath, filename))
        
        # Convert the image to a NumPy array and normalize its pixel values
        imgTemp = np.array(img)

        # Flatten the image into a 1D array
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

# Define the NaiveBayes class
class NaiveBayes():
    def __init__(self, input_dim, input_range):
        self.num_dim = input_dim
        self.val_range = input_range
        self.dim_intensity_counts = np.zeros((input_dim, input_range))
        self.dim_intensity_pos_counts = np.zeros((input_dim, input_range))
        self.dim_intensity_neg_counts = np.zeros((input_dim, input_range))
        self.tot_pos = 0
        self.tot_neg = 0

    def predict(self, x):
        # x is an m*n of m examples and n dimensions
        m = x.shape[0]
        n = x.shape[1]

        t = np.zeros(m)

        for i in range(m):
            tot_prob_pos = 1
            tot_prob_neg = 1
            for j in range(len(x[i])):
                tot_prob_pos *= (self.dim_intensity_pos_counts[(j, x[i][j])] + (1/self.val_range)) \
                    / (self.tot_pos + 1)
                tot_prob_neg *= (self.dim_intensity_neg_counts[(j, x[i][j])] + (1/self.val_range)) \
                    / (self.tot_neg + 1)

            if tot_prob_pos > tot_prob_neg:
                t[i] = 1
            else:
                t[i] = 0

        return t

    def train(self, x, t):
        # x is an m*n of m examples and n dimensions
        # t is binary classifiers for m examples
        m = x.shape[0]
        n = x.shape[1]

        if (n != self.num_dim):
            return -1
        if (len(t) != m):
            return -2

        for i in range(m):
            isPos = t[i] > 0

            for j in range(n):
                self.dim_intensity_counts[(j, x[i][j])] += 1

                if isPos:
                    self.dim_intensity_pos_counts[(j, x[i][j])] += 1
                else:
                    self.dim_intensity_neg_counts[(j, x[i][j])] += 1

            if isPos:
                self.tot_pos += 1
            else:
                self.tot_neg += 1
        return 0


# Train the naive bayes method
input_dim = trainSet.shape[1]
nb = NaiveBayes(input_dim, 256)
if (0 != nb.train(trainSet, trainLabels)):
    quit()

# Evaluate the naive bayes method
train_predictions = nb.predict(trainSet)
test_predictions = nb.predict(testSet)

train_accuracy = np.mean(train_predictions == trainLabels)
test_accuracy = np.mean(test_predictions == testLabels)

print('Training accuracy: {:.2f}%'.format(train_accuracy * 100))
print('Testing accuracy: {:.2f}%'.format(test_accuracy * 100))

# Plot the accuracy over epochs
plt.bar(["Training Data", "Testing Data"], [train_accuracy*100, test_accuracy*100])
plt.title('Accuracy of Naive Bayes')
plt.xlabel('Data Set')
plt.ylabel('Prediction Accuracy (%)')
plt.show()
