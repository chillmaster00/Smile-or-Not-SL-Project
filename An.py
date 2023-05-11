from PIL import Image
import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt

# Constants
SMILE_PATH = './data/smile'
NON_SMILE_PATH = './data/non_smile'
IMAGE_SIZE = (64, 64)

NUM_TRIALS = 30
NUM_EPOCHS = 300

# Read a Folder of Images
#   and return one array of 1D arrays representing
#   R, G, B values, and return number of images
def process_images(folder_path):
    # Create an empty array to hold the image data
    imgs_array = []
    print(f"Processing {folder_path}")

    # Loop through all the images in the folder
    for filename in os.listdir(folder_path):
        print(f"\tProccessing image {filename}")

        # Load the image file using PIL
        img = Image.open(os.path.join(folder_path, filename))
        
        # Convert the image to a NumPy array
        img_temp = np.array(img)

        # Add an extra dimension to the array
        img_temp = img_temp.flatten()

        # Concatenate the new image array with the existing array of images
        imgs_array.append(img_temp)

    # return completed array
    imgs = np.array(imgs_array)

    return imgs


# Read a Folder of Images
#   and return one array of 1D arrays representing
#   'intensity ranking' of 4 adjacent pixels
#   from 0 for lowest value to 4 for highest value
def process_images_novel(folder_path):
    # Create an empty array to hold the image data
    imgs_array = []
    print(f"Processing {folder_path}")

    # Loop through all the images in the folder
    for filename in os.listdir(folder_path):
        print(f"\tProccessing image {filename}")

        # Load the image file using PIL
        img = Image.open(os.path.join(folder_path, filename))
        
        # Convert the image to a NumPy array
        img_temp = np.array(img)

        # Get the pixel's "rank"
        img_rank = np.zeros(img_temp.shape, dtype=img_temp.dtype)
        for x in range(IMAGE_SIZE[0]):
            for y in range (IMAGE_SIZE[1]):
                # get the RGB values for the current pixel
                r, g, b = img_temp[x, y, 0], img_temp[x, y, 1], img_temp[x, y, 2]
                pixels_to_sort_R = []
                pixels_to_sort_R.append(r)
                pixels_to_sort_G = []
                pixels_to_sort_G.append(g)
                pixels_to_sort_B = []
                pixels_to_sort_B.append(b)

                # get the RGB values for the adjacent pixels, ignoring invalid pixels
                if (x > 0):
                    r, g, b = img_temp[x-1, y, 0], img_temp[x-1, y, 1], img_temp[x-1, y, 2]
                    pixels_to_sort_R.append(r)
                    pixels_to_sort_G.append(g)
                    pixels_to_sort_B.append(b)
                if (x < IMAGE_SIZE[0]-1):
                    r, g, b = img_temp[x+1, y, 0], img_temp[x+1, y, 1], img_temp[x+1, y, 2]
                    pixels_to_sort_R.append(r)
                    pixels_to_sort_G.append(g)
                    pixels_to_sort_B.append(b)
                if (y > 0):
                    r, g, b = img_temp[x, y-1, 0], img_temp[x, y-1, 1], img_temp[x, y-1, 2]
                    pixels_to_sort_R.append(r)
                    pixels_to_sort_G.append(g)
                    pixels_to_sort_B.append(b)
                if (y < IMAGE_SIZE[0]-1):
                    r, g, b = img_temp[x, y+1, 0], img_temp[x, y+1, 1], img_temp[x, y+1, 2]
                    pixels_to_sort_R.append(r)
                    pixels_to_sort_G.append(g)
                    pixels_to_sort_B.append(b)

                # sort each RGB value
                #   using method 1 from https://www.geeksforgeeks.org/python-returning-index-of-a-sorted-list/#
                # and find where current pixel is sorted to
                #   using example 1 from https://www.geeksforgeeks.org/how-to-find-the-index-of-value-in-numpy-array/
                sort_index = np.argsort(pixels_to_sort_R)
                rank_R = np.where(sort_index == 0)[0][0] + 1
                sort_index = np.argsort(pixels_to_sort_G)
                rank_G = np.where(sort_index == 0)[0][0] + 1
                sort_index = np.argsort(pixels_to_sort_B)
                rank_B = np.where(sort_index == 0)[0][0] + 1

                # store each rank
                img_rank[x, y, 0] = rank_R
                img_rank[x, y, 1] = rank_G
                img_rank[x, y, 2] = rank_B

        # Add an extra dimension to the array
        img_rank = img_rank.flatten()

        # Concatenate the new image array with the existing array of images
        imgs_array.append(img_rank)

    # return completed array
    imgs = np.array(imgs_array)

    return imgs



# Parallel shuffles two arrays of equal lengths
def p_shuffle(x, y, sz):
    shuffle_idx = np.random.permutation(sz)
    shuffled_x = x[shuffle_idx]
    shuffled_y = y[shuffle_idx]


    return shuffled_x, shuffled_y


# Split an array given a number of elements
#   for the first partition
def split_array(x, partition_sz):
    first_partition = x[:partition_sz]
    second_partition = x[partition_sz:]

    return first_partition, second_partition


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


# Define the Logistic Regression class
#   with predict and train methods
class LogisticRegression():
    def __init__(self, input_dim):
        self.w = np.zeros(input_dim + 1)
    
    def predict(self, x):
        s = np.dot(x, self.w[1:]) + self.w[0]
        sig = self.sigmoid(s)
        return np.where(sig >= 0.5, 1, -1)
    
    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def train(self, x, t, test_x, test_t, epochs=NUM_EPOCHS, eta=0.01):
        accuracyListTrain = []
        accuracyListTest = []

        for epoch in range(epochs):
            
            # 1. Calculate the gradient
            m = x.shape[0] # Number of rows
            n = x.shape[1] # Number of dimensions

            gradient = np.zeros(n + 1) # intialize gradient to 0
            for xi, ti in zip(x, t): # sum together
                xi = np.insert(xi, 0, 1)
                numerator = ti * xi
                denominator = 1 + math.exp(ti * np.dot(self.w, xi))
                gradient += numerator / denominator
            gradient = -1/m * gradient # divide

            # 2. Update the weights
            self.w -= eta*gradient / np.linalg.norm(gradient)
            
            
            # Get prediction rates after training
            predIteration = self.predict(x)
            trainAccuracy = np.mean(t == predIteration) * 100
            accuracyListTrain.append(trainAccuracy)
            predIteration = self.predict(test_x)
            trainAccuracy = np.mean(test_t == predIteration) * 100
            accuracyListTest.append(trainAccuracy)
        
        return accuracyListTrain, accuracyListTest


# Start of code
# Load Images
smile_images = process_images(SMILE_PATH)
non_smile_images = process_images(NON_SMILE_PATH)

smile_images_novel = process_images_novel(SMILE_PATH)
non_smile_images_novel = process_images_novel(NON_SMILE_PATH)

print('Shape of image array:', smile_images.shape)
print('Shape of image array:', non_smile_images.shape)



# Naive Bayes section
def NaiveBayes_Experiment():
    # Concatenate image arrays and get labels
    x_nb = np.concatenate((smile_images, non_smile_images), axis=0)
    y_nb = np.concatenate((np.ones(smile_images.shape[0]), np.zeros(non_smile_images.shape[0])))

    # Shuffle and get training/test sets
    shuffled_x_nb, shuffled_y_nb = p_shuffle(x_nb, y_nb, len(x_nb))

    training_x_nb, testing_x_nb = split_array(shuffled_x_nb, int(0.8*len(shuffled_x_nb)))
    training_y_nb, testing_y_nb = split_array(shuffled_y_nb, int(0.8*len(shuffled_y_nb)))

    # Train the naive bayes method
    input_dim = training_x_nb.shape[1]
    nb = NaiveBayes(input_dim, 256)
    result = nb.train(training_x_nb, training_y_nb)
    if (0 != result):
        print(f"error {result}")
        quit()

    # Evaluate the naive bayes method
    train_predictions = nb.predict(training_x_nb)
    test_predictions = nb.predict(testing_x_nb)

    train_accuracy = np.mean(train_predictions == training_y_nb)
    test_accuracy = np.mean(test_predictions == testing_y_nb)

    return train_accuracy, test_accuracy


# Novel Naive Bayes section
def NaiveBayesNovel_Experiment():
    # Concatenate image arrays and get labels
    x_nb = np.concatenate((smile_images_novel, non_smile_images_novel), axis=0)
    y_nb = np.concatenate((np.ones(smile_images_novel.shape[0]), np.zeros(non_smile_images_novel.shape[0])))

    # Shuffle and get training/test sets
    shuffled_x_nb, shuffled_y_nb = p_shuffle(x_nb, y_nb, len(x_nb))

    training_x_nb, testing_x_nb = split_array(shuffled_x_nb, int(0.8*len(shuffled_x_nb)))
    training_y_nb, testing_y_nb = split_array(shuffled_y_nb, int(0.8*len(shuffled_y_nb)))

    # Train the naive bayes method
    input_dim = training_x_nb.shape[1]
    nb = NaiveBayes(input_dim, 9)
    result = nb.train(training_x_nb, training_y_nb)
    if (0 != result):
        print(f"error {result}")
        quit()

    # Evaluate the naive bayes method
    train_predictions = nb.predict(training_x_nb)
    test_predictions = nb.predict(testing_x_nb)

    train_accuracy = np.mean(train_predictions == training_y_nb)
    test_accuracy = np.mean(test_predictions == testing_y_nb)

    return train_accuracy, test_accuracy

# Logistic Regression section
def LogisticRegression_Experiment():
    # Concatenate image arrays and get labels
    x_nb = np.concatenate((smile_images, non_smile_images), axis=0)
    y_nb = np.concatenate((np.ones(smile_images.shape[0]), -1*np.ones(non_smile_images.shape[0])))

    # Shuffle and get training/test sets
    shuffled_x, shuffled_y = p_shuffle(x_nb, y_nb, len(x_nb))

    training_x, testing_x = split_array(shuffled_x, int(0.8*len(shuffled_x)))
    training_y, testing_y = split_array(shuffled_y, int(0.8*len(shuffled_y)))

    # Train the logistic regression method
    input_dim = training_x.shape[1]
    regr = LogisticRegression(input_dim)
    accuracyListTrain, accuracyListTest = regr.train(training_x, training_y, testing_x, testing_y)

    return accuracyListTrain, accuracyListTest

# Logistic Regression (Novel) section
def LogisticRegressionNovel_Experiment():
    # Concatenate image arrays and get labels
    x_nb = np.concatenate((smile_images_novel, non_smile_images_novel), axis=0)
    y_nb = np.concatenate((np.ones(smile_images_novel.shape[0]), -1*np.ones(non_smile_images_novel.shape[0])))

    # Shuffle and get training/test sets
    shuffled_x, shuffled_y = p_shuffle(x_nb, y_nb, len(x_nb))

    training_x, testing_x = split_array(shuffled_x, int(0.8*len(shuffled_x)))
    training_y, testing_y = split_array(shuffled_y, int(0.8*len(shuffled_y)))

    # Train the logistic regression method
    input_dim = training_x.shape[1]
    regr = LogisticRegression(input_dim)
    accuracyListTrain, accuracyListTest = regr.train(training_x, training_y, testing_x, testing_y)

    return accuracyListTrain, accuracyListTest



# Run experiments for NUM_TRIALS trials

# Run Naive Bayes experiments
train_results = []
test_results = []
trainAvgVals =  np.zeros(NUM_EPOCHS)
testAvgVals =  np.zeros(NUM_EPOCHS)

# Run experiments
print("Running Naive Bayes Experiments")
for _ in range(NUM_TRIALS):
    print(f"\tRunning Experiment {_}")
    accuracyTrain, accuracyTest = NaiveBayes_Experiment()
    train_results.append(accuracyTrain)
    test_results.append(accuracyTest)

# Get average for results
trainAvgVals = np.mean(train_results)
testAvgVals = np.mean(test_results)

# Graph the accuracy
print(f"\tAverage Training Accuracy: {trainAvgVals*100}")
print(f"\tAverage Testing Accuracy: {testAvgVals*100}")

plt.bar(["Training Data", "Testing Data"], [trainAvgVals*100, testAvgVals*100])
plt.title('Accuracy of Naive Bayes')
plt.xlabel('Data Set')
plt.ylabel('Prediction Accuracy (%)')
plt.savefig('Figures\\NaiveBayes.png')
plt.clf()



# Run Naive Bayes (Novel) experiments
train_results = []
test_results = []
trainAvgVals =  np.zeros(NUM_EPOCHS)
testAvgVals =  np.zeros(NUM_EPOCHS)

# Run experiments
print("Running Naive Bayes (Novel) Experiments")
for _ in range(NUM_TRIALS):
    print(f"\tRunning Experiment {_}")
    accuracyTrain, accuracyTest = NaiveBayesNovel_Experiment()
    train_results.append(accuracyTrain)
    test_results.append(accuracyTest)

# Get average for results
trainAvgVals = np.mean(train_results)
testAvgVals = np.mean(test_results)

# Plot the accuracy over epochs
print(f"\tAverage Training Accuracy: {trainAvgVals*100}")
print(f"\tAverage Testing Accuracy: {testAvgVals*100}")

plt.bar(["Training Data", "Testing Data"], [trainAvgVals*100, testAvgVals*100])
plt.title('Accuracy of Naive Bayes (Novel)')
plt.xlabel('Data Set')
plt.ylabel('Prediction Accuracy (%)')
plt.savefig('Figures\\NaiveBayesNovel.png')
plt.clf()



# Run Logistic Regression experiments
train_results = []
test_results = []
trainAvgVals =  np.zeros(NUM_EPOCHS)
testAvgVals =  np.zeros(NUM_EPOCHS)

# Run experiments
print("Running Logistic Regression Experiments")
for _ in range(NUM_TRIALS):
    print(f"\tRunning Experiment {_}")
    accuracyListTrain, accuracyListTest = LogisticRegression_Experiment()
    train_results.append(accuracyListTrain)
    test_results.append(accuracyListTest)

# Get average for results
for i in range(NUM_EPOCHS):
    trainVals = [result[i] for result in train_results]
    trainAvgVals[i] = np.mean(trainVals)
    testVals = [result[i] for result in test_results]
    testAvgVals[i] = np.mean(testVals)

# Plot the accuracy over epochs
print(f"\tAverage Training Accuracy (raw): {trainAvgVals[NUM_EPOCHS-1]}")
print(f"\tAverage Testing Accuracy (raw): {testAvgVals[NUM_EPOCHS-1]}")

plt.plot(trainAvgVals, label='Training accuracy')
plt.plot(testAvgVals, label='Testing accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.savefig('Figures\\LRegr.png')
plt.clf()


# Plot every other accuracy over epochs
print(f"\tAverage Training Accuracy (raw): {trainAvgVals[NUM_EPOCHS-2]}")
print(f"\tAverage Testing Accuracy (raw): {testAvgVals[NUM_EPOCHS-2]}")

plt.plot(trainAvgVals[::2], label='Training accuracy')
plt.plot(testAvgVals[::2], label='Testing accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs (x2)')
plt.ylabel('Accuracy (%)')
plt.savefig('Figures\\LRegr_alt.png')
plt.clf()



# Run Logistic Regression (Novel) experiments
train_results = []
test_results = []
trainAvgVals =  np.zeros(NUM_EPOCHS)
testAvgVals =  np.zeros(NUM_EPOCHS)

# Run experiments
print("Running Logistic Regression (Novel) Experiments")
for _ in range(NUM_TRIALS):
    print(f"\tRunning Experiment {_}")
    accuracyListTrain, accuracyListTest = LogisticRegressionNovel_Experiment()
    train_results.append(accuracyListTrain)
    test_results.append(accuracyListTest)

# Get average for results
for i in range(NUM_EPOCHS):
    trainVals = [result[i] for result in train_results]
    trainAvgVals[i] = np.mean(trainVals)
    testVals = [result[i] for result in test_results]
    testAvgVals[i] = np.mean(testVals)

# Plot the accuracy over epochs
print(f"\tAverage Training Accuracy (raw): {trainAvgVals[NUM_EPOCHS-1]}")
print(f"\tAverage Testing Accuracy (raw): {testAvgVals[NUM_EPOCHS-1]}")

plt.plot(trainAvgVals, label='Training accuracy')
plt.plot(testAvgVals, label='Testing accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.savefig('Figures\\LRegrNovel.png')
plt.clf()


# Plot every other accuracy over epochs
print(f"\tAverage Training Accuracy (raw): {trainAvgVals[NUM_EPOCHS-2]}")
print(f"\tAverage Testing Accuracy (raw): {testAvgVals[NUM_EPOCHS-2]}")

plt.plot(trainAvgVals[::2], label='Training accuracy')
plt.plot(testAvgVals[::2], label='Testing accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs (x2)')
plt.ylabel('Accuracy (%)')
plt.savefig('Figures\\LRegrNovel_alt.png')
plt.clf()
