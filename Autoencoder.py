import numpy as np
np.random.seed(123)

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

X = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]])
Y = X

input_size = X.shape[1]
print(input_size)
hidden_size = 3
output_size = Y.shape[1]
eta = 0.05
epochs = 5000

net = neuralNet(input_size, hidden_size, output_size)
net.train(X, Y, epochs, eta)

print("Input:", X)
print("Output:", net.feedForward(X))
net.print_weights()
