# Shanker, Avinash
# 1001-668-570
# 2019-09-22
# Assignment-01-01

import numpy as np
import itertools

class Perceptron(object):
    def __init__(self, input_dimensions=2,number_of_classes=4,seed=None):
        """
        Initialize Perceptron model
        :param input_dimensions: The number of features of the input data, for example (height, weight) would be two features.
        :param number_of_classes: The number of classes.
        :param seed: Random number generator seed.
        """
        if seed != None:
            np.random.seed(seed)
        self.input_dimensions = input_dimensions
        self.number_of_classes=number_of_classes
        self._initialize_weights()
    def _initialize_weights(self):
        """
        Initialize the weights, initalize using random numbers.
        Note that number of neurons in the model is equal to the number of classes
        """

        ##Using np.random.randn function to generate random weights
        ##self.input_dimensions+1 added for bias value
        self.weights = np.random.randn(self.number_of_classes,self.input_dimensions+1)
        return self.weights
        raise Warning("You must implement _initialize_weights! This function should initialize (or re-initialize) your model weights. Bias should be included in the weights")

    def initialize_all_weights_to_zeros(self):
        """
        Initialize the weights, initalize using random numbers.
        """

        ##Using np.zeros function to initialize zero weights
        ##self.input_dimensions+1 added for bias value
        self.weights = np.zeros((self.number_of_classes,self.input_dimensions+1))
        return self.weights
        raise Warning("You must implement this function! This function should initialize (or re-initialize) your model weights to zeros. Bias should be included in the weights")

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples]. Note that the input X does not include a row of ones
        as the first row.
        :return: Array of model outputs [number_of_classes ,n_samples]
        """

        ##Add_one is using vstack function to stack 1 in the first row.
        ##Dot product of weights with inputs
        ## n = X.W
        Add_Ones = np.vstack((np.ones(X.shape[1]),X))
        model_product=np.dot(self.weights,Add_Ones)
        ##Model_product does the prediction as a Hadlim function
        model_product[model_product >= 0] = 1
        model_product[model_product < 0 ] = 0
        return model_product
        raise Warning("You must implement predict. This function should make a prediction on a matrix of inputs")


    def print_weights(self):
        """
        This function prints the weight matrix (Bias is included in the weight matrix).
        """
        ##Printing weight matrix with Bias
        Matrix_Bias = self.weights
        print("****** Weight Matrix With Bias ******\n",Matrix_Bias)
        raise Warning("You must implement print_weights")

    def train(self, X, Y, num_epochs=10, alpha=0.001):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the self.weights using Perceptron learning rule.
        Training should be repeted num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_classes ,n_samples]
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :return: None
        """
        ##
        ##Add_one is using vstack function to stack 1 in the first row.
        ##Slicing the row by removing the first line
        X_append = np.vstack((np.ones(X.shape[1]),X))
        for i in range(num_epochs):
            for j in range(X_append.shape[1]):
                slicing_row = X_append[:,j]
                target = Y[:,j]
                #reshaping matrix after perfoming slicing to make it compatible
                slicing_row = slicing_row.reshape(self.input_dimensions+1,1)
                target = target.reshape(self.number_of_classes,1)
                dot_product = np.dot(self.weights,slicing_row)
                dot_product = dot_product.reshape(self.number_of_classes,1)  
                ##dot_product does the prediction as a Hadlim function     
                dot_product[dot_product >= 0] = 1
                dot_product[dot_product < 0 ] = 0
                #caluclating error e = target – a(dot_product)
                #Then transposing and doing dot product with input
                err = target-dot_product
                alpla_prod = alpha*(np.dot(err,slicing_row.T))
                self.weights = self.weights + alpla_prod
        return ('')
        raise Warning("You must implement train")

    def calculate_percent_error(self,X, Y):
        """
        Given a batch of data this function calculates percent error.
        For each input sample, if the output is not hte same as the desired output, Y,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_classes ,n_samples]
        :return percent_error
        """
        ##
        ##Add_one is using vstack function to stack 1 in the first row.
        err = 0
        X_append = np.vstack((np.ones(X.shape[1]),X))
        model_product=np.dot(self.weights,X_append)
        ##Model_product does the prediction as a Hadlim function
        model_product[model_product >= 0] = 1
        model_product[model_product < 0 ] = 0
        for i in range(X_append.shape[1]):
            if np.array_equal(model_product[:,i],Y[:,i])!= True:
                err = 1 + err
        new_shape = X_append.shape[1]        
        percent_error = err/new_shape
        print("****** Output Sample For Error Calculation ******\n",model_product)       
        return percent_error
        raise Warning("You must implement calculate_percent_error")

if __name__ == "__main__":
    """
    This main program is a sample of how to run your program.
    You may modify this main program as you desire.
    """

    input_dimensions = 2
    number_of_classes = 2

    model = Perceptron(input_dimensions=input_dimensions, number_of_classes=number_of_classes, seed=1)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    print("\nFirst Prediction\n",model.predict(X_train))
    Y_train = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    model.initialize_all_weights_to_zeros()
    print("\n****** Model weights ******\n",model.weights,"\n")
    print("****** Input samples ******\n",X_train,"\n")
    print("****** Desired Output ******\n",Y_train,"\n")
    percent_error=[]
    for k in range (20):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.0001)
        percent_error.append(model.calculate_percent_error(X_train,Y_train))
    print("\n******  Percent Error ******\n",percent_error,"\n")
    print("****** Model weights ******\n",model.weights,"\n")

##References
## https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/
## https://medium.com/@thomascountz/19-line-line-by-line-python-perceptron-b6f113b161f3    
    
    