import numpy as np
import scipy

class Perceptron():
    def __init__(self):
        """
        Initialises Perceptron classifier with initializing 
        weights, alpha(learning rate) and number of epochs.
        """
        self.w = np.random.rand(10,3072)
        self.alpha = 0.3
        self.epochs = 100
        
    def train(self, X_train, y_train):
        """
        Train the Perceptron classifier. Use the perceptron update rule
        as introduced in Lecture 3.

        Inputs:
        - X_train: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y_train: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        """
        self.X_train = X_train
        self.y_train = y_train
        for n in range(self.epochs):
            for i in range(X_train.shape[0]):
                    score = np.dot(self.w, X_train[i])
                    v = np.argmax(score)

                    if(v==y_train[i]):
                        self.w[v] = self.w[y_train[i]]
                        
                    else:
                        self.w[y_train[i]] = self.w[y_train[i]] + self.alpha*X_train[i] 
                        self.w[v] = self.w[v]-self.alpha*X_train[i]
      """
        self.X_train = X_train
        self.y_train = y_train
        for n in range(self.epochs):
            for i in range(X_train.shape[0]):
                    score = np.dot(self.w, X_train[i])
                    v = np.argmax(score)

                    if(v==y_train[i]):
                        self.w[v] = self.w[y_train[i]]
                        
                    else:
                        for j in range(self.w.shape[0]):
                            if(j == y_train[i] ):
                                self.w[j] = self.w[j] + self.alpha*X_train[i] 
                            else:
                                self.w[j] = self.w[j]-self.alpha*X_train[i]
            self.alpha = 0.9*self.alpha
                        
                        
         
                        
       
    
    
        
                
                      
            
        
            
        

    def predict(self, X_test):
        """
        Predict labels for test data using the trained weights.

        Inputs:
        - X_test: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.

        Returns:
        - pred: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        
        self.X_test = X_test
        pred = []
        for i in range(X_test.shape[0]):
            val1 = np.dot(self.w, X_test[i])
            pred.append(np.argmax(val1))

        
        return pred 
       