import numpy as np
import scipy

class KNN():
    def __init__(self, k):
        """
        Initializes the KNN classifier with the k.
        """
        self.k = k
    
    def train(self, X_train, y):
        """
        Train the classifier. For k-nearest neighbors this is just 
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X_train
        self.y = y
        
    
    def find_dist(self, X_test):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train.

        Hint : Use scipy.spatial.distance.cdist

        Returns :
        - dist_ : Distances between each test point and training point
        """
        self.X_test = X_test
        dist = scipy.spatial.distance.cdist(self.X_train,self.X_test)
        
        return dist

    
    def predict(self, X_test):
        """
        Predict labels for test data using the computed distances.

        Inputs:
        - X_test: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.

        Returns:
        - pred: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        self.X_test = X_test
        d_sorted = np.argsort(self.find_dist(X_test),axis=0)
        a = d_sorted[:self.k]
        
        labels = self.y[a]
        
        work = np.transpose(labels)
        pred = []
        for i in range (work.shape[0]):
            votes = np.bincount(work[i])
            pred.append(np.argmax(votes))

           
            
        return pred #change it back to pred
                
                
            
            
            
            
            
       
        
      