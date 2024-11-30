'''naive_bayes_multinomial.py
Naive Bayes classifier with Multinomial likelihood for discrete features
Ghailan Fadah
CS 251 Data Analysis Visualization, Spring 2021
'''
from pickletools import int4
import numpy as np


class NaiveBayes:
    '''Naive Bayes classifier using Multinomial likeilihoods (discrete data belonging to any
     number of classes)'''
    def __init__(self, num_classes):
        '''Naive Bayes constructor

        TODO:
        - Add instance variable for `num_classes`
        '''
        # class_priors: ndarray. shape=(num_classes,).
        #   Probability that a training example belongs to each of the classes
        #   For spam filter: prob training example is spam or ham
        self.class_priors = None
        # class_likelihoods: ndarray. shape=(num_classes, num_features).
        #   Probability that each word appears within class c
        self.class_likelihoods = None
        
        # holds the number of classes in the data
        self.num_classes = num_classes

    def train(self, data, y):
        '''Train the Naive Bayes classifier so that it records the "statistics" of the training set:
        class priors (i.e. how likely an email is in the training set to be spam or ham?) and the
        class likelihoods (the probability of a word appearing in each class — spam or ham)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        TODO:
        - Compute the instance variables self.class_priors and self.class_likelihoods needed for
        Bayes Rule. See equations in notebook.
        '''
        num_samps, num_features = data.shape

        pr_list = []

        likely_list =[]

        li_list = []

        for c in range(self.num_classes):

            # create a subset of data for the specific class 'c'
            X_c = data[y == c]
            pr_list.append(X_c.shape[0] / num_samps)

            denomator = num_features + X_c.sum()
            
            li_list = []
            for f in range(num_features):
                X_f = X_c[:, f]
                numerator = X_f.sum() + 1
                li_list.append(numerator/denomator)
            
            likely_list.append(li_list)



        self.class_priors = np.array(pr_list)

        self.class_likelihoods = np.array(likely_list)



    def predict(self, data):
        '''Combine the class likelihoods and priors to compute the posterior distribution. The
        predicted class for a test sample from `data` is the class that yields the highest posterior
        probability.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each test data sample.

        TODO:
        - For the test samples, we want to compute the log of the posterior by evaluating
        the the log of the right-hand side of Bayes Rule without the denominator (see notebook for
        equation). This can be done without loops using matrix multiplication or with a loop and
        a series of dot products.
        - Predict the class of each test sample according to the class that produces the largest
        log(posterior) probability (hint: can be done without loops).

        NOTE: Remember that you are computing the LOG of the posterior (see notebook for equation).
        NOTE: The argmax function could be useful here.
        '''
        num_test_samps, num_features = data.shape

        p = np.log(self.class_priors.reshape(self.class_priors.shape[0],1))
        l = np.log(self.class_likelihoods)
      
        
        sub_matrix = p + l @ data.transpose()


        return np.argmax(sub_matrix, axis = 0)
        
        




    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''
        
        incorrect = np.count_nonzero((y - y_pred))
        correct = y.shape[0] - incorrect

                
        return correct / float(y.shape[0])
        
       

    def confusion_matrix(self, y, y_pred):
        '''Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
        by the classifier (`y_pred`).

        Recall: the rows represent the "actual" ground truth labels, the columns represent the
        predicted labels.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_samps,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_samps,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        ndarray. shape=(num_classes, num_classes).
            Confusion matrix
        '''
        # To get the number of classes, you can use the np.unique
        # function to identify the number of unique categories in the
        # y matrix.
        un_arr = np.unique(y)
        r = un_arr.shape[0]

        
        y_pred = np.array(y_pred, dtype=int)
        y = np.array(y, dtype=int)
        m = [[0] * r for i in range(r)]
        for pred, exp in zip(y_pred, y):
            m[pred][exp] += 1
        return m
        
  
        


