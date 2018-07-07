import numpy as np 

class NaiveBayesClassifier():
    
    def __init__(self,X,y):
    # shape of X is like [ [x11,x12,...],[x21,x22,...],[],[xn1,xn2,...]]
    # shape of y is like [ [y1],[y2],[],...]]

    # Check if X,Y can correspond to each other correctly in dimensionality
    num_input=X.shape[0]
    num_features=X.shape[1]
    if y.shape[0]!=num_input:
        print("The dimension of X doesn't correspond with that of X")
        return 0
    # convert y to pandas data structure Series
    y=pd.Series(y.ravel())
    
    # return the dict that stores the prior probabilities
    # of each label
    def get_prior_probabilities():
        # get types of label
        y_set=y.unique()
        # store the prior_probabilities in dict
        probabilities=dict()
        # loop through each label
        # 'y_series==y_' get boolean array and the sum of it
        # get the total number of 'True' 
        for y_ in y_set:
            probabilities[y_]=(y_series==y_).sum()/num_input
        return probabilities
    
    # get conditional probabilities p(x_i|y_i)
    def get_conditional_probabilities(x_i,y_i,offset):
        # x_pop consists the X that its label is y_i
        x_pop=X[y==y_i]
        # x_matched consists the X that its label is y_i and x_offset is x_i
        x_matched=x_pop[x_pop[:,offset]==x_i]
        return len(x_matched)/x_pop

    def predict(X):
        # shape of X is like [ [x11,x12,...],[x21,x22,...],[],[xn1,xn2,...]]
        # shape of y is like [ [y1],[y2],[],...]]
        
        # first get prior probabilities
        probabilities=get_prior_probabilities()
        # the prediction list
        y_output=[]
        # for each data input
        for x in X:
            y_posterior_probs=list()
            # loop though each possible label
            for y_ in y.unique():
                # the list consists of all the conditional probs for
                # each feature
                probs=list()
                
                for offset,x_ in enumerate(x):
                    cond_prob=get_conditional_probabilities(x_,y_,offset)
                    probs.append(cond_prob)
                # we get the posterior prob of each label
                p=np.prod(probs)*probabilities[y_]
                y_posterior_probs.append(p)
            y_posterior_probs=np.asarray(y_posterior_probs)
            output=y.unique[np.argmax(y_posterior_probs)]
            y_output.append(output)
        
        return np.asarray(y_output).reshape(-1,1)






