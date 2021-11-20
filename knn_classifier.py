import pandas as pd
import numpy as np
from collections import Counter 


class knn_classifier:
    def __init__(self,x,y):
        self.x_train=x 
        self.y_train=y
        
    def calc_dist(self,x_test):
        sim=np.dot(x_test,self.x_train.T)
        tr=np.square(self.x_train).sum(axis=1)
        te=np.square(x_test).sum(axis=1)
        sim=np.sqrt((-2*sim+ tr.reshape(1,-1) + te.reshape(-1,1)))
        return sim
    
    def predict_label(self,sim,k=1):
        num_test=x_test.shape[0 ] 
        y_pred=np.zeros(num_test)
        for i in range(num_test):
            y_pred[i]=Counter(self.y_train[np.argsort(sim[i,:])].flatten()[0:k]).most_common(1)[0][0]
        return y_pred 

      
      

      
 
