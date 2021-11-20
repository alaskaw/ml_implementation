import pandas as pd
import numpy as np
from collections import Counter 


class knn_classifier:
    def __init__(self):
        pass
    def train(self,x,y):
        self.x_train=x
        self.y_train=y
        
    def calc_dist(self,x_test):
        sim=np.dot(x_test,self.x_train.T)
        tr=np.square(self.x_train).sum(axis=1)
        te=np.square(x_test).sum(axis=1)
        sim=np.sqrt((-2*sim+ tr.reshape(1,-1) + te.reshape(-1,1)))
        return sim
    
    def predict_label(self,x_test,sim,k=1):
        num_test=x_test.shape[0 ] 
        y_pred=np.zeros(num_test)
        for i in range(num_test):
            y_pred[i]=Counter(self.y_train[np.argsort(sim[i,:])].flatten()[0:k]).most_common(1)[0][0]
        return y_pred 
    
    def cross_validation(self,x,y,k_list=[1,3,5,7,10,20,30,50,80,100]):
        n_fold=5
        x=np.array_split(x,n_fold)
        y=np.array_split(y,n_fold) 
        
        for k in k_list:
            for ifold in range(n_fold):
                x_tr=np.concatenate(x[:ifold]+x[ifold+1:])
                y_tr=np.concatenate(y[:ifold]+y[ifold+1:])
                x_te=x[ifold]
                y_te=y[ifold]
                self.train(x_tr,y_tr)
                sim=self.calc_dist(x_te)
                pred=self.predict_label(x_te,sim,k=k)
                acc= accuracy_score(y_te,pred)
                print(f'k={k},acc={acc}') 

                

                
                
                
                
                
