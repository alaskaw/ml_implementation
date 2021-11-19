import pandas as pd
import numpy as np


class logistic_regression:
    def __init__(self):
        self.w=None
        self.b=None
        self.loss_list=[]
    def init_param(self,dim):
        self.w=np.zeros((dim,1))
        self.b=0
    def calc_loss(self,x,y,w,b):
        num_train=x.shape[0]
        num_feat=x.shape[1]
        y_pred=1/(1+np.exp(-1* np.dot(x,w)+b))
        loss=-1*np.mean(y*np.log(y_pred)+ (1-y)*np.log(1-y_pred))
        dw= np.dot(x.T,(y_pred-y))/num_train
        db= np.mean(y_pred-y)
        return y_pred,loss,dw,db
    
    def fit(self,x,y,lr,epoches):
        num_train=x.shape[0]
        num_feat=x.shape[1]
        self.init_param(num_feat)
        for i in range(epoches):
            y_pred,loss,dw,db=self.calc_loss( x,y,self.w,self.b)
            self.w-=lr*dw
            self.b-=lr*db 
            self.loss_list.append(loss)
            if i%1000==0:print(f'{i} epoches...')
        
    def predict(self,x):
        return  1/(1+np.exp(-1* np.dot(x,self.w)+self.b))
    
    def predict_val(self,x,threshold=0.5):
        func= np.vectorize(lambda x:1 if x>=threshold else 0)
        pred=1/(1+np.exp(-1* np.dot(x,self.w)+self.b))
        return func(pred)
    
    
    
 






