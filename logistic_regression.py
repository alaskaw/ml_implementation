import pandas as pd
import numpy as np


class logistic_regression:
    def __init__(self):
        self.w=None
        self.b=None
        self.loss_list=[]
        self.mean=None 
        self.std=None
        self.solver='gradient'
    def init_param(self,dim):
        self.w=np.zeros((dim,1))
        self.b=0
    
    def std_fit(self,x):
        self.std=x.std(axis=0)
        self.mean=x.mean(axis=0) 
    def std_transform(self,x):
        return (x-self.mean)/self.std
    def std_fit_transform(self,x):
        self.std=x.std(axis=0)
        self.mean=x.mean(axis=0)
        return (x-self.mean)/self.std
        
    def calc_loss(self,x,y,w,b):
        dw=0
        db=0
        num_train=x.shape[0]
        num_feat=x.shape[1]
        y_pred=1/(1+np.exp(-1* np.dot(x,w)+b))
        loss=-1*np.mean(y*np.log(y_pred)+ (1-y)*np.log(1-y_pred))
        grad_w= np.dot(x.T,(y_pred-y))/num_train
        grad_b= np.mean(y_pred-y)
        hess_w= np.dot((x*x).T, (y_pred* (1-y_pred)))/num_train
        hess_b= np.mean(y_pred*(1-y_pred))
        if self.solver=='gradient': 
            dw=grad_w
            db=grad_b
        elif self.solver=='newton':
            dw=grad_w/hess_w 
            db=grad_b/hess_b
        return y_pred,loss,dw,db

    
    def fit(self,x,y,lr,epoches,solver='gradient'):
        num_train=x.shape[0]
        num_feat=x.shape[1]
        self.solver=solver
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
