import pandas as pd
import numpy as np 

class linear_regresion:
    def __init__(self):
        self.w=None
        self.b=None
    def init_param(self,dim):
        self.w=np.zeros((dim,1 ))
        self.b=0 
    def calc_loss(self,x,y,w,b):
        num_train=x.shape[0]
        num_feat=x.shape[1]
        y_pred=np.dot(x,w)+b
        loss= np.mean((y-y_pred)**2)
        dw= 2*np.dot(x.T,(y_pred-y))/num_train
        db= 2*np.sum(y_pred-y)/num_train
        return  y_pred,loss,dw,db
    
    def fit(self,x,y,lr,epochs): 
        num_train= x.shape[0] 
        num_feat= x.shape[1]
        self.init_param(num_feat) 
        for i in range(epochs):
            y_pred,loss,dw,db=self.calc_loss(x,y,self.w,self.b)
            self.w-=lr*dw
            self.b-=lr*db
            if i%1000==0:print(f'{i} epochs...')
        return loss
    
    def predict(self, x):
        y_pred=np.dot(x,self.w)+self.b
        return y_pred
    
    def print_param(self):
        print(self.w) 
        print(self.b) 
        print(np.mean(self.w),np.std(self.w))
        
       
      
        
