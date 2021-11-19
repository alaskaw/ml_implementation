
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from linear_regression import *

x= load_diabetes().data
y= load_diabetes().target.reshape(-1,1 )
print(x.shape,y.shape)
 
x_train=x[0:300]
y_train=y[0:300]
x_test=x[300:]
y_test=y[300:]

################################################################
#linear regression
################################################################
clf=linear_regresion()
clf.fit(x_train,y_train,1e-3, 100000) 
pred=clf.predict(x_train)
mse_score= mean_squared_error(y_train,pred)
print('train: mse_score=',mse_score)
pred=clf.predict(x_test)
mse_score= mean_squared_error(y_test,pred)
print('test: mse_score=',mse_score)

