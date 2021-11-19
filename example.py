
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from linear_regression import *
from logistic_regression import *



################################################################
#linear regression
################################################################
x= load_diabetes().data
y= load_diabetes().target.reshape(-1,1 )
print(x.shape,y.shape) 
x_train=x[0:300]
y_train=y[0:300]
x_test=x[300:]
y_test=y[300:]

clf=linear_regresion()
clf.fit(x_train,y_train,1e-3, 100000) 
pred=clf.predict(x_train)
mse_score= mean_squared_error(y_train,pred)
print('train: mse_score=',mse_score)

pred=clf.predict(x_test)
mse_score= mean_squared_error(y_test,pred)
print('test: mse_score=',mse_score)


################################################################
#logistic regression
################################################################
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score,accuracy_score
x,y=make_classification(n_samples=10000,n_features=100,n_redundant=0, n_informative=5, n_classes=2,n_clusters_per_class=10,random_state=42)
y=y.reshape(-1,1)
print(x.shape,y.shape)
x_train=x[:7000]
y_train=y[:7000]
x_test=x[7000:]
y_test=y[7000:]

clf=logistic_regression() 
clf.fit(x_train,y_train,lr=1e-3,epoches=1000)
pred=clf.predict(x_train)
auc_score=roc_auc_score(y_train,pred)
pred=clf.predict_val(x_train,threshold=0.45)
acc_score= accuracy_score(y_train,pred)
print('train: auc_score=',auc_score, 'acc_score=' ,acc_score)

pred=clf.predict(x_test)
auc_score=roc_auc_score(y_test,pred)
pred=clf.predict_val(x_test,threshold=0.45)
acc_score= accuracy_score(y_test,pred)
print('train: auc_score=',auc_score, 'acc_score=' ,acc_score)








