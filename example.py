
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
x,y=make_classification(n_samples=10000,n_features=400,n_redundant=0, n_informative=10, n_classes=2,n_clusters_per_class=5,random_state=42)
y=y.reshape(-1,1)
print(x.shape,y.shape)
x_train=x[:7000]
y_train=y[:7000]
x_test=x[7000:]
y_test=y[7000:]

clf=logistic_regression() 
clf.std_fit(x_train)
x_train=clf.std_transform(x_train)
clf.fit(x_train,y_train,lr=1e-3,epoches=1000, solver='newton')
pred=clf.predict(x_train)
auc_score=roc_auc_score(y_train,pred)
pred=clf.predict_val(x_train,threshold=0.5)
acc_score= accuracy_score(y_train,pred)
print('train: auc_score=',auc_score, 'acc_score=' ,acc_score)

x_test=clf.std_transform(x_test)
pred=clf.predict(x_test)
auc_score=roc_auc_score(y_test,pred)
pred=clf.predict_val(x_test,threshold=0.5)
acc_score= accuracy_score(y_test,pred)
print('train: auc_score=',auc_score, 'acc_score=' ,acc_score)


################################################################
#knn classifier
################################################################
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

iris = load_iris()
x, y = shuffle(iris.data, iris.target, random_state=13)
y=y.reshape(-1,1)
print(x.shape ,y.shape)
offset=int(x.shape[0]*0.7)
x_train=x[:offset] 
y_train=y[:offset]
x_test=x[offset:]
y_test=y[offset:]

clf=knn_classifier()
clf.train(x_train,y_train)
sim=clf.calc_dist(x_test)
pred=clf.predict_label(x_test,sim,k=5)
acc=accuracy_score(pred,y_test)
print(acc)












