import numpy as np 
import pandas as pd

class  pca:
    def __init__(self):
        pass
    def calc_covariance_matrix(self,x):
        n_sample=x.shape[0]
        x=x-np.mean(x,axis=0)
        return 1/n_sample * np.matmul(x.T,x)
    
    def pca(self,x,n_components):
        covar=self.calc_covariance_matrix(x)
        eigval,eigvec=np.linalg.eig(covar)
        idx=eigval.argsort()[::-1]
        eigvec=eigvec[:,idx]
        eigvec=eigvec[:,:n_components]
        return np.matmul(x,eigvec)
    
    def test_digit(self):
        from sklearn.datasets import load_digits
        import matplotlib.pyplot as plt
        import matplotlib.cm as cmx
        import matplotlib.colors as colors
        
        digit=load_digits()
        x=digit.data
        y=digit.target
        x_trans=pca().pca(x,2)
        print(x.shape,y.shape,x_trans.shape)

        x1=x_trans[:,0]
        x2=x_trans[:,1]
        cmap=plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]
        class_distr = []
        for i, l in enumerate(np.unique(y)):
            _x1 = x1[y == l]
            _x2 = x2[y == l]
            _y = y[y == l]
            class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))
        plt.legend(class_distr, y, loc=1)
        plt.suptitle("PCA Dimensionality Reduction")
        plt.title("Digit Dataset")
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()



'''
1.x:(m*n)标准化,均值0方差1
2.计算x的协方差矩阵covar=1/n_sample* x.T*x
3.计算covar的特征值和特征向量
4.按照特征值排序取top k个特征向量 P:(k*n)
5.y=p*x : m*k得到降维后的数据
'''



