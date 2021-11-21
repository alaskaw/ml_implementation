import pandas as pd 
import numpy as np

class kmeans:
    def __init__(self):
        pass
    def calc_distance(self,x1,x2):
        length=len(x1)
        dist=0
        for i in range(length):
            dist+= pow((x1[i]-x2[i]),2)
        return np.sqrt(dist)
    def init_center(self,k,x):
        n_sample=x.shape[0]
        n_feat=x.shape[1]
        centers=np.zeros((k,n_feat))
        selected=[] 
        select_idx=0
        for i in range(k):
            select_idx=np.random.choice(range(n_sample))
            while select_idx in selected:
                select_idx=np.random.choice(range(n_sample))
            selected.append(select_idx)
            centers[i]=x[select_idx]
        print('selected:',selected)
        return centers
    def nearest_center(self,sample,centers):
        min_dist=float('inf')
        min_ctr=0
        for i,ctr in enumerate(centers):
            dist=self.calc_distance(sample,ctr)
            if dist<min_dist:
                min_dist=dist
                min_ctr=i
        return min_ctr

    def create_clusters(self,x,centers,k):
        clusters=[[] for i in range(k)]
        for i,col in enumerate(x):
            ctr=self.nearest_center(col,centers)
            clusters[ctr].append(i)
        return clusters
    
    def get_cluster_label(self,clusters,x):
        n_sample=x.shape[0]
        y_pred=np.zeros(n_sample)
        for i,cluster in enumerate(clusters):
            for j in cluster:
                y_pred[j]=i
        return y_pred
    
    def calc_centers(self,clusters,x,k):
        n_feat=x.shape[1]
        centers=np.zeros((k,n_feat))
        for i,cluster in enumerate(clusters ):
            ctr=np.mean(x[cluster],axis=0)
            centers[i]=ctr
        return centers
    
    def kmeans(self,x,k,max_iter):
        centers=self.init_center(k,x)
        for _ in range(max_iter):
            clusters=self.create_clusters(x,centers,k)
            prev_centers=centers
            centers=self.calc_centers(clusters,x,k)
            diff=centers-prev_centers
            if not diff.any():break
        print(clusters)
        return clf.get_cluster_label(clusters,x)



            
