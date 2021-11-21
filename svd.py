class svd:
    def __init__(self,n_component):
        self.n_component=n_component
    
    def svd(self,x):
        m1=np.dot(x,x.T)
        eigval,eigvec=np.linalg.eig(m1)
        index=np.argsort(-eigval)[:self.n_component] 
        U=eigvec[:,index]
        sigma_sq=eigval[index]
        m1=np.dot(x.T,x)
        eigval,eigvec=np.linalg.eig(m1)
        index=np.argsort(-eigval)[:self.n_component]
        V=eigvec[:,index]
        sigma= np.diag(np.sqrt(sigma_sq))
        res=np.dot(np.dot(U,sigma),V.T)
        print(sigma.shape)
        return np.dot(U,sigma)
      
 




#https://mp.weixin.qq.com/s?subscene=23&__biz=MzI3MzkyMzE5Mw==&mid=2247484584&idx=1&sn=56cb909fa8542bf34502349acef04475&chksm=eb1aa9bedc6d20a893ae12d57e932407c652dcb4344c803cd1929c195a3b770bd92703e1cccc&scene=7&key=4e925b84958ecb0dab9631e6038b349866f34aa6d75e47faed32274bf2473afdfc550304886d13a2fcf1ae55beb331f944b2fdf8c6da8f742ef5422bd8026d9198c34043ec1267a3ea5857b292c9ad99&ascene=0&uin=MjY3Njc1MDU2Mg%3D%3D&devicetype=Windows+10&version=62080085&lang=zh_CN&exportkey=AT48EC6lhcbufVvMuQUCAL0%3D&pass_ticket=l%2FBjrFwUv7TSWEJX%2BkzwyrF%2F0r%2FlQb3j91Qm6%2BIp4KKKMeflHRAgbhobUHi6Ce6y






