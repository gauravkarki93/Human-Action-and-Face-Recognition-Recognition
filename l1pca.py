import numpy as np

def l1_pca(X,L):
    N = np.size(X,1)
    max_iter = 1000
    
    delta = np.zeros(N)
    obj_val = 0
    
    for l in range(0,L):
        b =  np.random.randint(2,size = N)
        b[b==0] = -1
        
        for iter in range(0,max_iter):
            for i in range(0,N):                
                bi = np.delete(b,i)                
                xi = np.delete(X, i, axis=1)
                firstHalf = -4 * b[i] * np.transpose(X[:, i])
                #print(firstHalf)
                #print(np.matmul(xi, bi))
                delta[i] = np.matmul(firstHalf, np.matmul(xi, bi))
                #print(delta[i])
                    
        maxValIndex = np.argmax(delta)
        if(delta[maxValIndex] > 0):
                b[maxValIndex] = -1 * b[maxValIndex]
        else:
            break

        temp = np.linalg.norm(np.matmul(X,b), 2)

        if(temp > obj_val):
            obj_val = temp
            bopt = b

        return np.matmul(X,bopt) / np.linalg.norm(np.matmul(X,bopt), 2)