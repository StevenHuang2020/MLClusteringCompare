#Steven python3 tensorflow2.2
#from tensorflow.keras.datasets import mnist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.linalg import eigh
import seaborn as sns
from sklearn.manifold import TSNE

from getDataSet import getCsv
#reference: https://medium.com/analytics-vidhya/pca-vs-t-sne-17bcd882bf3d

gSaveBase=r'./output/'
gIndex = 0

def getData():
    return getCsv(r'./db/mnist_train.csv',verbose=True)
    
def visaulizeData(data,idx=3):
    global gIndex
    plt.figure(figsize=(7,7))
    grid_data = data.iloc[idx].values.reshape(28,28)
    plt.imshow(grid_data,interpolation=None,cmap='viridis') # Accent
    plt.savefig(gSaveBase+'visaulizeData_'+str(gIndex)+'.png')
    gIndex+=1
    plt.show()
    
def plottingAllData(df,title):
    global gIndex
    sns.FacetGrid(df,hue='label',height=5).map(plt.scatter, '1st_principal','2nd_principal').add_legend()
    plt.title(title)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=0.94, wspace=None, hspace=None)
    plt.savefig(gSaveBase+'plottingAllData_'+str(gIndex)+'.png')
    gIndex+=1
    plt.show()
   
def PCACumulative(data):
    global gIndex
    print('data.shape=',data.shape)
    pca = PCA(n_components=784)
    pca_data = pca.fit_transform(data)
    
    percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)
    cum_var_explained = np.cumsum(percentage_var_explained)
    
    #plot the PCA spectrum
    plt.figure(1,figsize=(6,4))
    plt.clf()
    plt.plot(cum_var_explained,linewidth=2)
    plt.axis('tight')
    plt.grid()
    plt.xlabel('n_components')
    plt.ylabel('cumulative explained variance')
    plt.savefig(gSaveBase+'PCACumulative'+str(gIndex)+'.png')
    gIndex+=1
    plt.show()
     
def preprocessing(df):
    labels = df['label']
    data = df.drop('label', axis=1)
    print('labels.shape=',labels.shape)
    print('data.shape=',data.shape)
    
    #visaulizeData(data)
    data = StandardScaler().fit_transform(data)
    
    #PcaData(data,labels)
    #PCACumulative(data)
    scipyPCA(data,labels)
    #TsneData(data,labels)
    
def calculatePCA(data,N=784):
    print('scipyPCA data.shape=',data.shape)
    #co-variance of matrix
    cov_matrix = np.matmul(data.T,data)
    print('cov_matrix.shape=',cov_matrix.shape)
    
    #finding eigenvalue and corresponding vectors
    values,vectors = eigh(cov_matrix,eigvals=(N-2,N-1)) #eigvals=(N-2,N-1)
    print('vectors.shape=',vectors.shape)
    vectors = vectors.T
    print('vectors.shape=',vectors.shape)
    print('vectors=',vectors[:,:])
    #print('vectors=',vectors[:,-10:])
    
    print('values.shape=',values.shape)
    print('values=',values)
    
    #projecting the original data sample on the plane formed by two principal eigen vectors by vector-vector multiplication.
    new_cord = np.matmul(vectors,data.T)
    print('new_cord.shape=',new_cord.shape)
    return new_cord

def scipyPCA(data,labels):
    '''
    print('scipyPCA data.shape=',data.shape)
    #co-variance of matrix
    cov_matrix = np.matmul(data.T,data)
    print('cov_matrix.shape=',cov_matrix.shape)
    
    #finding eigenvalue and corresponding vectors
    values,vectors = eigh(cov_matrix,eigvals=(782,783))
    print('vectors.shape=',vectors.shape)
    vectors = vectors.T
    print('vectors.shape=',vectors.shape)
    print('vectors=',vectors[:,10:])
    print('vectors=',vectors[:,-10:])
    
    print('values.shape=',values.shape)
    print('values=',values)
    
    #projecting the original data sample on the plane formed by two principal eigen vectors by vector-vector multiplication.
    new_cord = np.matmul(vectors,data.T)
    print('new_cord.shape=',new_cord.shape)
    '''
    new_cord = calculatePCA(data)
    
    #appending label to the 2d projected data
    #creating a new data frame for plotting the labeled points
    new_cord = np.vstack((new_cord,labels)).T
    
    df = pd.DataFrame(data = new_cord,columns=('1st_principal','2nd_principal','label'))
    print(df.head())
    plottingAllData(df,'scipyPca')
        
def PcaData(data,labels):
    print('PCA data.shape=',data.shape)
    pca_data = PCA(n_components=2).fit_transform(data)
    print('pca_data.shape=',pca_data.shape)
    pca_data = np.vstack((pca_data.T,labels)).T
    
    df = pd.DataFrame(data = pca_data,columns=('1st_principal','2nd_principal','label'))
    print(df.head())
    plottingAllData(df,'PCA')
    
def TsneData(data,labels):
    print('TSNE data.shape=',data.shape)
    tsne_data = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000,random_state=0).fit_transform(data)
    tsne_data = np.vstack((tsne_data.T,labels)).T
    
    df = pd.DataFrame(data = tsne_data,columns=('1st_principal','2nd_principal','label'))
    print(df.head())
    plottingAllData(df,'Tsne')
    
def testPca():
    a = np.array([[1,2,3,4],
                  [5,6,7,12],
                  [6,7,8,14]])
    a = np.array([[1,2,3,4],
                  [1,2,3,4],
                  [1,2,3,4]])
    new_cord = calculatePCA(a,N=4)
    print('new_cord=',new_cord)
    
def main():
    #df = getData()
    #preprocessing(df)
    testPca()
    
if __name__ == "__main__":
    main()
    