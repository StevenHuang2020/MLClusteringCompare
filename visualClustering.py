import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from getDataSet import getFacebookLiveDataset,getSalesTransactionsDataset,getIrisDataset
from modelCreate import createKMeans,createAgglomerate,createDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import silhouette_score
from tsne import tsne

def visualClusterResult(data, labels, bestK, title):
    print('data.shape=',data.shape,'labels.shape=',labels.shape,'bestK=',bestK)
    #Y = tsne(data, 2, 10, 50,max_iter=500)
    Y = TSNE(n_components=2, verbose=0, perplexity=50, n_iter=1000).fit_transform(data)#scikit-learn
    
    title = title+'_k_'+str(bestK)
    plt.title(title)
    plt.scatter(Y[:, 0], Y[:, 1], 20, labels) #marker='s',edgecolor='black'
    plt.savefig('./images/'+title+'.png')
    plt.show()

def main():
    pass

if __name__=='__main__':
    main()