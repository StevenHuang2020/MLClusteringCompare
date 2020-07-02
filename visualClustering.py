import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from getDataSet import getFacebookLiveDataset,getSalesTransactionsDataset,getIrisDataset
from modelCreate import createKMeans,createAgglomerate,createDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import silhouette_score
from tsne import tsne
from manifoldLearning import manifoldLearningTransform,plotManifold

def visualClusterResult(data, labels, bestK, title):
    #return plotManifold(data,labels)
    print('data.shape=',data.shape,'labels.shape=',labels.shape,'bestK=',bestK)
    if 0:
        method='PCA'
        Y = PCA(n_components=2).fit_transform(data)
    elif 0:
        #Y = tsne(data, 2, 10, 50,max_iter=500)
        Y = TSNE(n_components=2, verbose=0, perplexity=50, n_iter=1000).fit_transform(data)#scikit-learn
        method='TSNE'
    else:
        name = 'LLE'
        name = 'LTSA'
        name = 'Hessian LLE'
        name = 'Modified LLE'
        name = 'Isomap'
        name = 'MDS'
        name = 'SE'
        name = 't-SNE'
        
        if name == 'Hessian LLE' or name == 'Modified LLE' or name == 'LTSA':
            Y = manifoldLearningTransform(data,name=name,eigen_solver='dense')
        else:
            Y = manifoldLearningTransform(data,name=name)
        
        method='manifold' + '_' + name
        
    title = title + '_' + method + '_k_'+str(bestK)
    plt.title(title)
    plt.scatter(Y[:, 0], Y[:, 1], 20, labels) #marker='s',edgecolor='black'
    plt.savefig('./images/'+title+'.png')
    plt.show()

def main():
    pass

if __name__=='__main__':
    main()