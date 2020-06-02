import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualClusterResult(data, labels, bestK, title):
    print('data.shape=',data.shape,'labels.shape=',labels.shape,'bestK=',bestK)
    #Y = tsne(data.values, 2, 10, 50,max_iter=1000)
    Y = TSNE(n_components=2, verbose=0, perplexity=50, n_iter=1000).fit_transform(data)#data.values
    plt.title(title)
    plt.scatter(Y[:, 0], Y[:, 1], 20, labels) #marker='s',edgecolor='black'
    plt.savefig('./images/'+title+'.png')
    #plt.show()