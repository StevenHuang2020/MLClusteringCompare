#Steven 05/17/2020
#clustering model design
from time import time
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn.metrics import make_scorer
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.neighbors.nearest_centroid import NearestCentroid

def createDBSCAN(eps=0.5,samples=5):
    model = DBSCAN(eps=eps, min_samples=samples)
    #print(model)
    return model

def createAgglomerate(k=2):
    model = AgglomerativeClustering(n_clusters=k)
    #print(model)
    return model

def createKMeans(k=2):
    model = KMeans(n_clusters=k, random_state=0)
    #print(model,k)
    #print('k=',k)
    return model

def pipeLineModelDesignClassifier(X_train, y_train,featureAlgorithm, model, param_grid):
    pipe_dt = make_pipeline(StandardScaler(), featureAlgorithm,model)
    #param_grid=[{'decisiontreeclassifier__max_depth': [1, 2, 3, 4, 5, 6, 7, None]}]
    gs = GridSearchCV(estimator=pipe_dt,
                    param_grid=param_grid,
                    scoring='accuracy',
                    cv=2)
    kfold = StratifiedKFold(n_splits=10, random_state=1).split(X_train, y_train)
    scores = []
    for k, (train, test) in enumerate(kfold):
        gs.fit(X_train[train], y_train[train])
        score = gs.score(X_train[test], y_train[test])
        scores.append(score)
        print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1, np.bincount(y_train[train]), score))
    scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    return pipe_dt

def s_score_silhouette(estimator, X):
    labels_ = estimator.fit_predict(X)
    score = 0
    #print(X.shape)
    #print(X)
    actualK = len(list(set(labels_)))
    if actualK>1:    
        #print(labels_)
        score = silhouette_score(X, labels_, metric='sqeuclidean') #'euclidean'
        #score = calinski_harabasz_score(X, labels_)
        #score = davies_bouldin_score(X, labels_)
    #print(score)
    return score

def squaredDistances(a,b):
    return np.sum((a-b)**2)

def calculateSSE2(data,labels,centroids):
    print(data.shape,type(data),centroids.shape)
    sse = 0
    for i,ct in enumerate(centroids):
        #print('i,ct=',i,ct)
        samples = []
        
        for k in range(data.shape[0]):
            label = labels[k]
            sample = data.iloc[k,:].values
            #print('sample,label=',i,sample,label)
            if label == i:
                #sse += squaredDistances(sample,ct)
                samples.append(sample)

        sse += squaredDistances(samples,ct)
    return sse 

def calculateSSE(data,labels,centroids):
    #print(data.shape,type(data),centroids.shape)
    sse = 0
    for i,ct in enumerate(centroids):
        #print('i,ct=',i,ct)
        #samples = data.iloc[np.where(labels == i)[0], :].values
        samples = data[np.where(labels == i)[0], :]
        sse += squaredDistances(samples,ct)
    return sse 

def calculateDaviesBouldin(data,labels):
    return davies_bouldin_score(data, labels)

def pipeLineModelDesignClustering(X_train,featureDecomposition, model, param_grid):
    #pipe_dt = make_pipeline(StandardScaler(), model) #featureDecomposition
    #pipe_dt = make_pipeline(featureDecomposition, StandardScaler(), model)
    #pipe_dt = make_pipeline(featureDecomposition, StandardScaler(), model)
    #pipe_dt = make_pipeline(featureDecomposition, MinMaxScaler(), model)
    pipe_dt = make_pipeline(model)
    
    gs = GridSearchCV(estimator=pipe_dt,
                    param_grid=param_grid,
                    scoring=s_score_silhouette,
                    return_train_score=False,
                    cv=None)
    #print(gs)
    
    t = time()
    gs.fit(X_train)
    tt = time()-t
    print("\niter run in %.2fs" % (tt))
    
    labels = gs.best_estimator_.fit_predict(X_train)
    k = len(list(set(labels))) #actual K
    sse,dbValue,mse = 0,0,0 
    if k > 1:
        clf = NearestCentroid()
        clf.fit(X_train, labels)
        sse = calculateSSE(X_train,labels,clf.centroids_)
        mse = sse/len(labels)
        dbValue=calculateDaviesBouldin(X_train,labels)
        
    print('sse =',sse,'dbValue=',dbValue,'mse =', mse, 'k=',k)    

    #print(gs.cv_results_)
    df = pd.DataFrame(gs.cv_results_)
    print(df)
    df.to_csv('result.csv',index=True)
        
    df = df.sort_values(by=['rank_test_score'],ascending=True)
    print('Train result order by rank_test_score:\n',df)
    csm = df.iloc[0]['mean_test_score']
    print('selected CSM =', csm)
    
    columns=['Best-K','tt(s)','SSE','DB','CSM']
    return pd.DataFrame([[k, tt,sse,dbValue,csm]], columns=columns),labels,k
  
def pipeLineModel_KMeans_Design(X_train):
    #pca=7, n_clusters=bestk=2  #tts sse db csm  4.435309  68.396242  1.737907  0.467619
    #pca=5, n_clusters=bestk=3  #tts sse db csm  3.559167  35.432664  1.027498  0.581384
    #pca=12, n_clusters=bestk=2  #tts sse db csm  4.363931  96.43872  2.131834  0.408379
    
    model = KMeans(n_clusters=3,init='k-means++', n_init=10, max_iter=300,tol=1e-04, random_state=0)
    param_grid = [{'kmeans__n_clusters': np.arange(2,6,1)}] 
    return pipeLineModelDesignClustering(X_train, None, model, param_grid)

def pipeLineModel_DBSCAN_Design(X_train):
    nFeatures = X_train.shape[1]
    print('nFeatures=',nFeatures)
    model = DBSCAN(eps=0.2, min_samples=6, metric='euclidean')
    
    #pca=5,eps=0.1,minsamples=6    bestk=4  #sse db csm 104.671905  1.413722  0.34593
    #pca=5,eps=0.2,minsamples=6    bestk=4  #sse db csm 286.959186  1.101676  0.694347
    #pca=5,eps=0.22,minsamples=6,7  bestk=3  #sse db csm 317.76992  1.147326  0.699217 
    #pca=5,eps=0.3, minsamples=95,96  bestk=2 #sse db csm 147.21034  1.881665  0.548519
    
    #pca=10,eps=0.2,minsamples=4    bestk=4  #sse db csm 96.238813  2.386751  0.239443
    #pca=7,eps=0.2,minsamples=3    bestk=4  #sse db csm 137.094597  1.974579  0.43688
    #pca=7,eps=0.28,minsamples=5    bestk=2  #sse db csm 160.974188  2.893825  0.55163
    
    param_grid = []
    #param_grid.append({'dbscan__eps': np.linspace(0.1, 0.5,20)}) #[0~nFeatures]
    param_grid.append({'dbscan__min_samples': np.arange(4,10,1)})
    return pipeLineModelDesignClustering(X_train, None, model, param_grid)

def pipeLineModel_Agglomerate_Design(X_train):
    #pca=12, n_clusters=bestk=2  #tts sse db csm  0.622431  109.154098  0.865911  0.515185
    #pca=7, n_clusters=bestk=2  #tts sse db csm  0.345117  81.432785  0.677056  0.447959
    #pca=5, n_clusters=bestk=2  #tts sse db csm  0.354799  47.857029  1.292439  0.589845
    #pca=8, n_clusters=bestk=2  #tts sse db csm  0.480715  88.761615  0.777836  0.45633
    
    #featureDecomposition = PCA(n_components=5)
    model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete') 
    param_grid = [{'agglomerativeclustering__n_clusters': range(2,6,1)}]  
    #param_grid.append({'pca__n_components': range(2,X_train.shape[1])})
    return pipeLineModelDesignClustering(X_train, None, model, param_grid)

'''  
def pipeLineModel_KMeans_Design(X_train):
    featureDecomposition = PCA(n_components=5)
    model = KMeans(n_clusters=2,init='k-means++', n_init=10, max_iter=300,tol=1e-04, random_state=0)
    param_grid = [{'kmeans__n_clusters': range(2,10)}]  #[2, 3, 4, 5, 6]
    #param_grid.append({'pca__n_components': range(2,X_train.shape[1])})
    return pipeLineModelDesignClustering(X_train, featureDecomposition, model, param_grid)

def pipeLineModel_DBSCAN_Design(X_train):
    featureDecomposition = PCA(n_components=5)
    model = DBSCAN(eps=0.3, min_samples=2*5, metric='euclidean')
    param_grid = [{'dbscan__eps': np.linspace(2.0,5.0,15)}]  
    param_grid.append({'dbscan__min_samples': np.arange(2,30,2)})
    #param_grid.append({'pca__n_components': range(2,X_train.shape[1])})
    return pipeLineModelDesignClustering(X_train, featureDecomposition, model, param_grid)

def pipeLineModel_Agglomerate_Design(X_train):
    featureDecomposition = PCA(n_components=5)
    model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete') 
    param_grid = [{'agglomerativeclustering__n_clusters': range(2,10)}]  
    #param_grid.append({'pca__n_components': range(2,X_train.shape[1])})
    return pipeLineModelDesignClustering(X_train, featureDecomposition, model, param_grid)
'''