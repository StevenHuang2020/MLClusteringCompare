#unicode Python3 Steven
#09/05/2020
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import precision_score, recall_score,auc
# from sklearn.metrics import roc_curve,roc_auc_score, plot_roc_curve
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from getDataSet import getDowJonesDataset,getWatertreatmentDataset
from getDataSet import getFacebookLiveDataset,getSalesTransactionsDataset,getIrisDataset
from sklearn.neighbors.nearest_centroid import NearestCentroid
from modelCreate import createKMeans,createAgglomerate,createDBSCAN
from modelCreate import pipeLineModel_KMeans_Design
from modelCreate import pipeLineModel_DBSCAN_Design
from modelCreate import pipeLineModel_Agglomerate_Design
from modelCreate import calculateSSE,calculateDaviesBouldin
from visualClustering import visualClusterResult


def trainTest():
    #rawdata = getWatertreatmentDataset() #getDowJonesDataset()
    #rawdata = getFacebookLiveDataset() #getSalesTransactionsDataset() #
    #rawdata = getIrisDataset()
    rawdata = getDowJonesDataset()
    #print('data.shape=',rawdata.shape)
    trainModel(rawdata)
    #trainPipline(rawdata)
    
def getModelMeasureByModel(data,model):
    sse,dbValue,csm = 0,0,0     
    k = len(list(set(model.labels_)))
    if k>1:
        #print(data.shape,model.labels_)
        csm = silhouette_score(data, model.labels_, metric='sqeuclidean')
        clf = NearestCentroid()
        clf.fit(data, model.labels_)
        #print(clf.centroids_)
        sse = calculateSSE(data,model.labels_,clf.centroids_)
        dbValue = calculateDaviesBouldin(data,model.labels_)
        
    print('SSE=', sse,'DB=',dbValue,'CSM=',csm,'clusters=',k)    
    #print("Silhouette Coefficient: %0.3f" % csm)
    #print('clusters=',k)
    return sse,dbValue,csm,k

def getModelMeasure(data,labels):
    sse,dbValue,csm = 0,0,0     
    k = len(list(set(labels)))
    if k>1:
        #print(data.shape,model.labels_)
        csm = silhouette_score(data, labels, metric='sqeuclidean')
        clf = NearestCentroid()
        clf.fit(data, labels)
        #print(clf.centroids_)
        sse = calculateSSE(data,labels,clf.centroids_)
        dbValue = calculateDaviesBouldin(data,labels)
        
    print('SSE=', sse,'DB=',dbValue,'CSM=',csm,'clusters=',k)    
    #print("Silhouette Coefficient: %0.3f" % csm)
    #print('clusters=',k)
    return sse,dbValue,csm,k    

def preprocessingData(data,N=5):
    fit = PCA(n_components=N).fit(data)
    data = fit.transform(data)
    
    #print("Explained Variance: %s" % (fit.explained_variance_))
    #print("Explained Variance ratio: %s" % (fit.explained_variance_ratio_))
    #print(fit.components_)
    #print('after PCA features.shape = ', data.shape)
    #print(data[:5])
    scaler = StandardScaler() #MinMaxScaler()#
    scaler.fit(data)
    data = scaler.transform(data)
    #print('\n',data[:5])    
    return data

def trainModel(data,N=20):
    data = preprocessingData(data)

    df = pd.DataFrame()
    columns=['Best-K','tt(s)','SSE','DB','CSM']
    for i in range(N):
        wantK = 2*i+2
        if 1:
            model = createDBSCAN(eps=(wantK)*0.2+2,samples=2*data.shape[1])
            modelName='DBSCAN'
        elif 0:
            model = createKMeans(wantK)
            modelName='KMeans'
        else:
            model = createAgglomerate(wantK)
            modelName='Agglomerative'
        
        t = time()
        model.fit(data)
        tt = time()-t
        print("\nmodel:%s iter i=%d run in %.2fs" % (modelName,i,tt))
        sse,dbValue,csm,k = getModelMeasureByModel(data,model)
       
        '''    
        if 1: #kMeans
            sse = model.inertia_
            print('Sum of Squares Errors (SSE) inertia_=', sse, model.inertia_/len(data)) #sse
            print('cluster_centers_:',model.cluster_centers_)
            print('n_iter_:',model.n_iter_)
            print('inertia_:',model.inertia_)
        elif 0:#DBSCAN
            #print('core_sample_indices_:',model.core_sample_indices_)
            #print('components_:',model.components_)
            print('labels_:',model.labels_)
        else:#AgglomerativeClustering
            print('n_clusters_:',model.n_clusters_)
            print('n_leaves_:',model.n_leaves_)
            print('n_connected_components_:',model.n_connected_components_)
            #print('children_:',model.children_)
            #print('labels_:',model.labels_)
        '''
         
        line = pd.DataFrame([[k, tt,sse,dbValue,csm]], columns=columns)
        df = df.append(line,ignore_index=True)

        visualClusterResult(data,model.labels_,k,modelName+'_K_'+str(k))
            
    print('Train result:\n',df)
    df = df.sort_values(by=['CSM'],ascending=False)
    print('Train result order by CSM:\n',df)
    df = df.sort_values(by=['SSE'],ascending=True)
    print('Train result order by SSE:\n',df)
    
    #return df.iloc[0,:]
    
def trainPipline(data):
    #return pipeLineModel_KMeans_Design(data)
    #return pipeLineModel_DBSCAN_Design(data)
    return pipeLineModel_Agglomerate_Design(data)
    
def prepareDataSet():
    dataset = []   
    dataset.append(('DowJones',getDowJonesDataset()))
    #dataset.append(('Watertreatment', getWatertreatmentDataset()))
    #dataset.append(('FacebookLive',getFacebookLiveDataset()))
    #dataset.append(('SalesTransactions', getSalesTransactionsDataset()))
    return dataset

def pipLinesDesign():
    pipLines = []
    pipLines.append(('KMeans',pipeLineModel_KMeans_Design))
    pipLines.append(('DBSCAN',pipeLineModel_DBSCAN_Design))
    pipLines.append(('Agglomerative',pipeLineModel_Agglomerate_Design))
    return pipLines

def prepareDataSetWithPreprocess(N=5):
    dataset = []       
    #dataset.append(('DowJones',preprocessingData(getDowJonesDataset(),N=N)))
    #dataset.append(('Watertreatment',preprocessingData(getWatertreatmentDataset(),N=N)))
    dataset.append(('FacebookLive',preprocessingData(getFacebookLiveDataset(),N=N)))
    #dataset.append(('SalesTransactions',preprocessingData(getSalesTransactionsDataset(),N=N)))
    return dataset

def train():
    print("*"*20,'train start','*'*20)
    t = time()
    dataset = prepareDataSetWithPreprocess(5) #prepareDataSet()
    pipLines = pipLinesDesign()
    
    df = pd.DataFrame()
    columns=['Dataset', 'Algorithm', 'Best-K', 'tt(s)', 'SSE','DB','CSM']
    
    for pipLine in pipLines:
        for i in dataset:
            data = i[1]
            dbName = i[0] + str(data.shape)
    
            algorithmName = pipLine[0]
            print('start training,dataset=',dbName,'algorithm=',algorithmName,',please wait...')
            res,labels,bestK = pipLine[1](data)
            #print('res=',res)
            
            oneLine = []
            oneLine.append(dbName)
            oneLine.append(algorithmName)
            oneLine.extend(list(res.values.flatten()))
            #print('oneLine=',oneLine)
            line = pd.DataFrame([oneLine], columns=columns)
            df = df.append(line,ignore_index=True)
            
            #visualClusterResult(data, labels, bestK, algorithmName + '_K_'+str(bestK))
            #break
        
    df['Best-K'] = df['Best-K'].astype('int64')
    #df.set_index(["Dataset"], inplace=True)
    print(df)
    df = df.sort_values(by=['Dataset'],ascending=True)
    print('\n----------order by Dataset----------\n',df)
    print("\nTotal run in %.2fs" % (time()-t))
    
def main():
    train()
    #trainTest()
    
if __name__ == "__main__":
    main()

#---------------------------------oupt put-------------------------------  
    
    '''#with PCA=5 MinMaxScaler
                      Dataset      Algorithm  Best-K      tt(s)          SSE        DB       CSM
0            DowJones(720, 5)         KMeans       3   1.068178    35.432664  1.027498  0.581384
1      Watertreatment(380, 5)         KMeans       2   1.058096    23.787674  1.941314  0.496058
2       FacebookLive(7050, 5)         KMeans       2   2.250669    32.212476  0.659134  0.912606
3   SalesTransactions(811, 5)         KMeans       2   0.926352    36.505490  0.722811  0.744269
4            DowJones(720, 5)         DBSCAN       4   0.371581   286.959186  1.101676  0.694347
5      Watertreatment(380, 5)         DBSCAN       2   0.157577    56.195588  2.457110  0.549143
6       FacebookLive(7050, 5)         DBSCAN       2  18.885948  1570.566869  1.592059  0.774685
7   SalesTransactions(811, 5)         DBSCAN       3   0.393871   365.026548  4.470893  0.791441
8            DowJones(720, 5)  Agglomerative       2   0.183655    47.857029  1.292439  0.589845
9      Watertreatment(380, 5)  Agglomerative       2   0.071812    26.500204  0.914561  0.738794
10      FacebookLive(7050, 5)  Agglomerative       2  23.589775    51.927974  0.181741  0.972497
11  SalesTransactions(811, 5)  Agglomerative       3   0.220941    57.048722  0.911441  0.748078

-------------------------------------


----------order by Dataset----------
                       Dataset      Algorithm  Best-K      tt(s)          SSE        DB       CSM
0            DowJones(720, 5)         KMeans       3   1.068178    35.432664  1.027498  0.581384
4            DowJones(720, 5)         DBSCAN       4   0.371581   286.959186  1.101676  0.694347
8            DowJones(720, 5)  Agglomerative       2   0.183655    47.857029  1.292439  0.589845
2       FacebookLive(7050, 5)         KMeans       2   2.250669    32.212476  0.659134  0.912606
6       FacebookLive(7050, 5)         DBSCAN       2  18.885948  1570.566869  1.592059  0.774685
10      FacebookLive(7050, 5)  Agglomerative       2  23.589775    51.927974  0.181741  0.972497
3   SalesTransactions(811, 5)         KMeans       2   0.926352    36.505490  0.722811  0.744269
7   SalesTransactions(811, 5)         DBSCAN       3   0.393871   365.026548  4.470893  0.791441
11  SalesTransactions(811, 5)  Agglomerative       3   0.220941    57.048722  0.911441  0.748078
1      Watertreatment(380, 5)         KMeans       2   1.058096    23.787674  1.941314  0.496058
5      Watertreatment(380, 5)         DBSCAN       2   0.157577    56.195588  2.457110  0.549143
9      Watertreatment(380, 5)  Agglomerative       2   0.071812    26.500204  0.914561  0.738794

Total run in 52.66s

    '''
    
    
    
    '''#without PCA
                        Dataset      Algorithm  Best-K       tt(s)           SSE       CSM
0             DowJones(720, 12)         KMeans       3    2.979096  1.579917e+19  0.167706
1       Watertreatment(380, 37)         KMeans       2    2.893579  1.769930e+10  0.205539
2         FacebookLive(7050, 8)         KMeans       2    6.062922  5.739497e+09  0.851037
3   SalesTransactions(811, 105)         KMeans       2    4.387590  1.461950e+06  0.805480
4             DowJones(720, 12)         DBSCAN       3    2.667292  1.185720e+20  0.371011
5       Watertreatment(380, 37)         DBSCAN       2    2.164607  1.310175e+10  0.088081
6         FacebookLive(7050, 8)         DBSCAN       2  104.262294  1.086716e+11  0.920020
7   SalesTransactions(811, 105)         DBSCAN       1   15.084448  0.000000e+00  0.000000
8             DowJones(720, 12)  Agglomerative       3    0.518780  1.643891e+19  0.066800
9       Watertreatment(380, 37)  Agglomerative       2    0.327166  1.801777e+10  0.166698
10        FacebookLive(7050, 8)  Agglomerative       2   44.312761  6.636813e+09  0.826064
11  SalesTransactions(811, 105)  Agglomerative       2    1.391160  1.545233e+06  0.804826

------------------------------------------------------

                        Dataset      Algorithm  Best-K       tt(s)           SSE       CSM
0             DowJones(720, 12)         KMeans       3    2.979096  1.579917e+19  0.167706
4             DowJones(720, 12)         DBSCAN       3    2.667292  1.185720e+20  0.371011
8             DowJones(720, 12)  Agglomerative       3    0.518780  1.643891e+19  0.066800
2         FacebookLive(7050, 8)         KMeans       2    6.062922  5.739497e+09  0.851037
6         FacebookLive(7050, 8)         DBSCAN       2  104.262294  1.086716e+11  0.920020
10        FacebookLive(7050, 8)  Agglomerative       2   44.312761  6.636813e+09  0.826064
3   SalesTransactions(811, 105)         KMeans       2    4.387590  1.461950e+06  0.805480
7   SalesTransactions(811, 105)         DBSCAN       1   15.084448  0.000000e+00  0.000000
11  SalesTransactions(811, 105)  Agglomerative       2    1.391160  1.545233e+06  0.804826
1       Watertreatment(380, 37)         KMeans       2    2.893579  1.769930e+10  0.205539
5       Watertreatment(380, 37)         DBSCAN       2    2.164607  1.310175e+10  0.088081
9       Watertreatment(380, 37)  Agglomerative       2    0.327166  1.801777e+10  0.166698

Total run in 190.61s
    '''
    