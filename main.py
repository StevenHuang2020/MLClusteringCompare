#unicode Python3 Steven
#09/05/2020
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.neighbors.nearest_centroid import NearestCentroid

from getDataSet import getDowJonesDataset,getWatertreatmentDataset
from getDataSet import getFacebookLiveDataset,getSalesTransactionsDataset,getIrisDataset
from modelCreate import createKMeans,createAgglomerate,createDBSCAN
from modelCreate import pipeLineModel_KMeans_Design
from modelCreate import pipeLineModel_DBSCAN_Design
from modelCreate import pipeLineModel_Agglomerate_Design
from modelCreate import calculateSSE,calculateDaviesBouldin
from visualClustering import visualClusterResult
from plotApplication import*

def trainAll():
    datasets = prepareDataSet()
    dfBest = pd.DataFrame()
    for i in datasets:
        bestLine = trainModel(i[0],i[1])
        dfBest = dfBest.append(bestLine,ignore_index=True)
    
    print('-------------------best-------------')
    dfBest.set_index(["Dataset"], inplace=True)
    print(dfBest)
    dfBest.to_csv('bestk.csv',index=True)
    
def getModelMeasureByModel(data,model):
    return getModelMeasure(data,model.labels_)

def getModelMeasure(data,labels):
    sse,dbValue,csm = 0,0,0     
    #k = len(np.unique(labels))
    k = len(list(set(labels)))
    if k>1:
        #print(data.shape,model.labels_)
        csm = silhouette_score(data, labels, metric='euclidean')
        clf = NearestCentroid()
        clf.fit(data, labels)
        #print(clf.centroids_)
        sse = calculateSSE(data,labels,clf.centroids_)
        dbValue = calculateDaviesBouldin(data,labels)
    
    sse = round(sse,4)
    csm = round(csm,4)
    dbValue = round(dbValue,4)
    print('SSE=', sse,'DB=',dbValue,'CSM=',csm,'clusters=',k)    
    #print("Silhouette Coefficient: %0.3f" % csm)
    #print('clusters=',k)
    return sse,dbValue,csm,k    

def descriptData(data): #after preprocessing
    print('data.shape=',data.shape)
    s = 0
    for i in range(data.shape[1]):
        column = data[:,[i]]
        min = np.min(column)
        max = np.max(column)
        dis = np.abs(min-max)
        print("column:",i,'min,max,dis=',min,max,dis)
        s += dis
    print('s=',s)

def preprocessingData(data,N=5):        
    scaler = MinMaxScaler()# StandardScaler() #
    # scaler.fit(data)
    # data = scaler.transform(data)
    data = scaler.fit_transform(data)
    
    #print('\n',data[:5])    
    #print('scaler=',data[:5])
    if 1:
        fit = PCA(n_components=N).fit(data)
        data = fit.transform(data)
        #print("Explained Variance: %s" % (fit.explained_variance_))
        #print("Explained Variance ratio: %s" % (fit.explained_variance_ratio_))
        #print(fit.components_)
        #print('after PCA features.shape = ', data.shape)
        #print(data[:5])
        
    descriptData(data)
    return data

def trainModel(dataName,data,N=12): 
    data = preprocessingData(data)
    df = pd.DataFrame()
    columns=['Dataset', 'Algorithm', 'K', 'tt(s)', 'SSE','DB','CSM']
    #columns=['K','tt(s)','SSE','DB','CSM']
    for i in range(2,N,1):  #2 N 1          
        if 0:
            model = createKMeans(i)
            modelName='K-Means'
        elif 0:
            model = createAgglomerate(i)
            modelName='Agglomerative'
        else:
            if dataName == 'DowJones':
                minEps = 0.10 #0.24
                maxEps = 0.24 
                samples = 10
            elif dataName == 'Watertreatment':
                minEps = 0.15 #0.30
                maxEps = 0.30
                samples = 10 
            elif dataName == 'FacebookLive':
                minEps = 0.008 #0.0275 best
                maxEps = 0.03
                samples = 10
            elif dataName == 'SalesTransactions':
                minEps = 0.15 #0.248
                maxEps = 0.248
                samples = 10
                                                
            model = createDBSCAN(eps=minEps+(i-2)*((maxEps-minEps)/(N-3)), min_samples=samples)
            modelName='DBSCAN'
        
        t = time()
        model.fit(data)
        
        tt = round(time()-t, 4)
        print("\ndataSet:%s model:%s iter i=%d run in %.2fs" % (dataName,modelName,i,tt))
        sse,dbValue,csm,k = getModelMeasure(data,model.labels_)
        
        if modelName == 'DBSCAN':
            labels = model.labels_
            noiseN = len(np.where(labels==-1)[0])
            #print(noiseN)
            print('noise:',noiseN,len(labels),round(noiseN/len(labels),4),'k=',k)
            if k<=2:
                continue
            
        dbName = dataName + str(data.shape)
        line = pd.DataFrame([[dbName, modelName, k, tt,sse,dbValue,csm]], columns=columns)
        df = df.append(line,ignore_index=True)
        #visualClusterResult(data,model.labels_,k,modelName+'_K_'+str(k))
        #plotSilhouetteValues(dataName,modelName,k, data, model.labels_)
        #print('cluster_labels=',np.unique(model.labels_))
    
    #df.to_csv(gSaveBase + dataName+'_' + modelName+'_result.csv',index=True)
    #print('Train result:\n',df)
    #plotModel(dataName,modelName,df)
    #plotModelCSM(dataName,modelName,df)
    
    #index,bestK = getBestkFromSse(dataName,modelName,df)
    index,bestK = getBestkFromCSM(dataName,modelName,df)
    
    bestLine = df.iloc[index,:]
    #print('bestLine=',index,'bestK=',bestK,'df=\n',bestLine)
    return bestLine

    #df = df.sort_values(by=['CSM'],ascending=False)
    #print('Train result order by CSM:\n',df)
    #df = df.sort_values(by=['SSE'],ascending=True)
    #print('Train result order by SSE:\n',df)
    #return df.iloc[0,:]
    
def getBestkFromSse(datasetName,modelName,df): #sse gradient
    print('df=\n',df)
    x = df.loc[:,['K']].values #K
    y = df.loc[:,['SSE']].values #SSE
    z = np.zeros((len(x))) #gradient

    for i in range(len(x)-1):
        z[i+1] = y[i] - y[i+1]
        
    index = np.argmax(z)
    bestK = x[index][0]
    print('z=',z,index,bestK)
    return index,bestK
    
def getBestkFromCSM(datasetName,modelName,df):
    print('df=\n',df)
    x = df.loc[:,['K']].values #K
    y = df.loc[:,['CSM']].values #CSM
 
    index = np.argmax(y)
    bestK = x[index][0]
    print('index,bestK=',index,bestK)
    return index,bestK
 
def clusteringPipline(data):
    #return pipeLineModel_KMeans_Design(data)
    #return pipeLineModel_DBSCAN_Design(data)
    return pipeLineModel_Agglomerate_Design(data)
    
def prepareDataSet():
    dataset = []   
    #dataset.append(('DowJones',getDowJonesDataset()))
    #dataset.append(('Watertreatment', getWatertreatmentDataset()))
    #dataset.append(('FacebookLive',getFacebookLiveDataset()))
    dataset.append(('SalesTransactions', getSalesTransactionsDataset()))
    return dataset

def pipLinesDesign():
    pipLines = []
    #pipLines.append(('KMeans',pipeLineModel_KMeans_Design))
    pipLines.append(('DBSCAN',pipeLineModel_DBSCAN_Design))
    #pipLines.append(('Agglomerative',pipeLineModel_Agglomerate_Design))
    return pipLines

def prepareDataSetWithPreprocess(N=5):
    dataset = []       
    dataset.append(('DowJones',preprocessingData(getDowJonesDataset(),N=N)))
    #dataset.append(('Watertreatment',preprocessingData(getWatertreatmentDataset(),N=N)))
    #dataset.append(('FacebookLive',preprocessingData(getFacebookLiveDataset(),N=N)))
    #dataset.append(('SalesTransactions',preprocessingData(getSalesTransactionsDataset(),N=N)))
    return dataset

def trainPipline():
    print("*"*20,'train start','*'*20)
    t = time()
    dataset = prepareDataSetWithPreprocess(N=5) #prepareDataSet() #
    
    pipLines = pipLinesDesign()
    
    df = pd.DataFrame()
    columns=['Dataset', 'Algorithm', 'Best-K', 'tt(s)', 'SSE','DB','CSM']
    for pipLine in pipLines:
        for i in dataset:
            data = i[1]
            dataName = i[0]
            dbName = dataName + str(data.shape)
    
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
            plotSilhouetteValues(dataName,algorithmName,bestK, data, labels)
            #break
        
    df['Best-K'] = df['Best-K'].astype('int64')
    #df.set_index(["Dataset"], inplace=True)
    print(df)
    df = df.sort_values(by=['Dataset'],ascending=True)
    print('\n----------order by Dataset----------\n',df)
    print("\nTotal run in %.2fs" % (time()-t))
    
def trainSalesTransactions():
    data = getSalesTransactionsDataset()
    data = preprocessingData(data)

    if 1: 
        k=3
        model = createKMeans(k)
        modelName = 'KMeans'
    elif 0:
        k=2        
        model = createAgglomerate(k)
        modelName = 'Agglomerate'
    elif 0:
        model = createDBSCAN(eps=0.247,min_samples=10) #0.248
        modelName = 'DBSCAN'
                
    model.fit(data)
    sse,dbValue,csm,k = getModelMeasure(data,model.labels_)
    
    if modelName == 'DBSCAN': #remove -1 label
        k-=1

    visualClusterResult(data,model.labels_,k,'SalesTransactions_'+modelName)
    
def main():
    #trainPipline()
    #trainAll()
    trainSalesTransactions()
    
if __name__ == "__main__":
    main()
    