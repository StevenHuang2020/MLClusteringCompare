import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from getDataSet import getCsv
from plotCommon import *
from sklearn.metrics import silhouette_samples
from matplotlib import cm

gSaveBase=r'./res/'

def plotModelCSM(datasetName,modelName,df): #CSM
    x = df.loc[:,['K']].values #K
    csm = df.loc[:,['CSM']]
    
    #plt.figure(figsize=(8,5))
    ax = plt.subplot(1,1,1)
    title = datasetName+ '_' + modelName + '_CSM'
    plt.title(title)
    plotSub(x,csm,ax,label='csm',c='r')
    
    plt.xticks(np.arange(1, 12))
    ax.set_ylabel('CSM')
    ax.set_xlabel('K clusters')
    ax.legend()
    ax.grid()
    plt.savefig(gSaveBase+title+'.png')
    plt.show()

def plotModel(datasetName,modelName,df): #sse sse gredient
    #df = df.loc[:,['K', 'tt(s)', 'SSE','DB','CSM']]
    #print(df.iloc[:,[0,1,2,4]]) #no db
    
    x = df.loc[:,['K']].values #K
    y = df.loc[:,['SSE']].values #SSE
    z = np.zeros((len(x)))
    for i in range(len(x)-1):
        z[i+1] = y[i] - y[i+1]
        
    #plt.figure(figsize=(8,5))
    ax = plt.subplot(1,1,1)
    title = datasetName+ '_' + modelName + '_SSE'
    plt.title(title)
    plotSub(x,y,ax,label='sse',c='k')
    scatterSub(x,y,ax,marker='o',c='g')
    plotSub(x,z,ax,label='sse decrease gradient',c='b')
    ax.set_ylabel('SSE')
    ax.set_xlabel('K clusters')
    ax.legend()
    ax.grid()
    plt.xticks(np.arange(1, 12))
    plt.savefig(gSaveBase+title+'.png')
    plt.show()
    
def plotSilhouetteValues(datasetName,modelName,k, X, y_km):
    if k<=1:
        return
    
    ax = plt.subplot(1,1,1)
    title = datasetName+ '_' + modelName + '_CSM_k=' +str(k)
    plt.title(title)
    
    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    
    print('cluster_labels=',cluster_labels,',n_clusters=',n_clusters)
    
    
    silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km == c]
        c_silhouette_vals.sort()
        
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i) / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
                edgecolor='none', color=color)

        yticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouette_vals)
        
    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color="red", linestyle="--") 

    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')

    plt.tight_layout()
    plt.savefig(gSaveBase+title+'.png')
    plt.show()
    
def plotTimeTaken(df): #sse sse gredient
    print(df)
    df = df.loc[:,['Dataset','tt(s)']]
    df.set_index(["Dataset"], inplace=True)
    print(df)
        
    #plt.figure(figsize=(8,5))
    #ax = plt.subplot(1,1,1)
    #title = 'tt(s)'
    #plt.title(title)
    
    ax = df.plot(kind='line')
    ax.set_title('time taken(s)')
    fontsize=8
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right",fontsize=fontsize)
    plt.setp(ax.get_yticklabels(),fontsize=fontsize)
    #plt.subplots_adjust(left=0.07, bottom=0.16, right=0.96, top=0.94, wspace=None, hspace=None)
    
    ax.legend()
    ax.grid()
    #plt.savefig(gSaveBase+title+'.png')
    plt.show()
    
def plotThree(df1,df2,df3,selColumn='tt(s)',xlabel='Datasets',ylabel='Time taken(s)',saveName='timetaken'): 
    d = {'Dataset': df1['Dataset'], 'KMeans': df1[selColumn],'Agglomerative': df2[selColumn],'DBSCAN': df3[selColumn]}
    #print('d=',d)
    df = pd.DataFrame(data=d)
    df.set_index(["Dataset"], inplace=True)
    print('okdf=',df)
    ax = df.plot(kind='bar') #line
    fontsize=8
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right",fontsize=fontsize)
    plt.setp(ax.get_yticklabels(),fontsize=fontsize)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(gSaveBase+saveName+'.png')
    plt.show()
    
def plotThreeAll(df1,df2,df3):
    df = pd.concat([df1, df2, df3])
    #print(df)
    df = df.loc[:,['Dataset','Algorithm','K','SSE','CSM','tt(s)']]    
    df = df.sort_values(by=['Dataset'],ascending=True)
    #df.set_index(["Dataset"], inplace=True)
    print(df)
    
    df1 = df[df['Dataset'] == 'DowJones(720, 5)']
    print(df1)
    calculateDf('DowJones',df1)

    df2 = df[df['Dataset'] == 'Watertreatment(527, 5)']
    print(df2)
    calculateDf('Watertreatment',df2)
    df3 = df[df['Dataset'] == 'FacebookLive(7050, 5)']
    print(df3)
    calculateDf('FacebookLive',df3)
    df4 = df[df['Dataset'] == 'SalesTransactions(811, 5)']
    print(df4)
    calculateDf('SalesTransactions',df4)
 
def softmaxFuc(x):
    return np.exp(x)/np.sum(np.exp(x))
   
def calculateDf(name,df):
    print('----------------------------',name)
    s_sse = np.round(1-softmaxFuc(df['SSE'].values), 4)
    print('s_sse=',s_sse)
    s_csm = np.round(softmaxFuc(df['CSM'].values),4)
    print('s_csm=',s_csm)
    s_tt = np.round(1-softmaxFuc(df['tt(s)'].values), 4)
    print('s_tt=',s_tt)
    
    total = 0.5*s_sse + 0.3*s_csm + 0.2*s_tt
    print('total=',total)
    
def plotTimeTakenAll():
    df1 = getCsv('./bestk_sseKMeans.csv')
    print(df1)
    df2 = getCsv('./bestk_alg.csv')
    print(df2)
    df3 = getCsv('./bestk_DBSCAN.csv')
    print(df3)
    
    #plotThree(df1,df2,df3)
    #plotThree(df1,df2,df3,selColumn='CSM',ylabel='CSM',saveName='CSM')
    #plotThree(df1,df2,df3,selColumn='SSE',ylabel='SSE',saveName='SSE')
    plotThreeAll(df1,df2,df3)
    
def main():
    #df = getCsv('./res/DowJones_KMeans_result.csv')
    #print(df)
    #df = df.iloc[:,1:]
    #plotModel('KMeans','Jons',df)
    
    #df = getCsv('./bestk_sse.csv')
    #plotTimeTaken(df)
    plotTimeTakenAll()
    
if __name__=='__main__':
    main()
    