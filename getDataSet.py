import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

# https://archive.ics.uci.edu/ml/datasets/Dow+Jones+Index#
# https://archive.ics.uci.edu/ml/datasets/Facebook+Live+Sellers+in+Thailand
# https://archive.ics.uci.edu/ml/datasets/Sales_Transactions_Dataset_Weekly
# https://archive.ics.uci.edu/ml/datasets/Water+Treatment+Plant

def getCsv(file,header=0,verbose=False):
    df = pd.read_csv(file,header=header)
    if verbose:
        print(df.describe().transpose())
        print(df.dtypes)
        print(df.columns)
        print(df.head())
    print('dataset=', df.shape)    
    return df

def getIrisDataset():
    df = pd.read_csv(r'./db/iris.data',header=None)
    df.columns = ['petal length', 'petal width',
                       'sepal length', 'sepal width',
                       'class']
    nrow, ncol = df.shape
    #print(df.dtypes)
    class_mapping = {
        'Iris-setosa':     0,
        'Iris-versicolor': 1,
        'Iris-virginica':  2
    }
    df['class'] = df['class'].map(class_mapping)

    N = -1#100#
    X, y = df.iloc[:N, :-1].values, df.iloc[:N, -1].values
    return X,y
    #return df.iloc[:, :-1]

def getDowJonesDataset(verbose=False):
    file = r'./db/dow_jones_index.data' 
    df = getCsv(file,verbose=verbose)
    
    def preProcessData(df):      
        #print(df.isnull())
        #print(df.isnull().sum())
        df = df.dropna()
        #'quarter','stock','date','open','high','low','close','volume','percent_change_price','percent_change_volume_over_last_wk','previous_weeks_volume','next_weeks_open','next_weeks_close','percent_change_next_weeks_price','days_to_next_dividend','percent_return_next_dividend'
        
        df = df[['quarter','stock','open','high','low','close','volume','percent_change_price','percent_change_volume_over_last_wk','previous_weeks_volume','next_weeks_open','next_weeks_close','percent_change_next_weeks_price','days_to_next_dividend','percent_return_next_dividend']]
        #df = df.iloc[:,3:-1] #start from column 'open' to last 
        #df[df.columns[1:]] = df[df.columns[1:]].apply(lambda x: x.str.replace(',',''))
        nrow, ncol = df.shape
        if verbose:
            print('\nBefore process:')
            print(df.columns)
            print(df.dtypes)
            print(df.head())
            
        # le = LabelEncoder()
        # le.fit(df['stock'])
        # df['stock'] = df['stock'].map(dict(zip(le.classes_, le.transform(le.classes_))))
        y = pd.get_dummies(df.stock, prefix='stock',dummy_na=False)
        print('y.shape=',y.shape)
        df = pd.concat([df,y],axis=1)
        df.drop(['stock'],axis=1, inplace=True)
        #print(df)
        
        for i in range(ncol):
            if df.dtypes[i] == object:
                if verbose:
                    print(i,df.columns[i])           
                
                df.iloc[:,i] = df.iloc[:,i].str.replace('$', '')
                #df.iloc[:,i] = df.iloc[:,i].apply(lambda x: x.replace('$',''))
                df.iloc[:,i] = pd.to_numeric(df.iloc[:,i])
                #df.iloc[:,i] = df.iloc[:,i].astype('float64')
        return df
    
    df = preProcessData(df)

    if verbose:
        print('\nAfter process:')
        print(df.columns)
        print(df.dtypes)
        print(df.head())
    print(df.shape)
    return df

def getWatertreatmentDataset(verbose=False):
    file = r'./db/water-treatment.data'
    df = getCsv(file,header=None,verbose=verbose)
    
    df.columns = ['Date', 
                  'Q-E', 'ZN-E', 'PH-E','DBO-E', 'DQO-E', 'SS-E', 'SSV-E', 'SED-E',
                  'COND-E', 'PH-P', 'DBO-P', 'SS-P', 'SSV-P', 'SED-P', 'COND-P', 'PH-D', 
                   'DBO-D', 'DQO-D', 'SS-D', 'SSV-D', 'SED-D', 'COND-D', 'PH-S', 'DBO-S', 
                   'DQO-S', 'SS-S', 'SSV-S', 'SED-S', 'COND-S', 'RD-DBO-P', 'RD-SS-P', 'RD-SED-P',  
                   'RD-DBO-S', 'RD-DQO-S', 'RD-DBO-G', 'RD-DQO-G', 'RD-SS-G', 'RD-SED-G']
    
    def preProcessData(df):                  
        #print(df.isnull())
        #print(df.isnull().sum())
        
        if 0: #replace 
            df.loc[:,:] = df.loc[:,:].replace('?', 0.0)
        else: #NaN
            df.loc[:,:] = df.loc[:,:][~(df.loc[:,:] == '?')]
        
        #print(df.isnull().sum())
        #df = df.dropna()
        #print(df.isnull().sum())
    
        df = df.iloc[:,1:] 
        nrow, ncol = df.shape
        if verbose:
            print('\nBefore process:')
            print(df.shape)
            print(df.columns)
            print(df.dtypes)
            print(df.head())
      
        for i in range(ncol):
            if df.dtypes[i] == object:
                if verbose:
                    print(i,df.columns[i])           
                #df.iloc[:,i] = df.iloc[:,i].str.replace('$', '')
                #df.iloc[:,i] = df.iloc[:,i].apply(lambda x: x.replace('$',''))
                #df.iloc[:,i] = pd.to_numeric(df.iloc[:,i])
                df.iloc[:,i] = df.iloc[:,i].astype('float64')
        
        # print('??=',df)
        # print('mean=',df.mean())
        #df = df.fillna(df.mean())
        df = df.fillna(0)
        return df
    
    df = preProcessData(df)
    
    if verbose:
        print('\nAfter process:')
        print(df.columns)
        print(df.dtypes)
        print(df.head())
    print(df.shape)
    return df

def getFacebookLiveDataset(verbose=False):
    file = r'./db/Live.csv'
    df = getCsv(file,verbose=verbose)
        
    def preProcessData(df):    
        #'status_id', 'status_type', 'status_published', 'num_reactions', 'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas', 'num_sads', 'num_angrys', 'Column1', 'Column2', 'Column3', 'Column4'
        #df = df.iloc[:, 3:-5] 
        df = df[['status_type', 'num_reactions',
       'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows',
       'num_hahas', 'num_sads', 'num_angrys']]
        
        #le = LabelEncoder()
        #le.fit(df['status_type'])
        #df['status_type'] = df['status_type'].map(dict(zip(le.classes_, le.transform(le.classes_))))
         
        y = pd.get_dummies(df.status_type, prefix='status_type',dummy_na=False)
        print('y.shape=',y.shape)
        df = pd.concat([df,y],axis=1)
        df.drop(['status_type'],axis=1, inplace=True)
                    
        #print(df.isnull())
        #print(df.isnull().sum())
        df = df.dropna()      

        nrow, ncol = df.shape
        if verbose:
            print('\nBefore process:')
            print(df.shape)
            print(df.columns)
            print(df.dtypes)
            print(df.head())
      
        for i in range(ncol):
            if df.dtypes[i] == object:
                if verbose:
                    print(i,df.columns[i])           
                df.iloc[:,i] = df.iloc[:,i].astype('float64')
        return df
    
    df = preProcessData(df)
    
    if verbose:
        print('\nAfter process:')
        print(df.columns)
        print(df.dtypes)
        print(df.head())
    print(df.shape)
    return df

def getSalesTransactionsDataset(verbose=False):
    file = r'./db/Sales_Transactions_Dataset_Weekly.csv'
    df = getCsv(file,verbose=verbose)
    
    def preProcessData(df):                  
        #print(df.isnull())
        #print(df.isnull().sum())
        #f = df.dropna()
        #print(df.isnull().sum())
        df = df.iloc[:,1:53] 
        nrow, ncol = df.shape
        if verbose:
            print('\nBefore process:')
            print(df.shape)
            print(df.columns)
            print(df.dtypes)
            print(df.head())
      
        for i in range(ncol):
            if df.dtypes[i] == object:
                if verbose:
                    print(i,df.columns[i])           
                #df.iloc[:,i] = df.iloc[:,i].str.replace('$', '')
                #df.iloc[:,i] = df.iloc[:,i].apply(lambda x: x.replace('$',''))
                #df.iloc[:,i] = pd.to_numeric(df.iloc[:,i])
                #df.iloc[:,i] = df.iloc[:,i].astype('float64')
        return df
    
    df = preProcessData(df)
    
    if verbose:
        print('\nAfter process:')
        print(df.columns)
        print(df.dtypes)
        print(df.head())
    print(df.shape)
    return df

def main():
    #getDowJonesDataset(True)
    #getWatertreatmentDataset(True)
    #getFacebookLiveDataset(True)
    getSalesTransactionsDataset(True)
    
if __name__ == "__main__":
    main()


