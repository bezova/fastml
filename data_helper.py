import pandas as pd
from numpy import nan as NaN
from sklearn.neighbors import KNeighborsRegressor
from typing import List
# disable pandas chain assignment warning
pd.options.mode.chained_assignment = None  # default='warn'
# discussion here
# https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas

def low_high_quantile(ser:pd.Series, low:float, high=None)->List[float]:
    ''' returns values for quantiles
    symmetric if High is omitted'''
    if high is None: high = 1-low
    return ser.quantile(low), ser.quantile(high)

def nan_quantile(df:pd.DataFrame, col:str, low:float, high=None):
    '''  In PLACE operation
    modyfies df with quantiles set to nan (low, high) for col
    quantiles '''
    bottom, top = low_high_quantile(df[col], low, high)
    df.loc[(df[col]>=top)|(df[col]<=bottom), col] = NaN

def nan_quantile_ref(df:pd.DataFrame, df_ref:pd.DataFrame, col:str, low:float, high=None):
    '''  In PLACE operation
    modyfies df with quantiles set to nan (low, high) for col
    quantiles calculated from refereced df_ref '''
    bottom, top = low_high_quantile(df_ref[col], low, high)
    df.loc[(df[col]>=top)|(df[col]<=bottom), col] = NaN


def knn_col_by_XY(df, col, cond_to_predict=None, LatLon=['Latitude', 'Longitude']):
    ''' In PLACE operation
    use closesst geographic neighbors to fill in missibng values base
    col - column to fill
    if condition=None, fill in all NaN values in the column
     'example'
        col = 'Elevation';
        cond_to_predict = (df[col].isna() | (df[col]<100) | (df[col]>1000))
        knn_col(df, col, cond_to_predict)    
    '''
    if cond_to_predict is None: cond_to_predict=df[col].isna()
    if len(df[cond_to_predict])==0: return None
    XY = df.loc[~cond_to_predict, LatLon+[col]]
    if len(XY)==0: return None
    knn = KNeighborsRegressor(2, weights='distance')
    _=knn.fit(XY[LatLon], XY[col])
    df.loc[cond_to_predict, col] = knn.predict(df.loc[cond_to_predict,LatLon])

def unknown_to_nan(df:pd.DataFrame, list_to_nan=['UNKNOWN']):
    ''' in Place Operation
    for categorical collumns rennames certain values ('UNKNOWN'))  -> Nan'''
    for col in df.columns[df.dtypes == 'object']: 
        for name in list_to_nan:
            df.loc[df[col]==name, col] = NaN

def nan_to_uknown(df:pd.DataFrame, unknownName='UNKNOWN'):
    ''' in Place Operation
    for categorical collumns rennames NaN to unknownName'''
    for col in df.columns[df.dtypes == 'object']: 
                df.loc[df[col].isna(), col] = unknownName

def unknowns_to_sameName(df:pd.DataFrame, unknownName='UNKNOWN', list_to_nan=['UNKNOWN']):
    ''' IN PLace Opeation
    rename all unknowns (list_to_nan) and NaN categorical values to same unknownName'''
    unknown_to_nan(df, list_to_nan)
    nan_to_uknown(df, unknownName)

def geo_con(df, gf, gflatlon=['lat', 'lon'], datalatlon=['Latitude_Mid', 'Longitude_Mid']):
    '''condition on df by geographycal fence
    gf={'lon':(-98, -97.73), 'lat': (28.83, 29.19)}
    '''
    gflat, gflon = gflatlon
    datlat, datlon = datalatlon
    cond = (df[datlat]>gf[gflat][0])&(df[datlat]<gf[gflat][1])&(df[datlon]>gf[gflon][0])&(df[datlon]<gf[gflon][1])
    return cond