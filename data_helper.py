import pandas as pd
from numpy import nan as NaN
import numpy as np
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from typing import List, Union
import copy
# disable pandas chain assignment warning
pd.options.mode.chained_assignment = None  # default='warn'
# discussion here
# https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas

def low_high_quantile(ser:pd.Series, low:float=None, high:float=None)->List[float]:
    ''' returns values for quantiles
    symmetric if High is omitted'''
    low = low if low else 0.
    high = high if high else 1.
    # if high is None: high = 1.-low
    return ser.quantile(low), ser.quantile(high)

def nan_quantile_ref(df:pd.DataFrame, df_ref:pd.DataFrame, col:str, low:float, high=None):
    '''  In PLACE operation
    modyfies df with quantiles set to nan (low, high) for col
    quantiles calculated from refereced df_ref '''
    bottom, top = low_high_quantile(df_ref[col], low, high)
    df.loc[(df[col]>top)|(df[col]<bottom), col] = NaN

def nan_quantile(df:pd.DataFrame, col:str, low:float=None, high:float=None):
    '''  In PLACE operation
    modyfies df with quantiles set to nan (low, high) for col quantiles '''
    nan_quantile_ref(df, df, col=col, low=low, high=high) 
    # bottom, top = low_high_quantile(df[col], low, high)
    # df.loc[(df[col]>top)|(df[col]<bottom), col] = NaN

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
    if df[col].dtype=='object': knn = KNeighborsClassifier(2, weights='distance')
    else: knn = KNeighborsRegressor(2, weights='distance')
    _=knn.fit(XY[LatLon], XY[col])
    df.loc[cond_to_predict, col] = knn.predict(df.loc[cond_to_predict,LatLon])

def unknown_to_nan(df:pd.DataFrame, list_to_nan=['UNKNOWN'], cols=None):
    ''' in Place Operation
    for categorical collumns (among 'cols' if given) rennames certain values ('UNKNOWN'))  -> Nan'''
    cols = df.columns if cols is None else cols
    for col in df[cols].columns[df[cols].dtypes == 'object']: 
        for name in list_to_nan:
            df.loc[df[col]==name, col] = NaN

def nan_to_uknown(df:pd.DataFrame, unknownName='UNKNOWN', cols=None):
    ''' in Place Operation
    for categorical collumns (among 'cols' if given) rennames NaN to unknownName'''
    cols = df.columns if cols is None else cols
    for col in df[cols].columns[df[cols].dtypes == 'object']: 
                df.loc[df[col].isna(), col] = unknownName

def unknowns_to_sameName(df:pd.DataFrame, unknownName='UNKNOWN', list_to_nan=['UNKNOWN'], cols=None, keepNAN=False):
    ''' IN PLace Opeation
    rename all unknowns (list_to_nan) and NaN categorical values (among 'cols' if given) to same unknownName'''
    unknown_to_nan(df, list_to_nan, cols)
    if not keepNAN: nan_to_uknown(df, unknownName, cols)

def geo_con(df, gf, gflatlon=['lat', 'lon'], datalatlon=['Latitude_Mid', 'Longitude_Mid']):
    '''condition on df by geographycal fence
    gf={'lon':(-98, -97.73), 'lat': (28.83, 29.19)} '''
    gflat, gflon = gflatlon
    datlat, datlon = datalatlon
    cond = (df[datlat]>gf[gflat][0])&(df[datlat]<gf[gflat][1])&(df[datlon]>gf[gflon][0])&(df[datlon]<gf[gflon][1])
    return cond

def select_by_distance(ref, df, R_mile, square=True, latlon=['Latitude_Mid', 'Longitude_Mid']):   
    ''' select wells from df within ceartain square a=2R or radius R around of reference well 
    
    #     At 38 degrees North latitude, 
    #     one degree of latitude equals approximately 364,000 ft (69 miles), 
    #     one minute equals 6068 ft (1.15 miles), one-second equals 101 ft;
    #     one-degree of longitude equals 288,200 ft (54.6 miles),
    #     one minute equals 4800 ft (0.91 mile), and one second equals 80 ft.
    '''
    # def ft_to_rad(ft):
    #     # convert distance in ft to radians
    #     kms_per_radian = 6371.0088 # mean earth radius >  https://en.wikipedia.org/wiki/Earth_radius
    #     ft_in_meters = 0.3048
    #     meter_in_km = 1000.
    #     return ft*ft_in_meters/meter_in_km/kms_per_radian
    # def mile_to_deg(mile):
    #     # convert distance in mile to radian on earth Lat long 
    #     FT_PER_MILE = 5280.
    #     return np.degrees(ft_to_rad(mile*FT_PER_MILE))
    milesInLonDeg = 54.6  # at 38 deg North latitude
    milesInLatDeg = 69.
    lat, lon = latlon
    latR, lonR = ref[latlon].values
    # theta_deg = mile_to_deg(R_mile)
    # if square: 
    #     condition  = ((df[lat]-latR).abs()<=theta_deg) & ((df[lon]-lonR).abs()<=theta_deg)
    # else: 
    #     condition  =((df[lat]-latR)**2 +(df[lon]-lonR)**2) <= (theta_deg**2)
    # return df[condition].copy()
    if square: 
        condition  = ((df[lat]-latR).abs()*milesInLatDeg<=R_mile) & ((df[lon]-lonR).abs()*milesInLonDeg<=R_mile)
    else: 
        condition  =(((df[lat]-latR)*milesInLatDeg)**2 +((df[lon]-lonR)*milesInLonDeg)**2) < (R_mile**2)
    return df[condition]
def raname_dict(dictionary, category, orig, new):
    '''rename category value in dictionary'''
    catDict = copy.deepcopy(dictionary)
    tt = catDict[category].values
    tt[tt==orig]=new
    return catDict

def cut_minmax(arr, minV, maxV): return arr[(arr<=maxV)&(arr>=minV)]
