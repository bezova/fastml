# utils for models analysis
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from matplotlib import pyplot as plt
import warnings
from sklearn.exceptions import DataConversionWarning
#import xgboost as xgb
pd.options.mode.chained_assignment = None  # default='warn'

def drop_inf(df):
    '''removes np.inf values'''
    return df.replace([np.inf, -np.inf], np.nan).dropna()

def exp_rmspe(y_pred, targ0):
    targ = np.exp(targ0)
    pct_var = (targ - np.exp(y_pred))/targ
    return np.sqrt((pct_var**2).mean())

def mape(y_pred, targ):
    pct_var = (targ - y_pred)/targ
    return np.abs(pct_var).mean()

def exp_pe(pred, targ0):
    targ  = np.exp(targ0)
    pct_var = (np.exp(pred)-targ)/targ
    #pct_var = drop_inf(pct_var)
    return pct_var

def exp_mape(y_pred, targ):
    return np.abs(exp_pe(y_pred, targ)).mean()

#def metric_r2(rf, xt, yt, xgboost=False):
def metric_r2(rf, xt, yt):
    '''returns r2_score(yt, yp)'''
    # if xgboost:
    #     xt = xgb.DMatrix(xt, label=yt, feature_names=xt.columns.tolist())
    yp = rf.predict(xt)
    return r2_score(yt, yp)

#def permutation_importances(rf, X_train, y_train, metric, xgboost=False):
def permutation_importances(rf, X_train, y_train, metric):
    #baseline = metric(rf, X_train, y_train, xgboost=xgboost)
    baseline = metric(rf, X_train, y_train)
    imp = []
    for col in X_train.columns:
        save = X_train[col].copy()
        X_train[col] = np.random.permutation(X_train[col])
        #m = metric(rf, X_train, y_train, xgboost=xgboost)
        m = metric(rf, X_train, y_train)
        X_train[col] = save
        imp.append(baseline - m)
    return np.array(imp)

#def plot_permutation_importances(tree, X_train, y_train, metric, vert_plot=True, xgboost=False):
def plot_permutation_importances(tree, X_train, y_train, metric, feature_importance=None, vert_plot=True, columns=None, ax=None):
    cols = X_train.columns.tolist() if columns is None else columns
    # Plot feature importance
    #feature_importance = permutation_importances(tree, X_train, y_train, metric, xgboost=xgboost)
    if feature_importance is None:
        feature_importance = permutation_importances(tree, X_train, y_train, metric)
    importance_df =pd.DataFrame({'Splits': feature_importance,'Feature':cols})


    if not vert_plot:
        if ax is None: fig, ax = plt.subplots(figsize=(8,15))
        importance_df.sort_values(by='Splits', inplace=True, ascending=True)
        importance_df.plot.barh(x='Feature', legend=None, ax=ax)
    else:
        if ax is None: fig, ax = plt.subplots(figsize=(12,3))
        importance_df.sort_values(by='Splits', inplace=True, ascending=False)
        importance_df.plot.bar(x='Feature', legend=None, ax=ax),
    ax.set_title('Permutation Importance')
    return ax

def plot_tree_importance(cols, tree, vert_plot=True):
    fi = pd.DataFrame({'imp':tree.feature_importances_}, index=cols)
    fi.imp = 100*fi.imp/fi.imp.sum()
    if not vert_plot:
        fi.sort_values(by='imp', inplace=True)
        fi.plot.barh(figsize=(5,12))
        plt.xlabel('Tree: Variable Importance')
    else: 
        fi.sort_values(by='imp', inplace=True, ascending=False)
        fi.plot.bar(figsize=(14,4))
        plt.ylabel('Relative Importance')
        plt.title('Tree: Variable Importance')

def plot_pred_vs_targ(x, y, figsize=(5,5), ax=None, pp=0.3, ax_names=None):
    xy_min = min(x.max(), y.max())
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')
    ax.scatter(x, y, s=8, c='k', alpha=0.5)
    ax.plot([0, xy_min],[0, xy_min], 'r')
    ax.plot([0,xy_min*(1+pp)],[0,xy_min*(1+pp)*(1-pp)], ls='--', c='b')
    ax.plot([0,xy_min*(1-pp)],[0,xy_min*(1-pp)*(1+pp)], ls='--', c='b')
    if ax_names: 
        ax.set_xlabel(ax_names[0]);  ax.set_ylabel(ax_names[1])
    plt.show()
    return ax

def calc_potential(datain:pd.DataFrame, fixing_wells_compl:pd.DataFrame, predict, 
    completion_features, location_transform, latLon=['Longitude_Mid', 'Latitude_Mid']):
    ''' ===========Reservoir potential=================
    for wells in locations from <datain> set <completion_fetures> from <fixing_wells_compl>
    applies <location_transform> if <completions_features> alter <location_features>
    calculated predict(result), returns df[[latLon], ..comp_api...., mean] ##df[[latLon], ..comp_api...., mean]

    EXAMPLE
    location_features= ['Longitude_Mid', 'Latitude_Mid', 'TVD_FT', '12MonthOilRatio', 'Elevation_FT', 'OilGravity_API', 'WellPadDirection', 'RSSubPlay', 'RSInterval' ,'Formation']
    completion_features =['FluidIntensity_BBLPerFT',  'ProppantIntensity_LBSPerFT', 'ProppantLoading_LBSPerGAL', 'Proppant_LBS', 'RSFracJobType',
                     'RSProdWellType',  'RSProppantType', 'TotalFluidPumped_BBL', 'MD_FT', 'PerfInterval_FT', 'FirstProdYear', 'RSOperator']
  
    def predict_exp(df): return np.exp(model.predict(df))
    def location_transform(df): df['MD_FT'] = df['PerfInterval_FT']+1.05*df['TVD_FT']
    '''
    potent_all = datain[latLon]
    for api, fixing_well in fixing_wells_compl.iterrows():
        data = datain.copy()
        #set completion to all locations
        for feat in completion_features: data[feat] = fixing_well[feat]
        location_transform(data)
        potent_all[f'{api}'] = predict(data)
        
    potent_all['mean']=potent_all.iloc[:,2:].T.mean()
    return potent_all[latLon+['mean']]
    #return potent_all

def ice_lines(data, feature_grid, feature_name, data_transformer, predict):
    '''calculate ice linese inteating over fature through feature grid 
        Example of data transformer
        def data_transformer_from_feature(feature_name):    
        if feature_name=='FluidIntensity_BBLPerFT':
            def data_transformer(df):
                df['TotalFluidPumped_BBL'] = df['FluidIntensity_BBLPerFT']*df['PerfInterval_FT']
                # keep same proppant loading
                df['ProppantIntensity_LBSPerFT'] = df['FluidIntensity_BBLPerFT']*df['ProppantLoading_LBSPerGAL']*42
                df['Proppant_LBS'] = df['ProppantIntensity_LBSPerFT']*df['PerfInterval_FT']

        if feature_name=='ProppantIntensity_LBSPerFT':
            def data_transformer(df):
                df['Proppant_LBS'] = df['ProppantIntensity_LBSPerFT']*df['PerfInterval_FT']
                # keep same proppant loading
                df['FluidIntensity_BBLPerFT'] = df['ProppantIntensity_LBSPerFT']/df['ProppantLoading_LBSPerGAL']/42
                df['TotalFluidPumped_BBL'] =  df['FluidIntensity_BBLPerFT']*df['PerfInterval_FT']

        return data_transformer
    '''
    ice_lines = pd.DataFrame(columns=feature_grid, index=data.index)
    # iterate over feature values
    for feature in feature_grid:
        points = data.copy()
        points[feature_name] = feature
        data_transformer(points)
        ice_lines[feature] = predict(points)
    return ice_lines

def ice_fixed_location1(fixing_well_location, fixing_wells_compl, location_features, feature_name, 
                        predict, location_transform, data_transformer, feature_grid=None, gridNum=40):
    ''' calculate ice linece for well fixining location 
    returned df  = [completion] x [feature_grid]
    '''
    data = fixing_wells_compl.copy()
    # assign all completions same locations
    for feat in location_features: data[feat] = fixing_well_location[feat].values[0]
    location_transform(data)

    if feature_grid is None: 
        minF, maxF = min(fixing_wells_compl[feature_name]), max(fixing_wells_compl[feature_name])
        feature_grid = np.linspace(minF, maxF, gridNum)

    return ice_lines(data, feature_grid, feature_name, data_transformer, predict)

def pdp_map_iterCompl1(fixing_wells_location, fixing_wells_compl, completion_features, feature_name, 
                       feature_grid, predict, location_transform, data_transformer, 
                       latlon=['Longitude_Mid', 'Latitude_Mid'], returnOut=False):
    '''  needs completion_features
    # iterate over completions (faster if more locations than completions)
     out = [[locatons] X [grdpoints] X [completions]]
     returns pdp per location and its amplitudes
    '''
    out = np.zeros((len(fixing_wells_location), len(feature_grid), len(fixing_wells_compl)))
    # iterate over completions (faster if more locations than completions)
    for compl_idx, api in enumerate(fixing_wells_compl.index):
        data = fixing_wells_location.copy()
        #set fixed completion to all locations
        for feat in completion_features: data[feat] = fixing_wells_compl.loc[api, feat]
        location_transform(data)
        ice = ice_lines(data, feature_grid, feature_name, data_transformer, predict)
        out[:,:, compl_idx] = ice.values

    out_pdp = pd.DataFrame(out.mean(axis=2), columns=feature_grid, index=fixing_wells_location.index)
    out_mm = fixing_wells_location[latlon]
    out_mm['abs'] = out_pdp.max(axis=1) - out_pdp.min(axis=1)
    out_mm['rel'] =  out_mm['abs'] / out_pdp.min(axis=1)
    if returnOut: return out_pdp, out_mm, out
    else: return out_pdp, out_mm

def pdp_map_iterCompl(fixing_wells_location, fixing_wells_compl, completion_features, feature_name, 
                       predict, func_dict, feature_grid=None, gridNum=20, 
                       latlon=['Longitude_Mid', 'Latitude_Mid'], returnOut=False):
    '''  needs completion_features
    # iterate over completions (faster if more locations than completions)
     out = [[locatons] X [grdpoints] X [completions]]
     returns pdp per location and its amplitudes
    '''
    location_transform, data_transformer = func_dict['location_transform'], func_dict['data_transformer']
    if feature_grid is None: 
        minF, maxF = min(fixing_wells_compl[feature_name]), max(fixing_wells_compl[feature_name])
        feature_grid = np.linspace(minF, maxF, gridNum)

    out = np.zeros((len(fixing_wells_location), len(feature_grid), len(fixing_wells_compl)))
    # iterate over completions (faster if more locations than completions)
    for compl_idx, api in enumerate(fixing_wells_compl.index):
        data = fixing_wells_location.copy()
        #set fixed completion to all locations
        for feat in completion_features: data[feat] = fixing_wells_compl.loc[api, feat]
        location_transform(data)
        ice = ice_lines(data, feature_grid, feature_name, data_transformer, predict)
        out[:,:, compl_idx] = ice.values

    out_pdp = pd.DataFrame(out.mean(axis=2), columns=feature_grid, index=fixing_wells_location.index)
    out_mm = fixing_wells_location[latlon]
    out_mm['abs'] = out_pdp.max(axis=1) - out_pdp.min(axis=1)
    out_mm['rel'] =  out_mm['abs'] / out_pdp.min(axis=1)
    if returnOut: return out_pdp, out_mm, out
    else: return out_pdp, out_mm

def pdp_map_iterLoc(fixing_wells_location, fixing_wells_compl, location_features, feature_name, 
                    feature_grid, predict, location_transform, data_transformer, 
                    latlon=['Longitude_Mid', 'Latitude_Mid']):
    '''  needs location_features
    # # iterate over locations faster f more completions than locations
     out = [[completions] X [grdpoints] X [locatons]]
     returns pdp per location and its amplitudes
    '''
    out = np.zeros((len(fixing_wells_compl), len(feature_grid), len(fixing_wells_location)))
    # iterate over locations faster f more completions than locations
    for well_idx, api in enumerate(fixing_wells_location.index):
        ice = ice_fixed_location(fixing_wells_location.loc[[api]], fixing_wells_compl, location_features, 
                feature_name,  feature_grid, predict, location_transform, data_transformer)
        out[:,:, well_idx] = ice

    out_pdp = pd.DataFrame(out.T.mean(axis=2), columns=feature_grid, index=fixing_wells_location.index)
    out_mm = fixing_wells_location[latlon]
    out_mm['abs'] = out_pdp.max(axis=1) - out_pdp.min(axis=1)
    out_mm['rel'] =  out_mm['abs'] / out_pdp.min(axis=1)
    return out_pdp, out_mm

def ice_fixed_location(fixing_well_location, fixing_wells_compl, location_features, feature_name, 
                        predict, funcs_dict, feature_grid=None, gridNum=40):
    ''' calculate ice linece for well fixining location 
    returned df  = [completion] x [feature_grid]
    EXAMPLE of funcs in funcs_dict
    def get_location_transform(df, alpha=1.05):
        # changes df in place !!
        # alpha = (MD-Perf_interval)/TVD'
        df['MD_FT'] = df['PerfInterval_FT']+alpha*df['TVD_FT']
    
    def data_transformer_from_feature(feature_name):    
        if feature_name=='FluidIntensity_BBLPerFT':
            def data_transformer(df):
                # change in place
                df['TotalFluidPumped_BBL'] = df['FluidIntensity_BBLPerFT']*df['PerfInterval_FT']
                # keep same proppant loading
                df['ProppantIntensity_LBSPerFT'] = df['FluidIntensity_BBLPerFT']*df['ProppantLoading_LBSPerGAL']*42
                df['Proppant_LBS'] = df['ProppantIntensity_LBSPerFT']*df['PerfInterval_FT']

        if feature_name=='ProppantIntensity_LBSPerFT':
            def data_transformer(df):
                # change in place
                df['Proppant_LBS'] = df['ProppantIntensity_LBSPerFT']*df['PerfInterval_FT']
                # keep same proppant loading
                df['FluidIntensity_BBLPerFT'] = df['ProppantIntensity_LBSPerFT']/df['ProppantLoading_LBSPerGAL']/42
                df['TotalFluidPumped_BBL'] =  df['FluidIntensity_BBLPerFT']*df['PerfInterval_FT']
        return data_transformer '''
    location_transform, data_transformer = funcs_dict['location_transform'], funcs_dict['data_transformer']
    data = fixing_wells_compl.copy()
    # assign all completions same locations
    for feat in location_features: data[feat] = fixing_well_location[feat].values[0]
    location_transform(data)

    if feature_grid is None: 
        minF, maxF = min(fixing_wells_compl[feature_name]), max(fixing_wells_compl[feature_name])
        feature_grid = np.linspace(minF, maxF, gridNum)

    return ice_lines(data, feature_grid, feature_name, data_transformer, predict)

def cost_lines(data, feature_grid, feature_name, fd, eco):
    '''calculate cost lines iteating over feture through feature grid
      EXAMPLE
    def get_points_cost(points, eco):
        #set cost for  "point": comepltion with fixed features (and location)
        point_cost = eco['cost_proppant_LB']*points['ProppantIntensity_LBSPerFT']
        point_cost += (eco['cost_water_BBL']+eco['cost_chemicals_BBL']
                       +eco['cost_service_BBL'])*points['FluidIntensity_BBLPerFT']
        point_cost += eco['cost_drilling_FT']*points['MD_FT']/points['PerfInterval_FT']
        point_cost += eco['cost_completion_FT']
        return point_cost
    
    def get_full_cost(cost_per_ft, points): return  cost_per_ft*points['PerfInterval_FT']'''
    data_transformer, points_cost, full_cost =  fd['data_transformer'], fd['points_cost'], fd['full_cost']
    
    cost_ft_lines = pd.DataFrame(columns=feature_grid, index=data.index)
    cost_lines = pd.DataFrame(columns=feature_grid, index=data.index)

    # iterate over feature values
    for feature in feature_grid:
        points = data.copy()
        points[feature_name] = feature
        data_transformer(points)
        cost_per_ft = points_cost(points, eco)
        cost_ft_lines[feature] = cost_per_ft
        cost_lines[feature] = full_cost(cost_per_ft, points)

    return {'$/ft':cost_ft_lines, '$':cost_lines}

def cost_fixed_location(fixing_well_location, fixing_wells_compl, location_features, feature_name, funcs_dict,
                                        feature_grid=None, gridNum=40, eco=None):
    ''' calculate cost lines for well fixining location 
    returned df  = [completion] x [feature_grid]
    '''
    location_transform = funcs_dict['location_transform']

    if eco is None:
        eco ={'cost_drilling_FT': 120., # per total drilled ft
                'cost_completion_FT': 200., # per laterla ft
                'cost_water_BBL': 10.0,
                'cost_chemicals_BBL': 1.0,
                'cost_proppant_LB': 0.1,
                'cost_service_BBL': 4.0,
                'discount_rate': 0.06,
                'price_BOE': 70.0} # in $ per BOE
    #========================================
    data = fixing_wells_compl.copy()
    # assign all completions same locations
    for feat in location_features: data[feat] = fixing_well_location[feat].values[0]
    location_transform(data)
    if feature_grid is None: 
        minF, maxF = min(fixing_wells_compl[feature_name]), max(fixing_wells_compl[feature_name])
        feature_grid = np.linspace(minF, maxF, gridNum)
    return cost_lines(data, feature_grid, feature_name,  funcs_dict, eco), eco

def economics(costD, ice_lines, eco):
    cost_ft_lines, cost_lines = costD['$/ft'], costD['$']
    cost_BOE = 1000*cost_ft_lines/ice_lines # cost/BOE = 1000*  (cost/ft) / (BOE/1000ft)
    return cost_BOE

#     NPV1y = eco['price_BOE']*fullice_lines/(1.0+eco['discount_rate']) -  fullcost_lines
#     NPV1yFT = eco['price_BOE']*ice_lines/(1.0+eco['discount_rate'])/1000 - cost_lines
