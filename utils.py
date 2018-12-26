# utils for models analysis
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from matplotlib import pyplot as plt
import warnings
from sklearn.exceptions import DataConversionWarning
#import xgboost as xgb


def drop_inf(df):
    '''removes np.inf values'''
    return df.replace([np.inf, -np.inf], np.nan).dropna()

def exp_rmspe(y_pred, targ):
    targ = np.exp(targ)
    pct_var = (targ - np.exp(y_pred))/targ
    return np.sqrt((pct_var**2).mean())

def exp_mape(y_pred, targ):
    targ = np.exp(targ)
    pct_var = (targ - np.exp(y_pred))/targ
    return np.abs(pct_var).mean()

def mape(y_pred, targ):
    pct_var = (targ - y_pred)/targ
    return pct_var.abs().mean()

def exp_pe(pred, targ):
    targ  = np.exp(targ)
    pct_var = (np.exp(pred)-targ)/targ
    #pct_var = drop_inf(pct_var)
    return pct_var

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
def plot_permutation_importances(tree, X_train, y_train, metric, vert_plot=True):
    cols = X_train.columns.values
    # Plot feature importance
    #feature_importance = permutation_importances(tree, X_train, y_train, metric, xgboost=xgboost)
    feature_importance = permutation_importances(tree, X_train, y_train, metric)
    importance_df =pd.DataFrame({'Splits': feature_importance,'Feature':cols.tolist()})

    if not vert_plot:
        importance_df.sort_values(by='Splits', inplace=True, ascending=True)
        importance_df.plot.barh(x='Feature', figsize=(8,15))
        plt.show()
    else: 
        importance_df.sort_values(by='Splits', inplace=True, ascending=False)
        importance_df.plot.bar(x='Feature', figsize=(12,3))
    plt.title('Permutation Importance')

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