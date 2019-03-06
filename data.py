from sklearn_pandas import DataFrameMapper, gen_features
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler, MinMaxScaler
import warnings
from sklearn.exceptions import DataConversionWarning
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from typing import List
pd.options.mode.chained_assignment = None  # default='warn'
from .data_helper import cut_minmax

# MinMaxScaler takes only 2D arrays. to make it 1D
class Make2D(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reshape(-1,1)
    
# x=np.array([0.77778, 0.11111, 0.66667])
# cc=MinMaxScaler1D().fit(x)
# print(cc.transform(x), cc.inverse_transform(cc.transform(x)))

def rename_rare(df, cols=None, thr=0.01, dropna=True, verbatim=False):
    '''IN PLACE modification to df
    rename rare values in categorical cols to "RARE"'''
    if cols is None: cols = df.columns[df.dtypes == "object"]
    if verbatim: print('renamed for next columns: ', end="", flush=True)
    for col in  df[cols].columns[df[cols].dtypes == "object"]: 
        counts = df[col].value_counts(dropna=dropna)
        d = counts/counts.sum()
        if verbatim and len(d[d<thr])>0: print(f"{col}, ", end="", flush=True)
        df[col] = df[col].apply(lambda x: 'RARE' if d.loc[x] <= thr else x)

def scale_vars(df, mapper=None, columns=None, inplace=True):
    '''from fastai.structured.py
    scales inplace all numeric cols or columns, returns mapper'''
    warnings.filterwarnings('ignore', category=DataConversionWarning)
    cols = df.columns if columns is None else columns

    if mapper is None:
        map_f = [([n], StandardScaler()) for n in cols if is_numeric_dtype(df[n])]
        mapper = DataFrameMapper(map_f, input_df=True, df_out=True).fit(df)
    if inplace: 
        df[mapper.transformed_names_] = mapper.transform(df)
    return mapper

def encode_cat(df, mapper=None, columns=None, minmax_encoded=False, inplace=True):
    '''maps categorical vars to numbers, returns mapper
    to apply to test data: _ = scale_vars(test, scale_mapper)
    # direct transform:   mapper.transform(df)
    # inverse transform: encode_dict = {n[0]: e for n, e in mapper.features}
    encode_dict['RSProppantType'].inverse_transform([0,1,2])
    encode_dict['RSProppantType'].classes_ gives ordered classes list same as in inversetransform
    or if MinMax applyed
    codes=encode_dict['RSSubPlay'][1].inverse_transform([1,0.1,0.2]).round().flatten().astype(int)
    encode_dict['RSSubPlay'][0].inverse_transform(codes)'''

    warnings.filterwarnings('ignore', category=DataConversionWarning)
    cols = df.columns if columns is None else columns
    cols = [n for n in cols if not is_numeric_dtype(df[n])]

    if mapper is None:
        if minmax_encoded:
            map_f = gen_features(cols, [LabelEncoder, Make2D, MinMaxScaler])
        else:
            map_f = gen_features(cols, [LabelEncoder])
        mapper = DataFrameMapper(map_f, input_df=True, df_out=True).fit(df)
    if inplace: 
        df[mapper.transformed_names_] = mapper.transform(df)
    return mapper

def train_cat_var_types(df, cat_vars, cont_vars):
    '''assign 'float32' and 'category' types to columns, 
    returns df, dict{col_name: [cat list]}'''
    for v in cont_vars: df[v] = df[v].astype('float32')   
    for v in cat_vars:  df[v] = df[v].astype('category').cat.as_ordered()        
    cat_dict = {n: df[n].cat.categories for n in cat_vars}
    # df[n].cat.codes gives codes
    return df, cat_dict

def test_apply_cats(df, cat_dict, cont_vars):
    #TODO: rtename to not confuce with pytest
    '''set categorical and continues vars using given dict'''
    cat_vars = list(cat_dict.keys())
    df = df[cat_vars+cont_vars]
    for v in cont_vars: df[v] = df[v].astype('float32')
    # transform cat_vars columns to categorcal
    # appply same ordered categories to df as in traning data (will make same .cat.codes even if some cat in test missing)
    for n in cat_vars: df[n] = pd.Categorical(df[n], categories=cat_dict[n], ordered=True)
    return df

def check_test_unknown_cats(tt, cat_dict):
    '''checks if test has cat not present in train, returns list of unknown cats'''
    new_cats=[]
    for n in cat_dict.keys():
        new_cat=set(tt[n].unique())-set(cat_dict[n])
        if new_cat: new_cats.append((n,list(new_cat)))
    return new_cats

def change_val(val, dic):
    '''will change val by dictionary dic if val in keys or return same val'''
    if val in dic.keys():
        return dic[val]
    else: return val

def subs_new_cat(tt, new_cat_subs):
    '''map categories in columns by new_cat_subs=[(col,{cat:new_cat,..}),..]'''
    for cat, dic in new_cat_subs:
        tt[cat] = tt[cat].map(lambda v: change_val(v, dic))

def split_by_val_idx(idxs, *a):
    """
    copy from fastai
    Split each array passed as *a, to a pair of arrays like this (elements selected by idxs,  the remaining elements)
    This can be used to split multiple arrays containing training data to validation and training set.

    :param idxs [int]: list of indexes selected
    :param a list: list of np.array, each array should have same amount of elements in the first dimension
    :return: list of tuples, each containing a split of corresponding array from *a.
            First element of each tuple is an array composed from elements selected by idxs,
            second element is an array of remaining elements.
    """
    mask = np.zeros(len(a[0]),dtype=bool)
    mask[np.array(idxs)] = True
    return [(o[mask],o[~mask]) for o in a]

def val_train_idxs(n, val_pct=0.2, seed=42):
#def get_cv_idxs(n, cv_idx=0, val_pct=0.2, seed=42):
    """ Get a list of index values for Validation and Traning set from a dataset
    
    Arguments:
        n : int, Total number of elements in the data set.
        cv_idx : int, starting index [idx_start = cv_idx*int(val_pct*n)] 
        val_pct : (int, float), validation set percentage 
        seed : seed value for RandomState
        
    Returns:
        list of indexes val_inx, trn_inx 
    """
    np.random.seed(seed)
    n_val = int(val_pct*n)
    #idx_start = cv_idx*n_val
    idxs = np.random.permutation(n)
    #return idxs[idx_start:idx_start+n_val], idxs[idx_start+n_val,:]
    val = idxs[:n_val]
    trn = idxs[n_val:]
    return val, trn

def prepare_trn(df, cat_vars, cont_vars, sample_size=None, 
                scale=True, scalecols=None,
                onehot=False, onehotecols=None, 
                labelencode=True, encodecols=None,
                minmax_encoded=False):
    '''
    assigns categorical and numerical columns by cat_vars, cont_vars
    scales if scale all numerical columns given [scalecols]
    onehote encodses if onehot=True all [cat_vars] or [onehotecols]&[numerial]
    LabelEncodes if labelecodecat=True all still numerial cols. or [encodecols]&[numerical]
    if minmax_labelencoded=True apply MinMax scaler to LabelEncoded Columns
    
    '''
    scale_mapper = None
    cat_mapper = None

    if sample_size is not None: df = df.sample(sample_size).copy()
    else: df = df.copy()    
    
    #take [cat_vars+cont_vars] and convert cat_vars -> categorical cont_vars->'float32'
    #cat dict # original sorted categories list for cat_vars
    df, cat_dict=train_cat_var_types(df, cat_vars, cont_vars)

    # scale numerical or numerical from [scalecols] 
    if scale: scale_mapper = scale_vars(df, columns=scalecols)
    # to apply to test data: _ = scale_vars(test, mapper=scale_mapper)
    ## direct transform:   mapper.transform(df)

    # OneHot encode (dummies) of all categorical or given cols
    if onehot: 
        onehotecols = cat_vars if onehotecols is None else onehotecols
        df=pd.get_dummies(df, columns=onehotecols)          

    # encode categoricals from [encodecols] colunmns (all categorical if encodecols=None)    
    # if minmax_encoded applay MinMaxScaler to encoded columns
    if labelencode: cat_mapper = encode_cat(df, columns=encodecols, minmax_encoded=minmax_encoded)

    return df, cat_dict, scale_mapper, onehotecols, cat_mapper

def prepare_test(df, cat_dict, cont_vars, scale_mapper, onehotecols, cat_mapper, new_cat_subs=None):
    new_cats = check_test_unknown_cats(df, cat_dict)
    if new_cats: 
        print('there are categories in test that were not in traini set')
        print(new_cats)
        print('consider passing subs new_cat_subs=[(column_name, {cat_in_test:cat_in_train}), ...]')
        return df

    #convert columns to cat and cont as in train
    # sedfing up same categories as in cat_dict taken from train ensures that dummies will generate all necessary columns
    df = test_apply_cats(df, cat_dict, cont_vars)

    #scale as in train
    if scale_mapper is not None: _ = scale_vars(df, mapper=scale_mapper)
    # to apply to test data: _ = scale_vars(test, mapper=scale_mapper)
    ## direct transform:   mapper.transform(df)
    
    #encode cat columns as in train
    if cat_mapper is not None: _ = encode_cat(df, mapper=cat_mapper)

    # one hot encode as in train
    # because categories were set  cat_dict taken from train ensures that dummies will generate all necessary columns
    if onehotecols is not None: df=pd.get_dummies(df, columns=onehotecols)
    return df

def rename_categories_F(target:str, combine:List[str], cat_idx=None):
    '''function which will rename categories
    catIdx is dictionary [cat] -> index'''
    combineLst = combine if cat_idx is None else [cat_idx[cat] for cat in combine]
    targetVal = target if cat_idx is None else cat_idx[target]
    return (lambda x: targetVal if x in combineLst else x)

def cat_to_idx(category, cat_dict):
    '''inverse category encoding dict, returns dict(category->index)'''
    return {name: idx for idx, name in enumerate(cat_dict[category])}

def rename_cat_df(df, category, target_combine, cat_dict=None):
    '''rename categories in df[catgory] from [combine] to target'''
    target, combine = target_combine
    cat_idx = None if cat_dict is None else cat_to_idx(category, cat_dict)
    renameF = rename_categories_F(target, combine, cat_idx)
    df = df.copy()
    df[category] = df[category].apply(renameF)
    return df

def equal_size_cat_idx(df, column, categories, n, concat=True, cat_dict=None, random_state=None, verbose=False)->List:
    '''index list where given categories represented equal n times (or smaller if not enougth) data points'''
    cat_idx = None if cat_dict is None else cat_to_idx(column, cat_dict)
    catList = categories if cat_idx is None else [cat_idx[cat] for cat in categories]
    combined_ind = []
    for cat in catList:
        condition = (df[column]==cat)
        nmax = condition.sum() # number of category values
        if verbose: print(f'{cat_dict[column][cat]}: {nmax}->{min(n, nmax)}')
        combined_ind.append(df[condition].sample(min(n, nmax), random_state=random_state).index)
    if concat: return np.concatenate(combined_ind)
    return dict(zip(categories, combined_ind))

def split_ice_by_categories(ice, fixing_wells_compl, feature_name, category, sub_categories, cat_dict, n_sub=500, use_limits=True):
    ''' use_limits will limit each category ice by the feature name valueas in this category '''
    apisDict = equal_size_cat_idx(fixing_wells_compl, category, sub_categories, n_sub, concat=False, cat_dict=cat_dict, random_state=54, verbose=True)
    icelines = {}
    for cat, apiList in apisDict.items():
        if use_limits:
            feat_vals = fixing_wells_compl.loc[apiList, feature_name]
            ice_columns= cut_minmax(ice.columns, feat_vals.min(),  feat_vals.max())
            icelines[cat] = ice.loc[apiList, ice_columns]
        else: icelines[cat] = ice.loc[apiList]
    return icelines