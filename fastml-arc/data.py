from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler, MinMaxScaler
import warnings
from sklearn.exceptions import DataConversionWarning
from pandas.api.types import is_string_dtype, is_numeric_dtype

import pandas as pd
import numpy as np

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


def encode_cat(df, mapper=None, columns=None, inplace=True):
    '''maps categorical vars to numbers, returns mapper
    to apply to test data: _ = scale_vars(test, scale_mapper)
    # direct transform:   mapper.transform(df)
    # inverse transform: encode_dict = {n[0]: e for n, e in mapper.features}
    encode_dict['RSProppantType'].inverse_transform([0,1,2])
    encode_dict['RSProppantType'].classes_ gives ordered classes list same as in inversetransform'''

    warnings.filterwarnings('ignore', category=DataConversionWarning)
    cols = df.columns if columns is None else columns
    if mapper is None:
        #map_f = [([n], LabelEncoder()) for n in cols if not is_numeric_dtype(df[n])]
        map_f = [(n, LabelEncoder()) for n in cols if not is_numeric_dtype(df[n])]
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

def get_cv_idxs(n, cv_idx=0, val_pct=0.2, seed=42):
    """ Get a list of index values for Validation set from a dataset
    
    Arguments:
        n : int, Total number of elements in the data set.
        cv_idx : int, starting index [idx_start = cv_idx*int(val_pct*n)] 
        val_pct : (int, float), validation set percentage 
        seed : seed value for RandomState
        
    Returns:
        list of indexes 
    """
    np.random.seed(seed)
    n_val = int(val_pct*n)
    idx_start = cv_idx*n_val
    idxs = np.random.permutation(n)
    return idxs[idx_start:idx_start+n_val]

def prepare_trn(df, cat_vars, cont_vars, sample_size=None, 
                scale=True, scalecols=None,
                onehot=False, onehotecols=None, 
                labelencode=True, encodecols=None,
                minmax_labelencoded=True):
    '''
    assigns categorical and numerical columns by cat_vars, cont_vars
    scales if scale all numerical columns given [scalecols]
    onehote encodses if onehot=True all [cat_vars] or [onehotecols]&[numerial]
    LabelEncodes if labelecodecat=True all still numerial cols. or [encodecols]&[numerical]
    if minmax_labelencoded=True apply MinMax scaler to LabelEncoded Columns
    
    '''
    scale_mapper = None
    cat_mapper = None

    if sample_size is not None: df.sample(sample_size)
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
    # encode only cols with more then min_cat categories. other will be dummy encoded
    #if min_cat: encodecols = [n for n, cats in cat_dict.items() if len(cats)>2]    
    if labelencode:  cat_mapper = encode_cat(df, columns=encodecols)
        if minmax_labelencoded:
            minmaxcols = cat_mapper.transformed_names_

    # to apply to test data: _ = encode_cat(test, mapper=cat_mapper)
    ## direct transform:   mapper.transform(df)
    ## inverse transform: encode_dict = {n[0]: e for n, e in mapper.features}
    # encode_dict['RSProppantType'].inverse_transform([0,1,2])
    # encode_dict['RSProppantType'].classes_ gives ordered classes list same as in inversetransform

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