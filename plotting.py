import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy import ndimage
from typing import List, Tuple, Dict, Optional 
from sklearn.neighbors import NearestNeighbors
from data_helper import low_high_quantile
from matplotlib import pyplot as plt
from matplotlib import patches, patheffects
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import OrderedDict
import statsmodels.api as sm

def nan_to_mean(arr:np.ndarray, axis:int=0)->np.ndarray:
    '''fills nan with mean over axis .
    uses masked array to apply mean to complete nan columns np.nanmean() can not do that
    other option would be to set some kind of spline extrapolation '''
    data_m = np.ma.masked_invalid(arr, copy=True)
    return np.where(np.isnan(arr), data_m.mean(axis=axis), arr)

Type_mapout = Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]

def map_grid(df:pd.DataFrame, nxny:Tuple[int]=(500,500),
                 lat_lon_names:List[str]=['Latitude_Mid','Longitude_Mid'])->Type_mapout:
    '''generates linear interpolated maps
    return: xi, yi, {col:interpolated}'''
    zis = {}
    cols = df.drop(columns=lat_lon_names).columns
    lat, lon = lat_lon_names
    y, x = df[lat], df[lon]
    nx, ny = nxny
    minx, maxx = x.min(), x.max()
    miny, maxy = y.min(), y.max()
    xi = np.linspace(minx, maxx, nx)
    yi = np.linspace(miny, maxy, ny)
    for col in cols:
        zi = griddata((x, y), df[col], (xi[None,:], yi[:,None]), method='linear')
        zis[col] = zi
    return xi, yi, zis

def blured_map(zis, sigma:float=5.)->Type_mapout:
    '''generates linear interpolated and blured maps
    return: xi, yi, {col:interpolated}, {col:blured}'''
    zibs = {}
    for col, zi in zis.items():
        zi_blurred = nan_to_mean(zi, axis=0) #need so blure not cut nan edges
        zi_blurred = ndimage.gaussian_filter(zi_blurred, sigma=sigma)
        zi_blurred[np.isnan(zi)] = np.nan       
        zibs[col] = zi_blurred
    return zibs

def plot_contour_map(xi:np.ndarray, yi:np.ndarray, zi:np.ndarray, mask:Optional=None, n_conturs:int=15, 
                     ax:Optional=None, fig:Optional=None, figsize=(10,10), 
                     vminmax:Optional=None, addColorbar=True, colorbarLabel=None, args={}, argsf={}):
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    if mask is not None: zi = np.ma.masked_where(~mask, zi)
    vmin, vmax = low_high_quantile(pd.Series(zi.flatten()),1/100) if vminmax is None else vminmax
    cs = ax.contourf(xi ,yi, zi, n_conturs, vmin=vmin, vmax=vmax, antialiased=True, **argsf)
    ax.contour(xi, yi, zi, n_conturs, linewidths=0.5, colors='k', antialiased=True, **args); #add vm
    ax.set_aspect(1)
    return fig, ax
    if addColorbar: 
        cbar =colorbar(cs, label=colorbarLabel)
        return fig, ax, cbar

def mask_by_dist(df, col, xi, yi, radius=0.3, lon_lat_names:List[str]=['Longitude_Mid', 'Latitude_Mid']):
    nx, ny = len(xi), len(yi)
    xm, ym = np.meshgrid(xi, yi)
    Xtrn = df[lon_lat_names]
    Xtest = pd.DataFrame({'x':xm.flatten(), 'y':ym.flatten()})

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(Xtrn, df[col])
    rad, index = nbrs.radius_neighbors(Xtest, radius=radius, return_distance=True)
    mask = np.array([(True if len(x)>0 else False) for x in rad]).reshape((ny,nx))
    return mask

def fence_draw(gf, ax, latlon=['lat', 'lon'], **args):
    ''' takes fennce coord 
    E.G. geo_fence={'lon':(-98, -97.73), 'lat': (28.83, 29.19)}
    adds patch to axes
    '''
    lat, lon = latlon
    dlon = gf[lon][1]-gf[lon][0]
    dlat = gf[lat][1]-gf[lat][0]
    rect = patches.Rectangle((gf[lon][0],gf[lat][0]),dlon,dlat,linewidth=1,edgecolor='r',facecolor='none', **args)
    ax.add_patch(rect)

def colorbar(mappable, ax=None, location='right', size="5%", pad=0.05, **args):
    if ax is None:
        try: ax = mappable.axes
        except: ax = mappable.ax # for contour plots 
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size=size, pad=pad)
    return fig.colorbar(mappable, cax=cax, **args)

def draw_outline(o, lw):
    '''from fastai'''
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])
    
def draw_text(ax, xy, txt, sz=14, outsz=2):
    '''from fastai'''
    #ax.annotate(txt, (df[lon].iloc[i], df[lat].iloc[i]))
    text = ax.text(*xy, txt, verticalalignment='top', color='white',
                   fontsize=sz)#, weight='bold')
    draw_outline(text, outsz)

def draw_rect(ax, b):
    '''from fastai'''
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], 
                         fill=False, edgecolor='white', lw=2))
    draw_outline(patch, 4)

def plot_pdp_std(wells_ice, smooth=True, zero_start=False, frac=0.15, ax=None, xlabel=None, 
    ylabel='annual boe/1000ft', title='Completion Impact'):
    if ax is None: fig, ax = plt.subplots(figsize=(12,7))
    if smooth: lowess = sm.nonparametric.lowess
    for api, ice in wells_ice.items():
        if zero_start: ice = ice.sub(ice.iloc[:,0], axis=0)
        ice_std = ice.std()
        ice_pdp = ice.mean()
        if smooth: 
            ice_pdp = lowess(ice_pdp.values, np.array(ice.columns), frac=frac,  return_sorted=False)
            ice_std = lowess(ice_std.values, np.array(ice.columns), frac=frac,  return_sorted=False)
        upper = ice_pdp + ice_std
        lower = ice_pdp - ice_std
        ax.fill_between(ice.columns, upper, lower, alpha=0.2)#, color='r')
        ax.plot(ice.columns, ice_pdp, label=api)
    #  ax.scatter(opDatGeo.loc[api, feature_name], opDatGeo.loc[api, targ])
    ax.legend(loc='upper left')
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, fontsize=14)
    return ax

def plot_ice_by_category(iceLines, completions, category, cat_dict=None, point=None, point_label='',
     xyLabels=('',''), title='Completion Impact', cmapName='tab10', figsize=(10,6), ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    argsP = {'s':80, 'lw':1, 'edgecolors':'k', 'zorder':3}
    cmap=plt.get_cmap(cmapName) # other maps: 'Set1'
    args = {'lw':0.3, 'alpha':0.4, 'zorder':1}
    
#     lines = ice.loc[apis]#.sample(n=500, random_state=60)
    unique_cats=completions.loc[iceLines.index, category].unique()
    color_num = dict(zip(unique_cats, range(len(unique_cats))))    
        
    x = iceLines.columns
    for index, row in iceLines.iterrows():
        factor_ind=completions.loc[index, category]
        label = factor_ind if cat_dict is None else cat_dict[category][factor_ind]
        plt.plot(x, row.values, c=cmap(color_num[factor_ind]), label=label, **args)

    if point is not None: ax.scatter(point[0], point[1], label=point_label, **argsP)
    ax.set(xlabel=xyLabels[0], ylabel=xyLabels[1])
    ax.set_title(title, fontsize=14)

    #drop repeated legends
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))

    leg = plt.legend(by_label.values(), by_label.keys())
    # transparency
    for legobj in leg.legendHandles:legobj.set_alpha(1) # OR legobj._legmarker.set_alpha(0)  
    
    #linewidth in legend; [-1] to skip line width for point legend 
    handles = leg.legendHandles if point is None else leg.legendHandles[:-1]
    for legobj in handles: legobj.set_linewidth(5.0)
    
    return ax

def plot_ice_by_continues(iceLines, completions, category, nLines=1000, point=None, 
        point_label='', xyLabels=('',''), title='Completion Impact', random_state=42,
            vminmax=None, figsize=(10,6), ax=None, cmapName='gist_stern'):
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    argsP = {'s':80, 'lw':1, 'edgecolors':'k', 'zorder':3}
    cmap=plt.get_cmap(cmapName) #'gist_stern', 'terrain', 'brg'
    args = {'lw':0.2, 'alpha':0.3, 'zorder':1}
    
    iceSample = iceLines.sample(nLines, random_state=random_state)
    # normalize colors
    vmin, vmax = low_high_quantile(completions[category],1./100.) if vminmax is None else vminmax
    norm=plt.Normalize(vmin=vmin,vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    x = iceSample.columns
    for index, row in iceSample.iterrows():
        factor_ind=completions.loc[index, category]
        plt.plot(x, row.values, c=cmap(norm(factor_ind)), **args)

    ax.set(xlabel=xyLabels[0], ylabel=xyLabels[1])
    ax.set_title(title, fontsize=14)
    
    if point is not None: ax.scatter(point[0], point[1], label=point_label, **argsP)
    ax.set(xlabel=xyLabels[0], ylabel=xyLabels[1])
    ax.set_title(title, fontsize=14)

    #drop repeated legends
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))

    leg = plt.legend(by_label.values(), by_label.keys())
    # transparency
    for legobj in leg.legendHandles:legobj.set_alpha(1) # OR legobj._legmarker.set_alpha(0)  
    
    #linewidth in legend; [-1] to skip line width for point legend 
    handles = leg.legendHandles if point is None else leg.legendHandles[:-1]
    for legobj in handles: legobj.set_linewidth(5.0)

    # make up the array of the scalar mappable. Urgh...
    sm._A = []
    #     cb=plt.colorbar(sm); cb.set_label(category)
    cbar =colorbar(sm, ax, label=category)
    return ax, sm