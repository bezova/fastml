import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy import ndimage
from typing import List, Tuple, Dict, Optional 
from sklearn.neighbors import NearestNeighbors
from .data_helper import low_high_quantile
from matplotlib import pyplot as plt
from matplotlib import patches, patheffects
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import OrderedDict
import statsmodels.api as sm 

from numpy import ma
from matplotlib import cbook
from matplotlib.colors import Normalize

from matplotlib.colors import LinearSegmentedColormap

#colormap from SHAP packakge
red_blue = LinearSegmentedColormap('red_blue', { # #1E88E5 -> #ff0052
    'red': ((0.0, 30./255, 30./255),
            (1.0, 255./255, 255./255)),

    'green': ((0.0, 136./255, 136./255),
              (1.0, 13./255, 13./255)),

    'blue': ((0.0, 229./255, 229./255),
             (1.0, 87./255, 87./255)),

    'alpha': ((0.0, 1, 1),
              (0.5, 0.3, 0.3),
              (1.0, 1, 1))
})

blue_green = LinearSegmentedColormap('blue_green', { # #1E88E5 -> #ff0052
    'green': ((0.0, 30./255, 30./255),
            (1.0, 255./255, 255./255)),

    'red': ((0.0, 50./255, 50./255),
              (1.0, 10./255, 10./255)),

    'blue': ((0.0, 229./255, 229./255),
             (1.0, 87./255, 87./255)),

    'alpha': ((0.0, 1, 1),
              (0.5, 0.3, 0.3),
              (1.0, 1, 1))
})

blue_green_solid = LinearSegmentedColormap('blue_green_solid', { # #1E88E5 -> #ff0052
    'green': ((0.0, 30./255, 30./255),
            (1.0, 255./255, 255./255)),

    'red': ((0.0, 50./255, 50./255),
              (1.0, 10./255, 10./255)),

    'blue': ((0.0, 229./255, 229./255),
             (1.0, 87./255, 87./255)),

    'alpha': ((0.0, 1, 1),
              (0.5, 1, 1),
              (1.0, 1, 1))
})

# setting midpoint for colorbar
# https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
class MidPointNorm(Normalize):    
    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self,vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin < midpoint < vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")       
        elif vmin == vmax:
            result.fill(0) # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            #First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint            
            resdat[resdat>0] /= abs(vmax - midpoint)            
            resdat[resdat<0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = ma.array(resdat, mask=result.mask, copy=False)                

        if is_scalar:
            result = result[0]            
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if cbook.iterable(value):
            val = ma.asarray(value)
            val = 2 * (val-0.5)  
            val[val>0]  *= abs(vmax - midpoint)
            val[val<0] *= abs(vmin - midpoint)
            val += midpoint
            return val
        else:
            val = 2 * (val - 0.5)
            if val < 0: 
                return  val*abs(vmin-midpoint) + midpoint
            else:
                return  val*abs(vmax-midpoint) + midpoint

def plot_shap_dependence(shapVals_df, df, feature='ProppantIntensity_LBSPerFT', 
                          feature_disp=None, cmap=plt.cm.coolwarm, s=10, title=None, color_bar=True, color_title=None):
    feature_disp = feature if feature_disp is None else feature_disp
    title = feature_disp if title is None else title
    color_title = 'Feature Impact' if color_title is None else color_title
    
    x = df[feature].values
    y = shapVals_df[feature].values
    cvals =y
    clow = np.nanpercentile(cvals, 5)
    chigh = np.nanpercentile(cvals, 95)
    norm = MidPointNorm(midpoint=0) if color_bar else MidPointNorm(midpoint=0, vmin=clow, vmax=chigh) # setting vmin/vmax will clip cbar
#     scalarm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     scalarm._A = []

    cvals_nans = np.isnan(cvals)
    cval_notNan = np.invert(cvals_nans)
    
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(x[cvals_nans], y[cvals_nans], s=s, color="#777777", alpha=1, rasterized=len(x) > 500)
    mapable = ax.scatter(x[cval_notNan], y[cval_notNan], s=s, c=cvals[cval_notNan], cmap=cmap, alpha=1,
                    norm=norm, rasterized=len(x) > 500)
    if color_bar: 
        cb = colorbar(mapable, size=0.15)
        cb.set_clim(clow, chigh) # setting vmin/vmaqx here will set even color beyond these numbers
#         cb = colorbar(scalarm, size=0.15)
        cb.set_label(color_title, size=13)
        cb.outline.set_visible(False)
        cb.set_alpha(1)
    ax.set_xlabel(feature_disp, fontsize=14)
    ax.set_ylabel('Feature Impact', fontsize=14)
    ax.set_title(title, fontsize=14)
    return ax

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
    ax.contour(xi, yi, zi, n_conturs, linewidths=0.5, colors='k', antialiased=True, **args) #add vm
    ax.set_aspect(1)
    cbar =colorbar(cs, label=colorbarLabel) if addColorbar else None
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
    ylabel='annual boe/1000ft', title='Completion Impact', quantile=True, addStd=True,
    addLegend=True, argF={'alpha':0.2}, argPDP={}, figsize=(12,7)):
    '''plot median line with 25, 75% quintiles [default] or mean with +-std'''
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    if smooth: lowess = sm.nonparametric.lowess
    for api, ice in wells_ice.items():
        if zero_start: ice = ice.sub(ice.iloc[:,0], axis=0)
        describe = ice.describe()    # gives mean std and quintile values    
        ice_pdp = describe.loc['50%'] if quantile else describe.loc['mean']
        ice_upper =  describe.loc['75%'] if quantile else describe.loc['mean'] + describe.loc['std']
        ice_lower =  describe.loc['25%'] if quantile else describe.loc['mean'] - describe.loc['std']
        upper = ice_upper.values
        lower = ice_lower.values
        pdp = ice_pdp.values
        if smooth: 
            pdp = lowess(ice_pdp.values, np.array(ice.columns), frac=frac,  return_sorted=False)
            if addStd:
                upper = lowess(ice_upper.values, np.array(ice.columns), frac=frac,  return_sorted=False)
                lower = lowess(ice_lower.values, np.array(ice.columns), frac=frac,  return_sorted=False)
        if addStd: ax.fill_between(ice.columns, upper, lower, **argF)#, color='r')
        ax.plot(list(ice.columns), pdp, label=api, **argPDP)
    if addLegend: ax.legend(loc='upper left')
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, fontsize=14)
    return ax

def plot_ice_by_category(iceLines, completions, category, cat_dict=None, point=None, point_label='',
     xyLabels=('',''), title='Completion Impact', cmapName='tab10', figsize=(10,6), ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    argsP = {'s':80, 'lw':1, 'edgecolors':'k', 'zorder':3}
    cmap=plt.get_cmap(cmapName) # other maps: 'Set1'
    args = {'lw':0.3, 'alpha':0.4, 'zorder':1}

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
            vminmax=None, figsize=(10,6), ax=None, cmapName='gist_stern',
            argsP = {'s':80, 'lw':1, 'edgecolors':'k', 'zorder':3},
            argsL = {'lw':0.2, 'alpha':0.3, 'zorder':1}, smooth=False, frac=0.15):
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    if smooth: lowess = sm.nonparametric.lowess
    cmap=plt.get_cmap(cmapName) #'gist_stern', 'terrain', 'brg' 
    nLines = min(nLines, iceLines.shape[0])
    iceSample = iceLines.sample(nLines, random_state=random_state)
    # normalize colors
    vmin, vmax = low_high_quantile(completions[category],1./100.) if vminmax is None else vminmax
    norm=plt.Normalize(vmin=vmin,vmax=vmax)
    scalarm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    x = iceSample.columns
    for index, row in iceSample.iterrows():
        factor_ind=completions.loc[index, category]
        values = lowess(np.array(row.values), x, frac=frac,  return_sorted=False) if smooth \
                     else row.values
        plt.plot(x, values, c=cmap(norm(factor_ind)), **argsL)

    ax.set(xlabel=xyLabels[0], ylabel=xyLabels[1])
    ax.set_title(title, fontsize=14)
    
    if point is not None: ax.scatter(point[0], point[1], label=point_label,\
         c=cmap(norm(point[2])), **argsP)
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
    scalarm._A = []
    #     cb=plt.colorbar(scalarm); cb.set_label(category)
    cbar =colorbar(scalarm, ax, label=category)
    return ax, scalarm