import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy import ndimage
from typing import List, Tuple, Dict, Optional 
from sklearn.neighbors import NearestNeighbors
from data_helper import low_high_quantile
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
                     vminmax:Optional=None, args={}, argsf={}):
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    if mask is not None: zi = np.ma.masked_where(~mask, zi)
    vmin, vmax = low_high_quantile(pd.Series(zi.flatten()),1/100) if vminmax is None else vminmax
    ax.contourf(xi ,yi, zi, n_conturs, vmin=vmin, vmax=vmax, antialiased=True, **argsf)
    ax.contour(xi, yi, zi, n_conturs, linewidths=0.5, colors='k', antialiased=True, **args); #add vm
    ax.set_aspect(1)
    return fig, ax

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

def colorbar(mappable, location='right', size="5%", pad=0.05, **args):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size=size, pad=pad)
    return fig.colorbar(mappable, cax=cax, **args)