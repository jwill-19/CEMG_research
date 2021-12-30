#this file contains a list of functions that aid in the analysis of North Atlantic TCs
#these functions aid in the plotting of results
#--------------------------------------------------------------------------------------------------------------------

#these packages are used in these functions and need to be imported to the program as well

import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import LineString, Point
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

#--------------------------------------------------------------------------------------------------------------------

def plot_tc_trajectories(lons, lats):
    '''
    Plots TC trajectories. These TC trajectories
    are randomly colored.
    
    Parameters
    ----------
    lons: ndarray
        Array of longitude values.
    lats: ndarray
        Array of latitude values.
        
    Returns
    -------
    None
    
    '''
    nstorms = np.shape(lons)[0]                                     #get total number of storms
    
    for i in range(nstorms):                                        #loop through each TC
        plt.plot(lons[i,:],lats[i,:],transform=ccrs.PlateCarree())  #plot each TC's track

#--------------------------------------------------------------------------------------------------------------------

def plot_tc_points(lons, lats):
    '''
    Plots all TC track points. These points
    are randomly colored.
    
    Parameters
    ----------
    lons: ndarray
        Array of longitude values.
    lats: ndarray
        Array of latitude values.
        
    Returns
    -------
    None
    
    '''
    nstorms = np.shape(lons)[0]                                           #get total number of storms
    
    for i in range(nstorms):                                              #loop through each TC
        plt.plot(lons[i,:],lats[i,:], '.', transform=ccrs.PlateCarree())  #plot each TC's track using points

#--------------------------------------------------------------------------------------------------------------------

def ibtracs_plot_trajectories_byintensity(lons, lats, max_w, ax):
    '''
    Plots TC trajectories color coded by intensity
    (maximum surface wind speed). In this case, 
    intensity is in knots so this function only
    works for IBTrACS files.
    
    Parameters
    ----------
    lons: ndarray
        Array of longitude values.
    lats: ndarray
        Array of latitude values.
    max_w: ndarray
        Array of maximum surface wind speeds
        (in knots).
    ax: plotting object
        Specifies what plotting object the 
        results will be plotted on.
        
    Returns
    -------
    None
    
    '''
    ax = ax                         #set axis to plot on
    nstorms = np.shape(lons)[0]     #set number of storms and times
    ntimes = np.shape(lons)[1]
    
    for i in range(nstorms):        #loop through each TC
        for j in range(ntimes):     #loop through all points within the TC track, plot individually
            if max_w[i,j] < 33.0:   #color coded by intensity (max wind speed in knots) using Saffir-Simpson scale
                color = 'y'                                           #yellow: tropical depression
                ax.plot(lons[i,j:j+2], lats[i,j:j+2], 
                         color=color, transform=ccrs.PlateCarree())
            elif 33.0 <= max_w[i,j] < 63.0:                  
                color = 'g'                                           #green: tropical storm 
                ax.plot(lons[i,j:j+2], lats[i,j:j+2], 
                         color=color, transform=ccrs.PlateCarree())
            elif 63.0 <= max_w[i,j] < 82.0:
                color = 'c'                                           #cyan: category 1
                ax.plot(lons[i,j:j+2], lats[i,j:j+2], 
                         color=color, transform=ccrs.PlateCarree())
            elif 82.0 <= max_w[i,j] < 95.0:
                color = 'b'                                           #blue: category 2
                ax.plot(lons[i,j:j+2], lats[i,j:j+2], 
                         color=color, transform=ccrs.PlateCarree())
            elif 95.0 <= max_w[i,j] < 112.0:
                color = 'r'                                           #red: category 3
                ax.plot(lons[i,j:j+2], lats[i,j:j+2], 
                         color=color, transform=ccrs.PlateCarree())
            elif max_w[i,j] >= 112.0:
                color = 'k'                                           #black: categories 4 and 5
                ax.plot(lons[i,j:j+2], lats[i,j:j+2], 
                         color=color, transform=ccrs.PlateCarree()) 
            else:
                continue

#--------------------------------------------------------------------------------------------------------------------

def ibtracs_plot_points_byintensity(lons, lats, max_w, ax):
    '''
    Plots TC track points color coded by intensity
    (maximum surface wind speed). In this case, 
    intensity is in knots so this function only
    works for IBTrACS files.
    
    Parameters
    ----------
    lons: ndarray
        Array of longitude values.
    lats: ndarray
        Array of latitude values.
    max_w: ndarray
        Array of maximum surface wind speeds
        (in knots).
    ax: plotting object
        Specifies what plotting object the 
        results will be plotted on.
        
    Returns
    -------
    None
    
    '''
    ax = ax                         #set axis to plot on
    nstorms = np.shape(lons)[0]     #set number of storms and times
    ntimes = np.shape(lons)[1]
    
    for i in range(nstorms):        #loop through each TC
        for j in range(ntimes):     #loop through all points within the TC track, plot individually
            if max_w[i,j] < 33.0:   #color coded by intensity (max wind speed in knots) using Saffir-Simpson scale
                color = 'y'                                           #yellow: tropical depression
                ax.plot(lons[i,j], lats[i,j], '.',
                         color=color, transform=ccrs.PlateCarree())
            elif 33.0 <= max_w[i,j] < 63.0:                  
                color = 'g'                                           #green: tropical storm 
                ax.plot(lons[i,j], lats[i,j], '.', 
                         color=color, transform=ccrs.PlateCarree())
            elif 63.0 <= max_w[i,j] < 82.0:
                color = 'c'                                           #cyan: category 1
                ax.plot(lons[i,j], lats[i,j], '.', 
                         color=color, transform=ccrs.PlateCarree())
            elif 82.0 <= max_w[i,j] < 95.0:
                color = 'b'                                           #blue: category 2
                ax.plot(lons[i,j], lats[i,j], '.', 
                         color=color, transform=ccrs.PlateCarree())
            elif 95.0 <= max_w[i,j] < 112.0:
                color = 'r'                                           #red: category 3
                ax.plot(lons[i,j], lats[i,j], '.', 
                         color=color, transform=ccrs.PlateCarree())
            elif max_w[i,j] >= 112.0:
                color = 'k'                                           #black: categories 4 and 5
                ax.plot(lons[i,j], lats[i,j], '.', 
                         color=color, transform=ccrs.PlateCarree())
            else:
                continue
                
#--------------------------------------------------------------------------------------------------------------------

def ref_plot_trajectories_byintensity(lons, lats, max_w, ax):
    '''
    Plots TC trajectories color coded by intensity
    (maximum surface wind speed). In this case, 
    intensity is in m/s so this function works for
    all files except IBTrACS.
    
    Parameters
    ----------
    lons: ndarray
        Array of longitude values.
    lats: ndarray
        Array of latitude values.
    max_w: ndarray
        Array of maximum surface wind speeds
        (in knots).
    ax: plotting object
        Specifies what plotting object the 
        results will be plotted on.
        
    Returns
    -------
    None
    
    '''
    ax = ax                         #set axis to plot on
    nstorms = np.shape(lons)[0]     #set number of storms and times
    ntimes = np.shape(lons)[1]
    
    for i in range(nstorms):        #loop through each TC
        for j in range(ntimes):     #loop through all points within the TC track, plot individually
            if max_w[i,j] < 17.0:   #color coded by intensity (max wind speed in m/s) using Saffir-Simpson scale
                color = 'y'                                           #yellow: tropical depression
                ax.plot(lons[i,j:j+2], lats[i,j:j+2], 
                         color=color, transform=ccrs.PlateCarree())
            elif 17.0 <= max_w[i,j] < 32.0:
                color = 'g'                                           #green: tropical storm 
                ax.plot(lons[i,j:j+2], lats[i,j:j+2], 
                         color=color, transform=ccrs.PlateCarree())
            elif 32.0 <= max_w[i,j] < 42.0:
                color = 'c'                                           #cyan: category 1
                ax.plot(lons[i,j:j+2], lats[i,j:j+2], 
                         color=color, transform=ccrs.PlateCarree())
            elif 42.0 <= max_w[i,j] < 49.0:
                color = 'b'                                           #blue: category 2
                ax.plot(lons[i,j:j+2], lats[i,j:j+2], 
                         color=color, transform=ccrs.PlateCarree())
            elif 49.0 <= max_w[i,j] < 57.0:
                color = 'r'                                           #red: category 3
                ax.plot(lons[i,j:j+2], lats[i,j:j+2], 
                         color=color, transform=ccrs.PlateCarree())
            elif max_w[i,j] >= 57.0:
                color = 'k'                                           #black: categories 4 and 5
                ax.plot(lons[i,j:j+2], lats[i,j:j+2], 
                         color=color, transform=ccrs.PlateCarree()) 
            else:
                continue

#--------------------------------------------------------------------------------------------------------------------

#ibtracs_plot_points_byintensity: this function plots TC track points color coded by intensity
#inputs: array of longitudes, array of latitudes, array of maximum wind speeds (in knots)

def ref_plot_points_byintensity(lons, lats, max_w, ax):
    '''
    Plots TC track points color coded by intensity
    (maximum surface wind speed). In this case, 
    intensity is in m/s so this function works for
    all files except IBTrACS.
    
    Parameters
    ----------
    lons: ndarray
        Array of longitude values.
    lats: ndarray
        Array of latitude values.
    max_w: ndarray
        Array of maximum surface wind speeds
        (in knots).
    ax: plotting object
        Specifies what plotting object the 
        results will be plotted on.
        
    Returns
    -------
    None
    
    '''
    ax = ax                         #set axis to plot on
    nstorms = np.shape(lons)[0]     #set number of storms and times
    ntimes = np.shape(lons)[1]
    
    for i in range(nstorms):        #loop through each TC
        for j in range(ntimes):     #loop through all points within the TC track, plot individually
            if max_w[i,j] < 17.0:   #color coded by intensity (max wind speed in knots) using Saffir-Simpson scale
                color = 'y'                                           #yellow: tropical depression
                ax.plot(lons[i,j], lats[i,j], '.',
                         color=color, transform=ccrs.PlateCarree())
            elif 17.0 <= max_w[i,j] < 32.0:                  
                color = 'g'                                           #green: tropical storm 
                ax.plot(lons[i,j], lats[i,j], '.', 
                         color=color, transform=ccrs.PlateCarree())
            elif 32.0 <= max_w[i,j] < 42.0:
                color = 'c'                                           #cyan: category 1
                ax.plot(lons[i,j], lats[i,j], '.', 
                         color=color, transform=ccrs.PlateCarree())
            elif 42.0 <= max_w[i,j] < 49.0:
                color = 'b'                                           #blue: category 2
                ax.plot(lons[i,j], lats[i,j], '.', 
                         color=color, transform=ccrs.PlateCarree())
            elif 49.0 <= max_w[i,j] < 57.0:
                color = 'r'                                           #red: category 3
                ax.plot(lons[i,j], lats[i,j], '.', 
                         color=color, transform=ccrs.PlateCarree())
            elif max_w[i,j] >= 57.0:
                color = 'k'                                           #black: categories 4 and 5
                ax.plot(lons[i,j], lats[i,j], '.',  
                         color=color, transform=ccrs.PlateCarree())
            else:
                continue

#--------------------------------------------------------------------------------------------------------------------
    
def get_color_list(max_w):
    '''
    Creates ordered list of colors to apply to
    a plot based on intensity (maximum surface
    wind speed). In this case, intensity is in
    m/s so this function works for all files
    except IBTrACS. 
    
    All points must be nonzero.
    
    Parameters
    ----------
    max_w: ndarray
        Array of maximum surface wind speeds.
        
    Returns
    -------
    color_list: one-dimensional list
        List of color values.
    
    '''
    nstorms = np.shape(max_w)[0]    #set number of storms and times
    color_list = []                 #initialize color list 
    
    for i in range(nstorms):        #loop through every point
            if max_w[i] < 17.0:     #color coded by intensity (max wind speed in knots) using Saffir-Simpson scale
                color_list.append('y')           #yellow: tropical depression
            elif 17.0 <= max_w[i] < 32.0:                  
                color_list.append('g')           #green: tropical storm 
            elif 32.0 <= max_w[i] < 42.0:
                color_list.append('c')           #cyan: category 1
            elif 42.0 <= max_w[i] < 49.0:
                color_list.append('b')           #blue: category 2
            elif 49.0 <= max_w[i] < 57.0:
                color_list.append('r')           #red: category 3
            elif max_w[i] >= 57.0:
                color_list.append('k')           #black: categories 4 and 5
            else:
                continue
    return color_list

#--------------------------------------------------------------------------------------------------------------------               
