#this file contains a list of functions that aid in the analysis of North Atlantic TCs
#these functions deal with location constraints needed in the research 
#--------------------------------------------------------------------------------------------------------------------

#these packages are used in these functions and need to be imported to the program as well

import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import LineString, Point

#--------------------------------------------------------------------------------------------------------------------

def remove_unwanted_points(lons, lats, max_w, time):
    '''
    Removes TC track points in the midwest, south, and southeast.
    
    Parameters
    ----------
    lons: ndarray
        Array of longitudes.
    lats: ndarray
        Array of latitudes.
    max_w: ndarray
        Array of maximum surface wind speeds.
    time: ndarray
        Array of times.
        
    Returns
    -------
    out_lons: ndarray
        New array of longitudes.
    out_lats: ndarray
        New array of latitudes.
    out_maxw: ndarray
        New array of maximum surface wind speeds.
    out_time: ndarray
        New array of times.
    
    '''
    out_lons = np.zeros_like(lons)      #create arrays with same shape and dtype as original
    out_lats = np.zeros_like(lats)               
    out_maxw = np.zeros_like(max_w)
    out_time = np.zeros_like(time)
    
    nstorms = np.shape(out_lons)[0]     #get total number of storms
    ntimes = np.shape(out_lons)[1]      #get total number of times
    
    for i in range(nstorms):
        m=0
        for j in range(ntimes):
            if np.isnan(lons[i,j]) == False and np.isnan(lats[i,j]) == False:  #remove nans from arrays
                if -110.0 <= lons[i,j] <= -97.5 and 22.0 <= lats[i,j] <= 45.0: #don't include southwest, midwest or north
                    continue
                elif -97.5 <= lons[i,j] <= -82.0 and 31.0 <= lats[i,j] <= 45.0:
                    continue
                elif -82.0 <= lons[i,j] <= -77.0 and 35.0 <= lats[i,j] <= 48.0:
                    continue
                elif -77.0 <= lons[i,j] <= -71.0 and 42.0 <= lats[i,j] <= 50.0:
                    continue
                elif -71.0 <= lons[i,j] <= -64.0 and 45.0 <= lats[i,j] <= 50.0:
                    continue
                else:   
                    out_lons[i,m] = lons[i,j]    #copy array values to new array if geometries intersect
                    out_lats[i,m] = lats[i,j]
                    out_maxw[i,m] = max_w[i,j]
                    out_time[i,m] = time[i,j]
                    m = m+1
            else:
                continue
    
    return out_lons, out_lats, out_maxw, out_time

#--------------------------------------------------------------------------------------------------------------------

def ref_na_filter(lons, lats, max_w, time):
    '''
    Filters out non-North Atlantic TCs based on the origin point of each storm.
    All information about each NA TC is then copied to new arrays. The new
    shape of the ndarrays is also printed during the function call. This function 
    only works for REF, RCP4.5, and RCP8.5 files.
    
    Parameters
    ----------
    lons: ndarray
        Array of longitudes.
    lats: ndarray
        Array of latitudes.
    max_w: ndarray
        Array of maximum surface wind speeds.
    time: ndarray
        Array of times.
        
    Returns
    -------
    out_lons: ndarray
        New array of longitudes.
    out_lats: ndarray
        New array of latitudes.
    out_maxw: ndarray
        New array of maximum surface wind speeds.
    out_time: ndarray
        New array of times.
    
    '''
    out_lons = np.zeros_like(lons)               #create arrays with same shape and dtype as original
    out_lats = np.zeros_like(lats)               
    out_maxw = np.zeros_like(max_w)
    out_time = np.zeros_like(time)
    
    nstorms = np.shape(out_lons)[0]              #get total number of storms
    ntimes = np.shape(out_lons)[1]               #get total number of times
    
    j = 0
    for i in range(nstorms-1):
        if 299.0 <= lons[i,0] <= 360.0 and 0.0 <= lats[i,0] <= 9.0: #various lat and lon constraints
            out_lons[j,:] = lons[i,:]                               #copy element into new array
            out_lats[j,:] = lats[i,:]                               #process is repeated
            out_maxw[j,:] = max_w[i,:]
            out_time[j,:] = time[i,:]
            j = j+1
            continue
        elif 276.0 <= lons[i,0] <= 360.0 and 9.0 <= lats[i,0] <= 14.0:
            out_lons[j,:] = lons[i,:]
            out_lats[j,:] = lats[i,:]
            out_maxw[j,:] = max_w[i,:]
            out_time[j,:] = time[i,:]
            j = j+1
            continue
        elif 270.0 <= lons[i,0] <= 360.0 and 14.0 <= lats[i,0] <= 17.0:
            out_lons[j,:] = lons[i,:]
            out_lats[j,:] = lats[i,:]
            out_maxw[j,:] = max_w[i,:]
            out_time[j,:] = time[i,:]
            j = j+1
            continue
        elif 260.0 <= lons[i,0] <= 360.0 and 17.0 <= lats[i,0] <= 70.0:
            out_lons[j,:] = lons[i,:]
            out_lats[j,:] = lats[i,:]
            out_maxw[j,:] = max_w[i,:]
            out_time[j,:] = time[i,:]
            j = j+1
            continue
        else:                                                    #if conditions are not met the track
            continue                                             #is not copied
    k = 0 
    while sum(out_lons[k,:]) != 0. and sum(out_lats[k,:]) != 0.: #determine number of non-zero tracks
        k=k+1

    new_nstorms = k                                              #number of storms changed
    out_lons.resize((new_nstorms, ntimes))                       #resize arrays to remove 0 entries
    out_lats.resize((new_nstorms, ntimes))
    out_maxw.resize((new_nstorms, ntimes))
    out_time.resize((new_nstorms, ntimes))
    
    print(np.shape(out_lons))                                    #show new number of storms/times
    
    return out_lons, out_lats, out_maxw, out_time

#--------------------------------------------------------------------------------------------------------------------

def hrmip_na_filter(lons, lats, max_w, time, warm_core):
    '''
    Filters out non-North Atlantic TCs based on the origin point of each storm.
    All information about each NA TC is then copied to new arrays. The new
    shape of the ndarrays is also printed during the function call. This function 
    only works for HighResMIP files.
    
    Parameters
    ----------
    lons: ndarray
        Array of longitudes.
    lats: ndarray
        Array of latitudes.
    max_w: ndarray
        Array of maximum surface wind speeds.
    time: ndarray
        Array of times.
    warm_core: ndarray
        Array classifiying the TC as warm or cold core.
        
    Returns
    -------
    out_lons: ndarray
        New array of longitudes.
    out_lats: ndarray
        New array of latitudes.
    out_maxw: ndarray
        New array of maximum surface wind speeds.
    out_time: ndarray
        New array of times.
    out_warm_core: ndarray
        Array classifiying the TC as warm or cold core.
    
    '''
    out_lons = np.zeros_like(lons)            #create arrays with same shape and dtype as original
    out_lats = np.zeros_like(lats)               
    out_maxw = np.zeros_like(max_w)
    out_time = np.zeros_like(time)
    out_warm_core = np.empty_like(warm_core)  #don't want to confuse 0's for false here
    
    out_time.fill(b'')                        #fill time array with empty bytes
    
    nstorms = np.shape(out_lons)[0]           #get total number of storms
    ntimes = np.shape(out_lons)[1]            #get total number of times
    
    j = 0
    for i in range(nstorms):
        if 299.0 <= lons[i,0] <= 360.0 and 0.0 <= lats[i,0] <= 9.0: #various lat and lon constraints
            out_lons[j,:] = lons[i,:]                               #copy element into new array
            out_lats[j,:] = lats[i,:]                               #process is repeated
            out_maxw[j,:] = max_w[i,:]
            out_time[j,:] = time[i,:]
            out_warm_core[j,:] = warm_core[i,:]
            j += 1
            continue
        elif 276.0 <= lons[i,0] <= 360.0 and 9.0 <= lats[i,0] <= 14.0:
            out_lons[j,:] = lons[i,:]
            out_lats[j,:] = lats[i,:]
            out_maxw[j,:] = max_w[i,:]
            out_time[j,:] = time[i,:]
            out_warm_core[j,:] = warm_core[i,:]
            j += 1
            continue
        elif 270.0 <= lons[i,0] <= 360.0 and 14.0 <= lats[i,0] <= 17.0:
            out_lons[j,:] = lons[i,:]
            out_lats[j,:] = lats[i,:]
            out_maxw[j,:] = max_w[i,:]
            out_time[j,:] = time[i,:]
            out_warm_core[j,:] = warm_core[i,:]
            j += 1
            continue
        elif 260.0 <= lons[i,0] <= 360.0 and 17.0 <= lats[i,0] <= 70.0:
            out_lons[j,:] = lons[i,:]
            out_lats[j,:] = lats[i,:]
            out_maxw[j,:] = max_w[i,:]
            out_time[j,:] = time[i,:]
            out_warm_core[j,:] = warm_core[i,:]
            j += 1
            continue
        else:                                          #if conditions are not met the track
            continue                                   #is not copied
    
    k = 0                                              #determine number of non-zero tracks
    while np.isnan(out_lons[k,0]) == False and np.isnan(out_lats[k,0]) == False: 
        k += 1

    new_nstorms = k                                    #number of storms changed
    out_lons.resize((new_nstorms, ntimes))             #resize arrays to remove 0 entries
    out_lats.resize((new_nstorms, ntimes))
    out_maxw.resize((new_nstorms, ntimes))
    out_time.resize((new_nstorms, ntimes))
    out_warm_core.resize((new_nstorms, ntimes))
    
    print(np.shape(out_lons))                          #show new number of storms/times
    
    return out_lons, out_lats, out_maxw, out_time, out_warm_core

#--------------------------------------------------------------------------------------------------------------------

def landfall_filter(lons, lats, max_w, time, shapefile):
    '''
    Filters out North Atlantic TCs that don't make landfall
    in the eastern U.S. coast. All information from TCs that
    do make landfall is copied into new arrays. The new size 
    of the arrays is also printed during the function call.
    This function works for all files.
    
    Parameters
    ----------
    lons: ndarray
        Array of longitudes.
    lats: ndarray
        Array of latitudes.
    max_w: ndarray
        Array of maximum surface wind speeds.
    time: ndarray
        Array of times.
    shapefile: string
        Name of shapefile that describes the landmass.
        
    Returns
    -------
    out_lons: ndarray
        New array of longitudes.
    out_lats: ndarray
        New array of latitudes.
    out_maxw: ndarray
        New array of maximum surface wind speeds.
    out_time: ndarray
        New array of times.
    
    '''
    out_lons = np.zeros_like(lons)      #create arrays with same shape and dtype as original
    out_lats = np.zeros_like(lats)               
    out_maxw = np.zeros_like(max_w)
    out_time = np.zeros_like(time)
    
    nstorms = np.shape(out_lons)[0]     #get total number of storms
    ntimes = np.shape(out_lons)[1]      #get total number of times
    
    landfall_region_gdf = gpd.read_file(shapefile)         #create geodataframe from shapefile
    landfall_region = landfall_region_gdf.loc[0].geometry  #extract multipolygon from geodataframe            
    
    n = 0
    for i in range(nstorms):
        point_list = []
        for j in range(ntimes):
            if np.isnan(lons[i,j]) == False and np.isnan(lats[i,j]) == False:  #remove nans from arrays
                point_list.append((lons[i,j], lats[i,j]))                      #append only numbers
            else:
                continue
        
        gs = gpd.GeoSeries([LineString(point_list)], crs="EPSG:4326")  #create geoseries containing linestring geometry  
        gs_proj = gs.to_crs("ESRI:54001")          #change crs from mercator to world platecarree
        tc_track = gs_proj.loc[0]                  #extract linestring
        
        if tc_track.intersects(landfall_region) == True:
            out_lons[n,:] = lons[i,:]        #copy array values to new array if geometries intersect
            out_lats[n,:] = lats[i,:]
            out_maxw[n,:] = max_w[i,:]
            out_time[n,:] = time[i,:]
            n = n+1
        else:                                #if conditions are not met the track
            continue                         #is not copied
    m = 0 
    while sum(out_lons[m,:]) != 0. and sum(out_lats[m,:]) != 0.: #determine number of non-zero tracks
        m=m+1

    new_nstorms = m                              #number of storms changed
    out_lons.resize((new_nstorms, ntimes))       #resize arrays to remove 0 entries
    out_lats.resize((new_nstorms, ntimes))
    out_maxw.resize((new_nstorms, ntimes))
    out_time.resize((new_nstorms, ntimes))
    
    print(np.shape(out_lons))                    #show new number of storms/times
        
    return out_lons, out_lats, out_maxw, out_time

#--------------------------------------------------------------------------------------------------------------------

def get_landfall_points(lons, lats, max_w, time, shapefile):
    '''
    Finds all TC track points within a specified shapefile
    region, often a landmass or a buffer region within a 
    certain distance of the coast. The new shape of the 
    arrays is printed during the function call. This
    function works for all files.
    
    Parameters
    ----------
    lons: ndarray
        Array of longitudes.
    lats: ndarray
        Array of latitudes.
    max_w: ndarray
        Array of maximum surface wind speeds.
    time: ndarray
        Array of times.
    shapefile: string
        Name of shapefile that describes the landmass.
        
    Returns
    -------
    out_lons: ndarray
        New array of longitudes.
    out_lats: ndarray
        New array of latitudes.
    out_maxw: ndarray
        New array of maximum surface wind speeds.
    out_time: ndarray
        New array of times.
    
    '''
    out_lons = np.zeros_like(lons)      #create arrays with same shape and dtype as original
    out_lats = np.zeros_like(lats)               
    out_maxw = np.zeros_like(max_w)
    out_time = np.zeros_like(time)
    
    out_lons.fill(np.nan)               #fill these values with NaN's so that zeros aren't plotted
    out_lats.fill(np.nan)
    
    nstorms = np.shape(out_lons)[0]     #get total number of storms
    ntimes = np.shape(out_lons)[1]      #get total number of times
    
    landfall_region_gdf = gpd.read_file(shapefile)         #create geodataframe from shapefile
    landfall_region = landfall_region_gdf.loc[0].geometry  #extract multipolygon from geodataframe
    
    for i in range(nstorms):
        point_list = []
        for j in range(ntimes):
            if np.isnan(lons[i,j]) == False and np.isnan(lats[i,j]) == False:  #remove nans from arrays
                point_list.append(Point(lons[i,j], lats[i,j]))                 #append only numbers
            else:
                continue
        
        point_gs = gpd.GeoSeries(point_list, crs="EPSG:4326")  #create geoseries
        point_gs = point_gs.to_crs("ESRI:54001")               #transform to platecarree  
        
        m = 0
        for j in range(len(point_gs)):
            if point_gs.loc[j].intersects(landfall_region) == True:
                out_lons[i,m] = lons[i,j]    #copy array values to new array if geometries intersect
                out_lats[i,m] = lats[i,j]
                out_maxw[i,m] = max_w[i,j]
                out_time[i,m] = time[i,j]
                m = m+1
            else:
                continue
        
    print(np.shape(out_lons))                #print the final shape of one array
        
    return out_lons, out_lats, out_maxw, out_time

#--------------------------------------------------------------------------------------------------------------------

def get_ocean_points(lons, lats, max_w, time, shapefile):
    '''
    Finds all points outside a specified shapefile region,
    often a landmass or a buffer region within a 
    certain distance of the coast. The new shape of the 
    arrays is printed during the function call. This
    function works for all files.
    
    Parameters
    ----------
    lons: ndarray
        Array of longitudes.
    lats: ndarray
        Array of latitudes.
    max_w: ndarray
        Array of maximum surface wind speeds.
    time: ndarray
        Array of times.
    shapefile: string
        Name of shapefile that describes the landmass.
        
    Returns
    -------
    out_lons: ndarray
        New array of longitudes.
    out_lats: ndarray
        New array of latitudes.
    out_maxw: ndarray
        New array of maximum surface wind speeds.
    out_time: ndarray
        New array of times.
    
    '''
    out_lons = np.zeros_like(lons)      #create arrays with same shape and dtype as original
    out_lats = np.zeros_like(lats)               
    out_maxw = np.zeros_like(max_w)
    out_time = np.zeros_like(time)
    
    out_lons.fill(np.nan)               #fill these values with NaN's so that zeros aren't plotted
    out_lats.fill(np.nan)
    
    nstorms = np.shape(out_lons)[0]     #get total number of storms
    ntimes = np.shape(out_lons)[1]      #get total number of times
    
    landfall_region_gdf = gpd.read_file(shapefile)         #create geodataframe from shapefile
    landfall_region = landfall_region_gdf.loc[0].geometry  #extract multipolygon from geodataframe
    
    for i in range(nstorms):
        point_list = []
        for j in range(ntimes):
            if np.isnan(lons[i,j]) == False and np.isnan(lats[i,j]) == False:  #remove nans from arrays
                point_list.append(Point(lons[i,j], lats[i,j]))                 #append only numbers
            else:
                continue
        
        point_gs = gpd.GeoSeries(point_list, crs="EPSG:4326")  #create geoseries
        point_gs = point_gs.to_crs("ESRI:54001")               #transform to platecarree  
        
        m = 0
        for j in range(len(point_gs)):
            if point_gs.loc[j].intersects(landfall_region) == False:
                out_lons[i,m] = lons[i,j]    #copy array values to new array if geometries intersect
                out_lats[i,m] = lats[i,j]
                out_maxw[i,m] = max_w[i,j]
                out_time[i,m] = time[i,j]
                m = m+1
            else:
                continue
        
    print(np.shape(out_lons))                #print the final shape of one array
        
    return out_lons, out_lats, out_maxw, out_time

#--------------------------------------------------------------------------------------------------------------------
