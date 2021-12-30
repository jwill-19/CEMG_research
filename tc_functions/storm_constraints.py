#this file contains a list of functions that aid in the analysis of North Atlantic TCs
#these functions are used when there are constraints related to properties of TCs
#needed during the research
#--------------------------------------------------------------------------------------------------------------------

#these packages are used in the functions and need to be imported to the program as well

import numpy as np
import xarray as xr

#--------------------------------------------------------------------------------------------------------------------

def remove_interpolated_pts(lons, lats, max_w, time, nature, iflag):
    '''
    Removes 3 hourly interpolated points in the IBTrACS
    dataset using the iflags variable. This function
    only works on IBTrACS files.
    
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
    nature: ndarray
        Array of TC categorizations (tropical
        storm, extratropical, etc).
    iflag:
        Array indicating the type of each TC
        point (observed, interpolated, etc).
        
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
    out_nature: ndarray
        New array of TC categorizations (tropical
        storm, extratropical, etc). 
    
    '''
    out_lons = np.empty_like(lons)        #create empty arrays
    out_lats = np.empty_like(lats)
    out_maxw = np.empty_like(max_w)
    out_time = np.empty_like(time)
    out_nature = np.empty_like(nature)
    
    out_lons.fill(np.nan)                 #fill with nans or empty bytes
    out_lats.fill(np.nan)
    out_maxw.fill(np.nan)
    out_time.fill(b'')
    out_nature.fill(b'')
    
    nstorms = np.shape(time)[0]           #get number of storms and times
    ntimes = np.shape(time)[1]
    
    iflag_test = []                       #create test lists to make sure function workes
    test_list = [b'P_____________', b'I_____________', b'V_____________', b'______________']
    
    for i in range(nstorms):
        n = 0                             #intialize new data point position in time dimension
        for j in range(ntimes):
            if iflag[i,j] == b'O_____________':   #only append points if they are observed
                out_lons[i,n] = lons[i,j]
                out_lats[i,n] = lats[i,j]
                out_maxw[i,n] = max_w[i,j]
                out_time[i,n] = time[i,j]
                out_nature[i,n] = nature[i,j]
                iflag_test.append(iflag[i,j])
                n = n+1                           #next appended point is one index greater
                
    for i in range(len(test_list)):            
        if iflag_test.count(test_list[i]) > 0:    #test to see if the values are only TS
            raise ValueError("Elements not copied correctly: Includes " + str(test_list[i]))
    
    print(np.shape(out_time))                     #shape should be unchanged
    
    return out_lons, out_lats, out_maxw, out_time, out_nature

#--------------------------------------------------------------------------------------------------------------------

def remove_coldcore_tcpts(lons, lats, max_w, time, nature):
    '''
    Removes coldcore TC track points in the IBTrACS
    dataset using the nature variable. Only keeps
    TCs designated as tropical storms. This function 
    only works on IBTrACS files.
    
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
    nature: ndarray
        Array of TC categorizations (tropical
        storm, extratropical, etc).
        
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
    out_lons = np.empty_like(lons)           #create empty arrays
    out_lats = np.empty_like(lats)
    out_maxw = np.empty_like(max_w)
    out_time = np.empty_like(time)
    
    out_lons.fill(np.nan)                    #fill with nans or empty bytes
    out_lats.fill(np.nan)
    out_maxw.fill(np.nan)
    out_time.fill(b'')
    
    nstorms = np.shape(time)[0]              #get number of storms and times
    ntimes = np.shape(time)[1]
    
    nature_test = []                         #create test lists to make sure function workes
    test_list = [b'NR', b'DS', b'ET', b'SS', b'MX']

    for i in range(nstorms):
        n = 0                                #intialize new data point position in time dimension
        for j in range(ntimes):
            if nature[i,j] == b'TS':         #only append points if they are tropical in nature
                out_lons[i,n] = lons[i,j]
                out_lats[i,n] = lats[i,j]
                out_maxw[i,n] = max_w[i,j]
                out_time[i,n] = time[i,j]
                nature_test.append(nature[i,j])
                n = n+1                      #next appended point is one index greater
    
    for i in range(len(test_list)):          #test to see if the values are only TS
        if nature_test.count(test_list[i]) > 0:   
            raise ValueError("Elements not copied correctly: Includes " + str(test_list[i]))

    print(np.shape(out_lons))                #shape should be unchanged
    
    return out_lons, out_lats, out_maxw, out_time

#--------------------------------------------------------------------------------------------------------------------

def hrmip_remove_coldcore_tcpts(lons, lats, max_w, time, warm_core):
    '''
    Removes coldcore TC track points in any HighResMIP
    dataset using the warm core variable. Only keeps
    TCs designated as warm core. This function 
    only works on HighResMIP files.
    
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
        Array of TC categorizations (warm 
        or cold core).
        
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
    out_lons = np.empty_like(lons)           #create empty arrays
    out_lats = np.empty_like(lats)
    out_maxw = np.empty_like(max_w)
    out_time = np.empty_like(time)
    
    out_lons.fill(np.nan)                    #fill with nans or empty bytes
    out_lats.fill(np.nan)
    out_maxw.fill(np.nan)
    out_time.fill(b'')
    
    nstorms = np.shape(time)[0]              #get number of storms and times
    ntimes = np.shape(time)[1]
    
    for i in range(nstorms):
        n = 0                                #intialize new data point position in time dimension
        for j in range(ntimes):
            if warm_core[i,j] == 1:          #only append points if they are tropical in nature
                out_lons[i,n] = lons[i,j]
                out_lats[i,n] = lats[i,j]
                out_maxw[i,n] = max_w[i,j]
                out_time[i,n] = time[i,j]
                n = n+1                      #next appended point is one index greater
            
    print(np.shape(out_lons))                #shape should be unchanged
    
    return out_lons, out_lats, out_maxw, out_time

#--------------------------------------------------------------------------------------------------------------------

def remove_td(lons, lats, max_w, time, filename):
    '''
    Removes TCs that don't reach tropical storm intensity
    (>= 17.0 m/s or >= 34.0 kts) at any point in their
    lifetime. 
    
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
    filename: string
        Name of file to be analyzed.
        
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
    
    out_lons.fill(np.nan)               #fill with nans or empty bytes
    out_lats.fill(np.nan)
    out_maxw.fill(np.nan)
    out_time.fill(b'')
    
    nstorms = np.shape(out_lons)[0]     #get total number of storms
    ntimes = np.shape(out_lons)[1]      #get total number of times
    
    maxw_test = []
    
    if filename.count('IBTrACS') == 1:
        out_time.fill(b'')
        n=0
        for i in range(nstorms):
            for j in range(ntimes):
                if max_w[i,j] >= 34.0:              #only append storms if they become TS
                    out_lons[n,:] = lons[i,:]
                    out_lats[n,:] = lats[i,:]
                    out_maxw[n,:] = max_w[i,:]
                    out_time[n,:] = time[i,:]
                    maxw_test.append(max_w[i,j])
                    n = n+1                         #next appended storm is one index greater
                    break
                                                      
    else:
        n=0
        for i in range(nstorms):
            for j in range(ntimes):
                if max_w[i,j] >= 17.0:              #only append storms if they become TS
                    out_lons[n,:] = lons[i,:]
                    out_lats[n,:] = lats[i,:]
                    out_maxw[n,:] = max_w[i,:]
                    out_time[n,:] = time[i,:]
                    maxw_test.append(max_w[i,j])
                    n = n+1                         #next appended storm is one index greater
                    break
            
    print(len(maxw_test))                           #print test length to make sure it matches
    
    k = 0                                           #determine number of non-zero tracks
    while np.isnan(out_lons[k,0]) == False and np.isnan(out_lats[k,0]) == False: 
        k=k+1

    new_nstorms = k                                 #number of storms changed
    out_lons.resize((new_nstorms, ntimes))          #resize arrays to remove 0 entries
    out_lats.resize((new_nstorms, ntimes))
    out_maxw.resize((new_nstorms, ntimes))
    out_time.resize((new_nstorms, ntimes))
                                   
    print(np.shape(out_time))                       #print new shape
    
    return out_lons, out_lats, out_maxw, out_time

#--------------------------------------------------------------------------------------------------------------------
