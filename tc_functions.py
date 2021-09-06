#this file contains a list of functions that aid in the analysis of North Atlantic TCs
#--------------------------------------------------------------------------------------------------------------------

#these packages are used in the functions and need to be imported to the program as well

import numpy as np
import xarray as xr
import math
import geopandas as gpd
from shapely.geometry import LineString, Point
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

#--------------------------------------------------------------------------------------------------------------------

#change_lon_range: function to convert lon range from 0-360 to -180-180, this function also removes the horizontal 
#lines near 0 longitude since there is no 360 to 0 interface
#inputs: array of longitudes

def change_lon_range(lons):
    nstorms = np.shape(lons)[0]      #get number of storms and times
    ntimes = np.shape(lons)[1]    
    for i in range(nstorms):
        for j in range(ntimes):
            if lons[i,j] > 180.0:    #subtract 360 if the lon value is above 180 to get (-180,180) range
                lons[i,j] = lons[i,j] - 360.0     
            else:
                continue
    return lons

#--------------------------------------------------------------------------------------------------------------------

#get_distance: function that calculates the distance between two lat/lon points
#Only confirmed to work in the northwestern hemisphere
#inputs: two lat/lon points

def get_distance(point1, point2):
    r = 6371.0                            #define radius of earth and points
    lon1 = math.radians(point1[0])
    lat1 = math.radians(point1[1])
    lon2 = math.radians(point2[0])
    lat2 = math.radians(point2[1])
    dlon = lon2-lon1                      #get difference in lat/lon in radians
    dlat = lat2-lat1                      #implement haversine formula
    a = math.sin(dlat/2.0)*math.sin(dlat/2.0) + math.cos(lat1)*math.cos(lat2)*(math.sin(dlon/2.0))*(math.sin(dlon/2.0)) 
    c = 2.0*math.atan2(math.sqrt(a), math.sqrt(1.0-a))
    d = r*c
    return d

#--------------------------------------------------------------------------------------------------------------------

#remove_unwanted_points_100km: remove points in the midwest, north, and southeast 
#inputs: array of longitudes, array of latitudes, array of maximum winds, array of times

def remove_unwanted_points(lons, lats, max_w, time):
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
                elif -77.0 <= lons[i,j] <= -71.0 and 42.0 <= lats[i,j] <= 48.0:
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

#remove_interpolated_pts: function to remove 3 hourly interpolated points in IBTrACS dataset using the iflags variable
#inputs: array of longitudes, array of latitudes, array of maximum wind speeds, array of times, iflags array

def remove_interpolated_pts(lons, lats, max_w, time, iflag):
    out_lons = np.empty_like(lons)        #create empty arrays
    out_lats = np.empty_like(lats)
    out_maxw = np.empty_like(max_w)
    out_time = np.empty_like(time)
    
    out_lons.fill(np.nan)                 #fill with nans or empty bytes
    out_lats.fill(np.nan)
    out_maxw.fill(np.nan)
    out_time.fill(b'')
    
    nstorms = np.shape(time)[0]           #get number of storms and times
    ntimes = np.shape(time)[1]
    
    for i in range(nstorms):
        n = 0                             #intialize new data point position in time dimension
        for j in range(ntimes):
            if iflag[i,j] == b'O_____________':   #only append points if they are observed
                out_lons[i,n] = lons[i,j]
                out_lats[i,n] = lats[i,j]
                out_maxw[i,n] = max_w[i,j]
                out_time[i,n] = time[i,j]
                n = n+1                           #next appended point is one index greater
    
    print(np.shape(time))                         #shape should be unchanged
    
    return out_lons, out_lats, out_maxw, out_time

#--------------------------------------------------------------------------------------------------------------------

#intensity_prob: function to calculate the probability a data point in a specific intensity distribution is
#above or below a certain value
#inputs: array of maximum wind speeds, the intensity value you want to compare the dataset to, direction
#(either 'greater' or 'less')

def intensity_prob(max_w, intensity_value, direction):
    nstorms = np.shape(max_w)[0]    #get number of storms and times
    ntimes = np.shape(max_w)[1]
    
    if direction == 'greater':      #calculate the prob of a data point being greater than the set value
        total = 0                   #set counters equal to 0
        num_greater = 0
        for i in range(nstorms):
            for j in range(ntimes):
                if max_w[i,j] == np.nan or max_w[i,j] == 0:   #skip point if its a 0 or NaN
                    continue
                else:
                    if max_w[i,j] > intensity_value:   #only update num_greater counter if the int is greater than the value
                        total = total+1
                        num_greater = num_greater+1
                    else:                              #always update the total points counter
                        total = total+1
        probability = num_greater / total              #calculate probability
    elif direction == 'less':      #calculate the prob of a data point being less than the set value
        total = 0                  #set counters equal to 0
        num_less = 0
        for i in range(nstorms):
            for j in range(ntimes):
                if max_w[i,j] == np.nan or max_w[i,j] == 0:  #skip point if its a 0 or NaN
                    continue
                else:
                    if max_w[i,j] < intensity_value:   #only update num_greater counter if the int is less than the value
                        total = total+1
                        num_less = num_less+1
                    else:                              #always update the total points counter 
                        total = total+1
        probability = num_less / total                 #calculate probability
    else:                                              #raise error if the keyword entered is not one of the options
        raise ValueError('Did not enter "greater" or "less"')
    
    return probability

#--------------------------------------------------------------------------------------------------------------------

#ref_na_filter: this function filters out non-North Atlantic TCs from input arrays and includes other
#relevant information about them
#this function only works for the ref simulation files (REF, RCP4.5, RCP8.5)
#inputs: array of longitudes, array of latitudes, array of maximum wind speeds, array of times

def ref_na_filter(lons, lats, max_w, time):
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

#ibtracs_landfall_filter: this function filters out non-landfalling/land impacting TCs in the North Atlantic
#this function only works for the IBTrACS file "IBTrACS.NA.v04r00.nc"
#inputs: array of longitudes, array of latitudes, array of maximum winds, array of times, landfall array from 
#IBTrACS file (gives min distance to land in kilometers), buffer_size: distance in km from land
#that defines a significant impact (0 would be landfall)

#this is only a test function, use landfall_filter in all cases for analysis

def ibtracs_landfall_filter(lons, lats, max_w, time, landfall, buffer_size):
    out_lons = np.zeros_like(lons)      #create arrays with same shape and dtype as original
    out_lats = np.zeros_like(lats)               
    out_maxw = np.zeros_like(max_w)
    out_time = np.zeros_like(time)
    
    nstorms = np.shape(out_lons)[0]     #get total number of storms
    ntimes = np.shape(out_lons)[1]      #get total number of times
    
    land_distance = 0.0 + buffer_size   #set desired distance to land for TC to count as landfall
    
    n = 0
    for i in range(nstorms):
        for j in range(ntimes):
            dist2land = landfall[i,j]            #analyze each point in each TC's track
            if dist2land <= land_distance:
                out_lons[n,:] = lons[i,:]        #copy array values to new array if TC is close to land
                out_lats[n,:] = lats[i,:]
                out_maxw[n,:] = max_w[i,:]
                out_time[n,:] = time[i,:]
                n = n+1
                break
            else:
                continue
    
    m = 0 
    while sum(out_lons[m,:]) != 0. and sum(out_lats[m,:]) != 0.: #determine number of non-zero tracks
        m=m+1

    new_nstorms = m                              #number of storms changed
    out_lons.resize((new_nstorms, ntimes))       #resize arrays to remove 0 entries
    out_lats.resize((new_nstorms, ntimes))
    out_maxw.resize((new_nstorms, ntimes))
    out_time.resize((new_nstorms, ntimes))
    
    print(np.shape(out_lons))                                   #show new number of storms/times
    
    return out_lons, out_lats, out_maxw, out_time

#--------------------------------------------------------------------------------------------------------------------

#landfall_filter: this function filters out non-landfalling/land impacting TCs in the North Atlantic
#this function only works for all files (IBTrACS, REF, RCP4.5, RCP8.5)
#inputs: array of longitudes, array of latitudes, array of max winds, array of times, name of shapefile

def landfall_filter(lons, lats, max_w, time, shapefile):
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

#get_landfall_points: this function returns all points within a specified shapefile region (any points in this 
#region are said to have significantly impacted land)
#inputs: array of longitudes, array of latitudes, array of maxmimum winds, array of times, shapefile name

def get_landfall_points(lons, lats, max_w, time, shapefile):
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
#get_ocean_points: this function returns all points outside a specified shapefile region (in this case
#these points are in the ocean)
#inputs: array of longitudes, array of latitudes, array of maxmimum winds, array of times, shapefile name

def get_ocean_points(lons, lats, max_w, time, shapefile):
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

#get_buffer_points: this function returns all points in the buffer region of two shapefiles (ex. only points in
#a 100km buffer of the eastern U.S. coast)
#inputs: array of longitudes, array of latitudes, array of maxmimum winds, array of times, shapefile name, 
#buffer file name (file that contains the buffer region)

#THIS FUNCTION DOES NOT WORK, IT CAN BE FIXED BUT IT HASN'T BEEN FIXED YET
#the indexing issue occurs when iterating through point_gs, the indices aren't the same as the lat and lon list
#use series of get_ocean_points, get_landfall_points, remove_unwanted_points to do the same thing 

def get_buffer_points(lons, lats, max_w, time, shapefile, bufferfile):
    out_lons = np.zeros_like(lons)      #create arrays with same shape and dtype as original
    out_lats = np.zeros_like(lats)               
    out_maxw = np.zeros_like(max_w)
    out_time = np.zeros_like(time)
    
    out_lons.fill(np.nan)               #fill these values with NaN's so that zeros aren't plotted
    out_lats.fill(np.nan)
    
    nstorms = np.shape(out_lons)[0]     #get total number of storms
    ntimes = np.shape(out_lons)[1]      #get total number of times
    
    landfall_region = gpd.read_file(shapefile)         #create geodataframe from shapefile
    landfall_region = landfall_region.loc[0].geometry  #extract multipolygon from geodataframe
    
    buffer_region = gpd.read_file(bufferfile)          #same process but with buffer file
    buffer_region = buffer_region.loc[0].geometry
    
    for i in range(nstorms):
        point_list = []
        for j in range(ntimes):
            if np.isnan(lons[i,j]) == False and np.isnan(lats[i,j]) == False:  #remove nans from arrays
                if -110.0 <= lons[i,j] <= -97.5 and 25.0 <= lats[i,j] <= 35.0: #don't include southwest, midwest or north
                    continue
                elif -97.5 <= lons[i,j] <= -82.0 and 31.0 <= lats[i,j] <= 45.0:
                    continue
                elif -82.0 <= lons[i,j] <= -77.0 and 35.0 <= lats[i,j] <= 45.0:
                    continue
                elif -77.0 <= lons[i,j] <= -71.0 and 42.0 <= lats[i,j] <= 48.0:
                    continue
                elif -71.0 <= lons[i,j] <= -66.0 and 45.0 <= lats[i,j] <= 48.0:
                    continue
                else:            
                    point_list.append(Point(lons[i,j], lats[i,j]))             #append only numbers
            else:
                continue
        
        point_gs = gpd.GeoSeries(point_list, crs="EPSG:4326")  #create geoseries
        point_gs = point_gs.to_crs("ESRI:54001")               #transform to platecarree  
        
        m = 0
        for k in range(len(point_gs)): #FIND A NEW METHOD TO COPY VALUES
            if point_gs.loc[k].intersects(landfall_region) == False and point_gs.loc[k].intersects(buffer_region) == True:                     
                out_lons[i,m] = lons[i,k]    #copy array values to new array if geometries intersect
                out_lats[i,m] = lats[i,k]
                out_maxw[i,m] = max_w[i,k]
                out_time[i,m] = time[i,k]
                m = m+1
            else:
                continue
    
    print(np.shape(out_lons))               #print the final shape of one array
        
    return out_lons, out_lats, out_maxw, out_time


#--------------------------------------------------------------------------------------------------------------------

#plot_tc_trajectories: this function plots all trajectories of TCs given to it
#inputs: array of longitudes, array of latitudes

def plot_tc_trajectories(lons, lats):
    nstorms = np.shape(lons)[0]                                     #get total number of storms
    for i in range(nstorms):                                      #loop through each TC
        plt.plot(lons[i,:],lats[i,:],transform=ccrs.PlateCarree())  #plot each TC's track

#--------------------------------------------------------------------------------------------------------------------

#plot_tc_trajectories: this function plots all trajectories of TCs given to it using points
#inputs: array of longitudes, array of latitudes

def plot_tc_points(lons, lats):
    nstorms = np.shape(lons)[0]                                           #get total number of storms
    for i in range(nstorms):                                            #loop through each TC
        plt.plot(lons[i,:],lats[i,:], '.', transform=ccrs.PlateCarree())  #plot each TC's track using points

#--------------------------------------------------------------------------------------------------------------------

#ibtracs_plot_trajectories_byintensity: this function plots TC trajectories color coded by TC intensity
#inputs: array of longitudes, array of latitudes, array of maximum wind speeds (in knots)

def ibtracs_plot_trajectories_byintensity(lons, lats, max_w, ax):
    ax=ax                           #set axis to plot on
    nstorms = np.shape(lons)[0]     #set number of storms and times
    ntimes = np.shape(lons)[1]
    for i in range(nstorms):        #loop through each TC
        for j in range(ntimes):     #loop through all points within the TC track, plot individually
            if max_w[i,j] < 34.0:   #color coded by intensity (max wind speed in knots) using Saffir-Simpson scale
                color = 'y'                                           #yellow: tropical depression
                ax.plot(lons[i,j:j+2], lats[i,j:j+2], 
                         color=color, transform=ccrs.PlateCarree())
            elif 34.0 <= max_w[i,j] < 64.0:                  
                color = 'g'                                           #green: tropical storm 
                ax.plot(lons[i,j:j+2], lats[i,j:j+2], 
                         color=color, transform=ccrs.PlateCarree())
            elif 64.0 <= max_w[i,j] < 83.0:
                color = 'c'                                           #cyan: category 1
                ax.plot(lons[i,j:j+2], lats[i,j:j+2], 
                         color=color, transform=ccrs.PlateCarree())
            elif 83.0 <= max_w[i,j] < 96.0:
                color = 'b'                                           #blue: category 2
                ax.plot(lons[i,j:j+2], lats[i,j:j+2], 
                         color=color, transform=ccrs.PlateCarree())
            elif 96.0 <= max_w[i,j] < 113.0:
                color = 'r'                                           #red: category 3
                ax.plot(lons[i,j:j+2], lats[i,j:j+2], 
                         color=color, transform=ccrs.PlateCarree())
            elif max_w[i,j] >= 113.0:
                color = 'k'                                           #black: categories 4 and 5
                ax.plot(lons[i,j:j+2], lats[i,j:j+2], 
                         color=color, transform=ccrs.PlateCarree()) 
            else:
                continue

#--------------------------------------------------------------------------------------------------------------------

#ibtracs_plot_points_byintensity: this function plots TC track points color coded by intensity
#inputs: array of longitudes, array of latitudes, array of maximum wind speeds (in knots)

def ibtracs_plot_points_byintensity(lons, lats, max_w, ax):
    ax=ax                           #set axis to plot on
    nstorms = np.shape(lons)[0]     #set number of storms and times
    ntimes = np.shape(lons)[1]
    for i in range(nstorms):        #loop through each TC
        for j in range(ntimes):     #loop through all points within the TC track, plot individually
            if max_w[i,j] < 34.0:   #color coded by intensity (max wind speed in knots) using Saffir-Simpson scale
                color = 'y'                                           #yellow: tropical depression
                ax.plot(lons[i,j], lats[i,j], '.',
                         color=color, transform=ccrs.PlateCarree())
            elif 34.0 <= max_w[i,j] < 64.0:                  
                color = 'g'                                           #green: tropical storm 
                ax.plot(lons[i,j], lats[i,j], '.', 
                         color=color, transform=ccrs.PlateCarree())
            elif 64.0 <= max_w[i,j] < 83.0:
                color = 'c'                                           #cyan: category 1
                ax.plot(lons[i,j], lats[i,j], '.', 
                         color=color, transform=ccrs.PlateCarree())
            elif 83.0 <= max_w[i,j] < 96.0:
                color = 'b'                                           #blue: category 2
                ax.plot(lons[i,j], lats[i,j], '.', 
                         color=color, transform=ccrs.PlateCarree())
            elif 96.0 <= max_w[i,j] < 113.0:
                color = 'r'                                           #red: category 3
                ax.plot(lons[i,j], lats[i,j], '.', 
                         color=color, transform=ccrs.PlateCarree())
            elif max_w[i,j] >= 113.0:
                color = 'k'                                           #black: categories 4 and 5
                ax.plot(lons[i,j], lats[i,j], '.', 
                         color=color, transform=ccrs.PlateCarree())
            else:
                continue
                
#--------------------------------------------------------------------------------------------------------------------

#ref_plot_trajectories_byintensity: this function plots TC trajectories color coded by TC intensity
#inputs: array of longitudes, array of latitudes, array of maximum wind speeds (in m/s), number of storms

def ref_plot_trajectories_byintensity(lons, lats, max_w, ax):
    ax=ax                           #set axis to plot on
    nstorms = np.shape(lons)[0]     #set number of storms and times
    ntimes = np.shape(lons)[1]
    for i in range(nstorms):        #loop through each TC
        for j in range(ntimes):     #loop through all points within the TC track, plot individually
            if max_w[i,j] < 18.0:   #color coded by intensity (max wind speed in m/s) using Saffir-Simpson scale
                color = 'y'                                           #yellow: tropical depression
                ax.plot(lons[i,j:j+2], lats[i,j:j+2], 
                         color=color, transform=ccrs.PlateCarree())
            elif 18.0 <= max_w[i,j] < 33.0:
                color = 'g'                                           #green: tropical storm 
                ax.plot(lons[i,j:j+2], lats[i,j:j+2], 
                         color=color, transform=ccrs.PlateCarree())
            elif 33.0 <= max_w[i,j] < 43.0:
                color = 'c'                                           #cyan: category 1
                ax.plot(lons[i,j:j+2], lats[i,j:j+2], 
                         color=color, transform=ccrs.PlateCarree())
            elif 43.0 <= max_w[i,j] < 50.0:
                color = 'b'                                           #blue: category 2
                ax.plot(lons[i,j:j+2], lats[i,j:j+2], 
                         color=color, transform=ccrs.PlateCarree())
            elif 50.0 <= max_w[i,j] < 58.0:
                color = 'r'                                           #red: category 3
                ax.plot(lons[i,j:j+2], lats[i,j:j+2], 
                         color=color, transform=ccrs.PlateCarree())
            elif max_w[i,j] >= 58.0:
                color = 'k'                                           #black: categories 4 and 5
                ax.plot(lons[i,j:j+2], lats[i,j:j+2], 
                         color=color, transform=ccrs.PlateCarree()) 
            else:
                continue

#--------------------------------------------------------------------------------------------------------------------

#ibtracs_plot_points_byintensity: this function plots TC track points color coded by intensity
#inputs: array of longitudes, array of latitudes, array of maximum wind speeds (in knots)

def ref_plot_points_byintensity(lons, lats, max_w, ax):
    ax=ax                           #set axis to plot on
    nstorms = np.shape(lons)[0]     #set number of storms and times
    ntimes = np.shape(lons)[1]
    for i in range(nstorms):        #loop through each TC
        for j in range(ntimes):     #loop through all points within the TC track, plot individually
            if max_w[i,j] < 18.0:   #color coded by intensity (max wind speed in knots) using Saffir-Simpson scale
                color = 'y'                                           #yellow: tropical depression
                ax.plot(lons[i,j], lats[i,j], '.',
                         color=color, transform=ccrs.PlateCarree())
            elif 18.0 <= max_w[i,j] < 33.0:                  
                color = 'g'                                           #green: tropical storm 
                ax.plot(lons[i,j], lats[i,j], '.', 
                         color=color, transform=ccrs.PlateCarree())
            elif 33.0 <= max_w[i,j] < 43.0:
                color = 'c'                                           #cyan: category 1
                ax.plot(lons[i,j], lats[i,j], '.', 
                         color=color, transform=ccrs.PlateCarree())
            elif 43.0 <= max_w[i,j] < 50.0:
                color = 'b'                                           #blue: category 2
                ax.plot(lons[i,j], lats[i,j], '.', 
                         color=color, transform=ccrs.PlateCarree())
            elif 50.0 <= max_w[i,j] < 58.0:
                color = 'r'                                           #red: category 3
                ax.plot(lons[i,j], lats[i,j], '.', 
                         color=color, transform=ccrs.PlateCarree())
            elif max_w[i,j] >= 58.0:
                color = 'k'                                           #black: categories 4 and 5
                ax.plot(lons[i,j], lats[i,j], '.',  
                         color=color, transform=ccrs.PlateCarree())
            else:
                continue

#--------------------------------------------------------------------------------------------------------------------

#get_color_list: this function gets a list of colors chosen by intensity 
#inputs: array of maximum wind speeds (wind must be in m/s and all points must be nonzero)
    
def get_color_list(max_w):
    nstorms = np.shape(max_w)[0]    #set number of storms and times
    color_list = []                 #initialize color list 
    for i in range(nstorms):        #loop through every point
            if max_w[i] < 18.0:     #color coded by intensity (max wind speed in knots) using Saffir-Simpson scale
                color_list.append('y')                #yellow: tropical depression
            elif 18.0 <= max_w[i] < 33.0:                  
                color_list.append('g')                #green: tropical storm 
            elif 33.0 <= max_w[i] < 43.0:
                color_list.append('c')                #cyan: category 1
            elif 43.0 <= max_w[i] < 50.0:
                color_list.append('b')                #blue: category 2
            elif 50.0 <= max_w[i] < 58.0:
                color_list.append('r')                #red: category 3
            elif max_w[i] >= 58.0:
                color_list.append('k')                #black: categories 4 and 5
            else:
                continue
    return color_list

#--------------------------------------------------------------------------------------------------------------------               
                
                
                
                