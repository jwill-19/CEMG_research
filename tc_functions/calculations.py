#this file contains a list of functions that aid in the analysis of North Atlantic TCs
#these functions are for numerical calculations 
#--------------------------------------------------------------------------------------------------------------------

#these packages are used in these functions and need to be imported to the program as well

import numpy as np
import xarray as xr
import math

#--------------------------------------------------------------------------------------------------------------------

def change_lon_range(lons):
    '''
    Converts a longitude range of 0-360 to -180-180.
    
    Parameters
    ----------
    lons: ndarray
        Array of longitude values.
       
    Returns
    -------
    lons: ndarray
        Array of longitude values converted to -180-180 range.
    
    '''
    nstorms = np.shape(lons)[0]      #get number of storms and times
    ntimes = np.shape(lons)[1]    
    for i in range(nstorms):
        for j in range(ntimes):
            if lons[i,j] > 180.0:    #subtract 360 if the lon value is above 180 to get (-180,180) range
                lons[i,j] = lons[i,j] - 360.0     
    return lons

#--------------------------------------------------------------------------------------------------------------------

def get_distance(point1, point2):
    '''
    Calculates the distance between two latitude and longitude
    points on a sphere (Earth) using the Haversine formula.
    Only confirmed to work in the northwestern hemisphere.
    
    Parameters
    ----------
    point1: tuple (lon1, lat1)
        First TC track point.
    point2: tuple (lon2, lat2)
        Second TC track point.
        
    Returns
    -------
    d: float
        Distance between the two points.
    
    '''
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

def intensity_prob(max_w, intensity_value, direction):
    '''
    Calculates the probability a track point in a specific intensity
    distribution is above or below a certain value.
    
    Parameters
    ----------
    max_w: ndarray
        Array of maximum surface wind speeds.
    intensity_value: float
        Value that points in the max_w array are compared to.
    direction: string
        Either 'less' or 'greater'. The direction of comparison.
        
    Returns
    -------
    probability: float
        The probability a track point is above or below the 
        specified value.
    
    '''
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
                        total += 1
                        num_greater += 1
                    else:                              #always update the total points counter
                        total += 1
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
                        total += 1
                        num_less += 1
                    else:                              #always update the total points counter 
                        total += 1
        probability = num_less / total                 #calculate probability
    else:                                              #raise error if the keyword entered is not one of the options
        raise ValueError('Did not enter "greater" or "less"')
    
    return probability

#--------------------------------------------------------------------------------------------------------------------

def get_total_storms(track_file):
    '''
    Get the total number of TCs in a track file.
    Sometimes storms with zero points are included 
    in files and need to be discarded when calculating
    the total number of TCs.
    
    Parameters
    ----------
    track_file: string
        Name of track_file to be analyzed.
        
    Returns
    -------
    num_storms: int
        Total number of TCs.
    
    '''
    DS = xr.open_dataset(track_file) #open file and extract intensity values
    max_w = DS.vmax_2D.values
    DS.close()
    
    nstorms = np.shape(max_w)[0]     #get first dimension of ndarray
    num_storms = 0
    for i in range(nstorms):         #get number of storms excluding 0 entries
        storm = max_w[i,:]
        storm = storm[storm > 0]     #remove 0's and nans
        if len(storm) == 0:
            continue
        else:
            num_storms += 1

    return num_storms

#--------------------------------------------------------------------------------------------------------------------

def get_total_points(track_file):
    '''
    Get the total number of TC track points in a track file.
    
    Parameters
    ----------
    track_file: string
        Name of track_file to be analyzed.
        
    Returns
    -------
    num_pts: int
        Total number of TC track points.
    
    '''
    DS = xr.open_dataset(track_file) #open file and extract intensity values
    max_w = DS.vmax_2D.values
    DS.close()
    
    nstorms = np.shape(max_w)[0]     #get number of storms and times
    ntimes = np.shape(max_w)[1]
    
    total_list = []                             # append all non-nan points
    for i in range(nstorms):
        for j in range(ntimes):
            if np.isnan == True:
                continue
            else:
                total_list.append(max_w[i,j])

    total_array = np.asarray(total_list)         
    total_array = total_array[total_array > 0]  # remove 0 entries  
    num_pts = len(total_array)                  # set total points to length of array
    
    return num_pts

#--------------------------------------------------------------------------------------------------------------------

def get_all_intdist(track_file):
    '''
    Calculates the distribution of all intensity 
    (maximum surface wind speed) values.
    
    Parameters
    ----------
    track_file: string
        Name of the track file to be analyzed.
    
    Returns
    -------
    all_intdist: one-dimensional list
        All intensity values distribution.
        
    '''
    DS = xr.open_dataset(track_file) #open track file and get intensity values
    max_w = DS.vmax_2D.values
    DS.close()

    nstorms = np.shape(max_w)[0]     #get number of storms and times
    ntimes = np.shape(max_w)[1]

    all_list = []
    for i in range(nstorms):
        for j in range(ntimes):
            all_list.append(max_w[i,j])

    all_intdist = np.asarray(all_list)         #calculate the wind speed for each storm's points in buffer region
    all_intdist = all_intdist[all_intdist > 0] #remove 0 entries
    
    if track_file.count('IBTrACS') >= 1:
        all_intdist = all_intdist * 0.51444444444444  #convert to m/s from knots (if IBTrACS)
    
    return all_intdist

#--------------------------------------------------------------------------------------------------------------------

def get_max_intdist(track_file):
    '''
    Calculates the distribution of the max intensity 
    (maximum surface wind speed) values from each
    storm.
    
    Parameters
    ----------
    track_file: string
        Name of the track file to be analyzed.
    
    Returns
    -------
    max_intdist: one-dimensional list
        Max intensity values distribution.
        
    '''
    DS = xr.open_dataset(track_file) #open track file and get intensity values
    max_w = DS.vmax_2D.values
    DS.close()

    nstorms = np.shape(max_w)[0]     #get number of storms and times
    ntimes = np.shape(max_w)[1]
    
    max_list = []                    #append max intensity value of each TC to list
    for i in range(nstorms):
        max_int = max(max_w[i,:])
        max_list.append(max_int)

    max_intdist = np.asarray(max_list)         #calculate the max wind speed for each storm's points in buffer region
    max_intdist = max_intdist[max_intdist > 0] #remove 0 entries
    
    if track_file.count('IBTrACS') >= 1:
        max_intdist = max_intdist * 0.51444444444444 #convert to m/s if file is IBTrACS
    
    return max_intdist

#--------------------------------------------------------------------------------------------------------------------

def get_avg_intdist(track_file):
    '''
    Calculates the distribution of the average  
    intensity (maximum surface wind speed) values
    from each storm.
    
    Parameters
    ----------
    track_file: string
        Name of the track file to be analyzed.
    
    Returns
    -------
    avg_intdist: one-dimensional list
        Average intensity values distribution.
        
    '''
    DS = xr.open_dataset(track_file) #open track file and get intensity values
    max_w = DS.vmax_2D.values
    DS.close()

    nstorms = np.shape(max_w)[0]     #get number of storms and times
    ntimes = np.shape(max_w)[1]
    
    avg_list = []                    #append avg intensity value of each TC to list
    for i in range(nstorms):
        if np.sum(max_w[i,:]) == 0:
            continue
        else:
            storm = max_w[i,:]
            storm = storm[storm > 0]
            avg_int = np.mean(storm)
            avg_list.append(avg_int)
    
    avg_intdist = np.asarray(avg_list)         #calculate the avg wind speed for each storm's points in buffer region
    avg_intdist = avg_intdist[avg_intdist > 0] #remove 0 entries
    
    if track_file.count('IBTrACS') >= 1:
        avg_intdist = avg_intdist * 0.51444444444444 #convert to m/s if file is IBTrACS
    
    return avg_intdist

#--------------------------------------------------------------------------------------------------------------------

def time_diff(point1, point2):
    '''
    Calculates the time difference between two given points.
    
    Parameters
    ----------
    point1: int32 or |S19 (numpy.bytes_)
        First time point.
    point2: int32 or |S19 (numpy.bytes_)
        Second time point.
        
    Returns
    -------
    time_diff: float
        Time difference in hours between the two points.
    
    '''
    if (point1 == b'') or (point2 == b''):     #return nan if there's no time values
        return np.nan
    
    if (point1.dtype == '|S19') and (point2.dtype == '|S19'):  #edit points based on data type 
        try:
            time1 = np.datetime64(point1.decode('UTF-8'))      #convert to np.datetime64     
            time2 = np.datetime64(point2.decode('UTF-8'))
        except ValueError:                     #HadGEM has ValueErrors, don't include them
            return np.nan
    
    elif (point1.dtype == 'int32') and (point2.dtype == 'int32'):
        if (point1 <= 0) or (point2 <= 0):
            return np.nan
        time1_str = str(point1)                #get initial strings in integer format
        time2_str = str(point2)                #convert to np.datetime64 format
        time1_dt = f'{time1_str[0:4]}-{time1_str[4:6]}-{time1_str[6:8]} {time1_str[8:10]}:00:00'
        time2_dt = f'{time2_str[0:4]}-{time2_str[4:6]}-{time2_str[6:8]} {time2_str[8:10]}:00:00'
        time1 = np.datetime64(time1_dt)
        time2 = np.datetime64(time2_dt)
    
    else:                                      #raise error if not either of these data types
        raise TypeError(f'Times not int32 or |S19: {point1} and {point2}')
                                               
                                               #calculate the time difference as a float
    difference = np.timedelta64(time2-time1, 'h') / np.timedelta64(1, 'h')
                                              
    return difference

#--------------------------------------------------------------------------------------------------------------------

def percent_consecutive(track_file):
    '''
    Calculates the percent of points in a track file that
    are of correct 6 hour intervals.
    
    Parameters
    ----------
    track_file: string
        Name of file to be analyzed.
        
    Returns
    -------
    correct: int
        Integer number of 6 hourly instances in the file.
    percent: float
        The percentage of points that are 6 hourly.
    
    '''
    DS = xr.open_dataset(track_file)  #open file and get time array
    try: 
        time = DS.time_str.values
    except AttributeError:
        time = DS.time_byte.values
    DS.close() 

    nstorms = np.shape(time)[0]       #get number of storms and times
    ntimes = np.shape(time)[1]
    
    correct = 0                       #initialize the counters for correct (6hrs) 
    incorrect = 0                     #and incorrect (not 6hrs)
    
    for i in range(nstorms):          #loop through all points
        for j in range(ntimes):       #create time1 and time2 differently depending on file
            if j == (ntimes-1):
                continue

            delta_t = time_diff(time[i,j], time[i,j+1]) #calculate time difference
            
            if np.isnan(delta_t) == True:               #don't include nans in calculation
                continue
            elif delta_t != 6.0:      #add 1 to correct category depending on result
                incorrect += 1
            else:
                correct += 1
    
    total = correct + incorrect       #calculate total 
    percent = 100*(correct/total)     #calculate percent consecutive
    percent = round(percent,2)
    
    return correct, percent

#--------------------------------------------------------------------------------------------------------------------

def get_6hr_tsdist(track_file='', *args):
    '''
    Calculates the translation speed distribution for a TC track file. Only considers
    consecutive TC track points, in this case those that are 6hrs apart.
    
    Parameters
    ----------
    track_file: string, default=''
        Name of file to be analyzed.
    *args: tuple, optional
        Tuple of time, lons, and lats arrays. ORDER (time,lons,lats).
        
    Returns
    -------
    ts_dist: one-dimensional list
        Translation speed distribution of consecutive 6 hourly points.
        
    '''
    ts_dist = []                     #initialize ts dist list
    
    if track_file == '':
        time = args[0]
        lons = args[1]
        lats = args[2]
    else:
        DS = xr.open_dataset(track_file) #open track file and get arrays
        try:
            time = DS.time_str.values
        except AttributeError:
            time = DS.time_byte.values
        lons = DS.clon.values
        lats = DS.clat.values
        DS.close() 

    nstorms = np.shape(time)[0]   #get number of storms and times              
    ntimes = np.shape(time)[1]
    
    for i in range(nstorms):      #loop through all storms
        lon_array = lons[i,:]     #get lons and lats for each storm
        lat_array = lats[i,:]
        
        for j in range(ntimes):   #loop through all points
            if j == (ntimes-1):   #create time1 and time2 differently depending on file
                continue
            
            delta_t = time_diff(time[i,j], time[i,j+1]) #calculate time difference
            
            if delta_t != 6.0:    #skip calculation if the time difference isn't 6 hours
                continue
            else:                 #calculate translation speed in km/h for each point
                track_pt1 = (lon_array[j], lat_array[j])  
                track_pt2 = (lon_array[j+1], lat_array[j+1])
                ts = get_distance(track_pt1, track_pt2) / delta_t 
                ts_dist.append(ts)
    
    return ts_dist

#--------------------------------------------------------------------------------------------------------------------

def avg_int_peryear(track_file):
    '''
    Function to return the distribution of average TC 
    intensity per year.
    
    Parameters
    ----------
    track_file: string
        Name of track file to be analyzed.
    
    Returns
    -------
    years: np.array
        Array of all years in the track file.
    avg_intpy_dist: np.array
        Distribution of average intensities per year.
    
    '''
    DS = xr.open_dataset(track_file) #open track file and get arrays
    try:
        time = DS.time_str.values
    except AttributeError:
        time = DS.time_byte.values
    max_w = DS.vmax_2D.values
    DS.close()
    
    nstorms = np.shape(time)[0]      #get number of storms, times, years              
    ntimes = np.shape(time)[1]
    
    year_list = []                   #get list of the year of each TC
    
    for i in range(nstorms):
        storm = max_w[i,0]
        storm = storm[storm > 0]
        if len(storm) == 0:          #skip if the length of the TC is 0
            year_list.append(0)
            continue
        
        if time[i,0].dtype == '|S19':
            time_str = time[i,0].decode('UTF-8')
        elif time[i,0].dtype == 'int32':
            time_str = str(time[i,0])
        else:
            raise TypeError(f' type of time[i,0] is {time[i,0].dtype}')
    
        year = int(time_str[0:4])     
        year_list.append(year)
        
    if len(year_list) != nstorms:
        raise ValueError(f'year_list length: {len(year_list)} does not equal nstorms ({nstorms})')
        
    nyears = 30                #initialize number of years, years list, first year, and avg dist
    years = []
    first_year = year_list[0]
    avg_intpy_dist = []
    
    if (first_year != 1985) and (first_year != 2021) and (first_year != 2070):  #find first year
        year_arr = np.asarray(year_list)
        year_arr = year_arr[year_arr > 0]
        fy = year_arr[0]
        if fy < 2000:
            first_year = 1985
        elif 2021 < fy < 2051:
            first_year = 2021
        else:
            first_year = 2070
    
    for i in range(nyears):           #create list of all possible years
        years.append(first_year+i)
        
    for i in range(nyears):           #calculate average tc intensity per year    
        try:
            start = year_list.index(years[i])
        except ValueError:
            avg = np.nan
            avg_intpy_dist.append(avg)
            year += 1
            continue
        
        total_pts = []               #initialize list of tc track points per year
        
        for j in range(year_list.count(years[i])):
            storm = max_w[start+j,:]   #set index and calculate avg intensity of the storm
            storm = storm[storm > 0]
            
            for k in range(len(storm)):     
                total_pts.append(storm[k])
            
        avg = np.mean(total_pts)            
        avg_intpy_dist.append(avg)
        
        if np.isnan(avg) == True:
            print(year, total_pts)
            
        year += 1                      #increment year
        
    return np.asarray(years), np.asarray(avg_intpy_dist)


#--------------------------------------------------------------------------------------------------------------------

def tc_count_peryear(track_file):
    '''
    Function to return the distribution of TC
    count per year.
    
    Parameters
    ----------
    track_file: string
        Name of track file to be analyzed.
    
    Returns
    -------
    tc_countpy_dist: np.array
        Distribution of TC counts per year.
    years: np.array
        Array of all years in the track file.
        
    '''
    DS = xr.open_dataset(track_file) #open track file and get arrays
    try:
        time = DS.time_str.values
    except AttributeError:
        time = DS.time_byte.values
    max_w = DS.vmax_2D.values
    DS.close()
    
    nstorms = np.shape(time)[0]      #get number of storms, times, years              
    ntimes = np.shape(time)[1]
    
    year_list = []                   #get list of the year of each TC
    
    for i in range(nstorms):
        storm = max_w[i,0]
        storm = storm[storm > 0]
        if len(storm) == 0:          #skip if the length of the TC is 0
            year_list.append(0)
            continue
        
        if time[i,0].dtype == '|S19':
            time_str = time[i,0].decode('UTF-8')
        elif time[i,0].dtype == 'int32':
            time_str = str(time[i,0])
        else:
            raise TypeError(f' type of time[i,0] is {time[i,0].dtype}')
    
        year = int(time_str[0:4])     
        year_list.append(year)
        
    if len(year_list) != nstorms:
        raise ValueError(f'year_list length: {len(year_list)} does not equal nstorms ({nstorms})')
        
    nyears = 30                #initialize number of years, years list, first year, and avg dist
    years = []
    first_year = year_list[0]
    tc_countpy_dist = []
    
    if (first_year != 1985) and (first_year != 2021) and (first_year != 2070):  #find first year
        year_arr = np.asarray(year_list)
        year_arr = year_arr[year_arr > 0]
        fy = year_arr[0]
        if fy < 2000:
            first_year = 1985
        elif 2021 < fy < 2051:
            first_year = 2021
        else:
            first_year = 2070
    
    for i in range(nyears):           #create list of all possible years
        years.append(first_year+i)
    
    for i in range(nyears):           #create distribution
        try:
            start = year_list.index(years[i])
        except ValueError:
            tc_countpy_dist.append(0)
            continue
        
        tc_countpy_dist.append(year_list.count(years[i]))
        
    return np.asarray(years), np.asarray(tc_countpy_dist)

#--------------------------------------------------------------------------------------------------------------------

def avg_ts_peryear(track_file):
    '''
    Function to return the distribution of average TC 
    translation speed per year.
    
    Parameters
    ----------
    track_file: string
        Name of track file to be analyzed.
    
    Returns
    -------
    avg_tspy_dist: np.array
        Distribution of average translation speed per year.
    years: np.array
        Array of all years in the track file.
        
    '''
    DS = xr.open_dataset(track_file) #open track file and get arrays
    lons = DS.clon.values
    lats = DS.clat.values
    try:
        time = DS.time_str.values
    except AttributeError:
        time = DS.time_byte.values
    max_w = DS.vmax_2D.values
    DS.close()
    
    nstorms = np.shape(time)[0]      #get number of storms, times, years              
    ntimes = np.shape(time)[1]
    
    year_list = []                   #get list of the year of each TC
    
    for i in range(nstorms):
        storm = max_w[i,0]
        storm = storm[storm > 0]
        if len(storm) == 0:          #skip if the length of the TC is 0
            year_list.append(0)
            continue
        
        if time[i,0].dtype == '|S19':
            time_str = time[i,0].decode('UTF-8')
        elif time[i,0].dtype == 'int32':
            time_str = str(time[i,0])
        else:
            raise TypeError(f' type of time[i,0] is {time[i,0].dtype}')
    
        year = int(time_str[0:4])     
        year_list.append(year)
        
    if len(year_list) != nstorms:
        raise ValueError(f'year_list length: {len(year_list)} does not equal nstorms ({nstorms})')
        
    nyears = 30                #initialize number of years, years list, first year, and avg dist
    years = []
    first_year = year_list[0]
    avg_tspy_dist = []
    
    if (first_year != 1985) and (first_year != 2021) and (first_year != 2070):  #find first year
        year_arr = np.asarray(year_list)
        year_arr = year_arr[year_arr > 0]
        fy = year_arr[0]
        if fy < 2000:
            first_year = 1985
        elif 2021 < fy < 2051:
            first_year = 2021
        else:
            first_year = 2070
    
    for i in range(nyears):           #create list of all possible years
        years.append(first_year+i)
        
    for i in range(nyears):           #create avg translation speed per year distribution
        try:
            start = year_list.index(years[i])
            end = start + year_list.count(years[i])
        except ValueError:
            avg = np.nan
            avg_tspy_dist.append(avg)
            year += 1
            continue
        
        time_y = time[start:end,:]
        lons_y = lons[start:end,:]
        lats_y = lats[start:end,:]
        tsdist_y = get_6hr_tsdist('', time_y, lons_y, lats_y)
        avg = np.mean(tsdist_y)
        avg_tspy_dist.append(avg)
        
    return np.asarray(years), np.asarray(avg_tspy_dist)

#--------------------------------------------------------------------------------------------------------------------