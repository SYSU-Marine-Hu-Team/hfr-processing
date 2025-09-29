
# Created Sep 2024 by Cathleen Qiao

# This file is for combining multiple radial files into a total file. It 
# selects radial files based on a time and combines the radial vectors to 
# create total vectors. It can then process (quality control tests) and 
# plot the total data.

# This file is derived from these files:
# https://github.com/LorenzoCorgnati/EU_HFR_NODE_pyHFR/blob/main/EU_HFR_NODE_HISTprocessor.py
# https://github.com/LorenzoCorgnati/EU_HFR_NODE_pyHFR/blob/main/totals.py
# https://github.com/rucool/hfradarpy/blob/master/hfradarpy/radials.py

# Look at these repositories for more information:
# https://github.com/LorenzoCorgnati/EU_HFR_NODE_pyHFR/tree/main
# https://github.com/rucool/hfradarpy/tree/master


# import packages
import xarray as xr
import pandas as pd
import numpy as np
import datetime as dt
import glob
import os
import math
from pyproj import Geod

from radials import Radial
from totals import Total

from calc import createGrid, convertAngle

# read in the data of a radial file
radial_dir1 = '../radial-data/processed/MARA/'
radial_dir2 = '../radial-data/processed/WEST/'
radial_dir3 = '../radial-data/processed/JEFF/'

# import file for conducting interpolation on radial data
import radial_interpolation as ri
#import total_interpolation as ti


'''
FUNCTIONS IN THIS FILE:
    selectRadials(paths, time)
    processRadials(rads, interpolate=False)
    findBins(cell, rad, sR, g)
    totalLeastSquare(velDF)
    makeTotalVector(rBins, rDF)
    combineRadials(rDF, grid, search, gRes, tStp)
    applyFlag(T)
    applyQC(T, dx, dy)
    processTotals(T, dx, dy)
    performRadialCombination(paths, time, interpolate=False, gridRes=3000, searchRad=5000)
'''

import pickle

############################### FUNCTIONS ######################################

# selectRadials - create DataFrame of radial files to be combined ##############
def selectRadials(paths, times):
    
    """
    this function lists the input radial files that are within the processing
    time interval
    
    creates the DataFrame containing info needed for the combination of radial
    files into totals
    
    INPUTS: paths = list of possible paths of data
            time = list of datetimes of the processing period (strings)
 
    OUTPUTS: radialsTBP = DataFrame containing all the radials 
                          to be processed for the input  
             bb = bounding box information for creating a grid
    """
    
    # create output total Series
    radialsTBP = pd.DataFrame(columns=['filename', 'filepath', 'timestamp'])
    
    files = []
    
    for t in times:
    
        # wildcard = '*' + t + '.ruv'
        wildcard = '*' + t + '*'
   
        for p in paths:
            # search for radial files
            files.extend((glob.glob(os.path.join(p, wildcard))))
    
    #print(files)
    
    lon_min = []
    lon_max = []
    lat_min = []
    lat_max = []
    
    # list all radial files                
    for f in files:
        
        # get file parts
        filePath = os.path.dirname(f)
        fileName = os.path.basename(f)
        
        # get file timestamp
        radial = Radial(f)
        timeStamp = radial.time.strftime("%Y %m %d %H %M %S")                    
        dateTime = radial.time.strftime("%Y-%m-%d %H:%M:%S")  
        
        lon_min.append(radial.data.LOND.min())
        lon_max.append(radial.data.LOND.max())
        lat_min.append(radial.data.LATD.min())
        lat_max.append(radial.data.LATD.max())

        # prepare data to be inserted into the output DataFrame
        dataRadial = {'filename': [fileName], 'filepath': [filePath], 'timestamp': [timeStamp], 'datetime': [dateTime]}
        
        dfRadial = pd.DataFrame(dataRadial)
        
        # insert into the output DataFrame
        radialsTBP = pd.concat([radialsTBP, dfRadial], ignore_index=True)
        
    bb = [min(lon_min), max(lon_max), min(lat_min), max(lat_max)]
    
    return radialsTBP, bb


#-------------------------------------------------------------------------------

# processRadials - combine radials into totals
def processRadials(rads, interpolate=False):
    
    """
    this function processes the input radial files and then combines them into totals
    
    INPUTS: rads = DataFrame containing the radials to be processed grouped by timestamp
                   for the input network with the related information
            interpolate = if True, then interpolate the radial data

    OUTPUTS:
        
    """

    # add Radial objects to the DataFrame
    rads['Radial'] = (rads.filepath + '/' + rads.filename).apply(lambda x: Radial(x))
    
    if interpolate:
        rads['Radial'] = (rads['Radial']).apply(lambda x: ri.interpolation(x, 2))

'''    
    # Rename indices with site codes
    indexMapper = dict(zip(rads.index.values.tolist(),rads['station_id'].to_list()))
    rads.rename(index=indexMapper,inplace=True)    
    
          
    # convert Radials to standard data format (netCDF)
    
        # European standard data model
        rads = rads.apply(lambda x: applyEHNradialDataModel(x, networkData,             stationData.loc[stationData['station_id'] == x.station_id],vers,logger), axis=1)
        
    if 'dfTot' in locals():
          
        # apply QC to Totals
        dfTot['Total'] = dfTot.apply(lambda x: applyEHNtotalQC(x, networkData, vers, logger), axis=1)        
            
    # convert Totals to standard data format (netCDF)
    
        # European standard data model
        dfTot = dfTot.apply(lambda x: applyEHNtotalDataModel(x, networkData, stationData, vers, logger), axis=1)

    return

'''

#-------------------------------------------------------------------------------

# RadBinsInSearchRadius
def findBins(cell, rad, sR, g):
    
    """
    this function finds out which radial bins are within the spatthresh of the
    origin grid cell (the WGS84 CRS is used for distance calculations)
    
    INPUT: cell = Series containing lon and lat of the origin grid cell
           rad = Radial object
           sR = search radius (in meters)
           g = Geod object according to the Total CRS
        
    OUTPUT: radInSr = list of the radial bins falling within the search radius of the
                      origin grid cell
    """
    
    # convert grid cell Series and radial bins DataFrame to numpy arrays
    #print(cell)
    #print(type(cell))
    cell = cell.to_numpy()
    #print(cell)
    radLon = rad.data['LOND'].to_numpy()
    radLat = rad.data['LATD'].to_numpy() 
    #print(radLon)
    #print(radLat)
    
    # evaluate distances between origin grid cells and radial bins
    az12,az21,cellToRadDist = g.inv(len(radLon) * [cell[0]],len(radLat) * [cell[1]], radLon, radLat)
    #print(cellToRadDist)
    #print(type(cellToRadDist))
    
    # figure out which radial bins are within the spatthresh of the origin grid cell
    radInSR = np.where(cellToRadDist < sR)[0].tolist()
    
    #if (len(radInSR) > 1):
        #print(radInSR)
    
    #print(type(radInSR))
    
    return radInSR


# totalLeastSquare
def totalLeastSquare(velDF):
    
    """
    this function calculates the u/v components of a total vector from 2 to n 
    radial vector components using weighted Least Square method
    
    INPUT: velDF = DataFrame containing contributing radial velocities, bearings
                        and standard deviations (aka contributions)
        
    OUTPUT: u = U component of the total vector
            v = V component of the total vector
            cov = covariance matrix
            covGDOP = covariance matrix assuming uniform unit errors 
                    for all radials (i.e. all radial std=1)
    """
    
    # convert angles from true convention to math convention
    #print(velDF['HEAD'])
    velDF['HEAD'] = convertAngle(velDF['HEAD'].to_numpy())
    #print(velDF['HEAD'])
    
    # form the design matrix (i.e. the angle matrix)
    A = np.stack((np.array([np.cos(np.deg2rad(velDF['HEAD'])) / velDF['STD']]), np.array([np.sin(np.deg2rad(velDF['HEAD'])) / velDF['STD']])), axis = -1)[0,:,:]
    #print(A)
    
    # form the velocity vector
    b = (velDF['VELO'].to_numpy()) / velDF['STD']    
    #print(b)
    
    # evaluate the covariance matrix cov (variance(U) = cov(1,1) and variance(V) = cov(2,2))
    A2 = np.matmul(A.T, A)
    if np.linalg.det(A2) > 0:
        cov = np.linalg.inv(A2)
    
        # calculate the u and v for the total vector
        a = np.matmul(cov, np.matmul(A.T, b))
        u = a[0]
        v = a[1]    
        
        # form the design matrix for GDOP evaluation (i.e. setting all radial std to 1)
        des = np.stack((np.array([np.cos(np.deg2rad(velDF['HEAD']))]), np.array([np.sin(np.deg2rad(velDF['HEAD']))])), axis = -1)[0,:,:]
        
        # evaluate the covariance matrix covGDOP for GDOP evaluation (i.e. setting all radial std to 1)
        des = np.matmul(des.T, des)
        if np.linalg.det(des):
            covGDOP = np.linalg.inv(des)
            #print(cov)
            #print(covGDOP)
            return u, v, cov, covGDOP
        
    u = np.nan
    v = np.nan
    cov = np.nan
    covGDOP = np.nan
    return u, v, cov, covGDOP
        

# makeTotalVector
def makeTotalVector(rBins, rDF):
    
    """
    this function gets the total vector for each grid cell
    (use weighted Least Square method for combination)
    
    INPUT: rBins = Series containing contributing radial indices for each site
           rDF = DataFrame containing input Radials
        
    OUTPUT: totalData = Series containing u/v components and related errors of 
                        total vector for each grid cell
    """
    
    #print(rBins)
    #print(rDF)
    
    # set minimum number of contributing radial sites
    minContrSites = 2
    
    # set minimum number of contributing radial vectors
    minContrRads = 3
    
    # create output total Series
    totalData = pd.Series(np.nan,index=range(9))
    
    # only consider contributing radial sites
    contrRad = rBins[rBins.str.len() != 0]
    #print(contrRad)
    #print(type(contrRad))
    
    # check if there are at least two contributing radial sites
    if contrRad.size >= minContrSites:
    
        # initialize empty dataframe for each grid cell
        contributions = pd.DataFrame()
        
        # loop over contributing radial indices for collecting velocities and angles
        
        # for each radial site
        for idx in contrRad.index:
            #print(contrRad[idx])
            contrVel = rDF.loc[idx]['Radial'].data.VELO[contrRad[idx]]
            contrHead = rDF.loc[idx]['Radial'].data.HEAD[contrRad[idx]]
            contrStd = rDF.loc[idx]['Radial'].data.ETMP[contrRad[idx]]
            contrStd = contrStd.rename("STD")
            contributions = pd.concat([contributions, pd.concat([contrVel, contrHead, contrStd], axis=1)])
        #print(contributions)
                    
        # only keep contributing radials with valid standard deviation values
        contributions = contributions[contributions.STD.notnull()]
        contributions = contributions[contributions.STD != 0]
        
        # check if there are at least three contributing radial vectors
        if len(contributions.index) >= minContrRads:
            
            # combine radial contributions to get total vector for the current grid cell
            u, v, cov, covGDOP = totalLeastSquare(contributions)
            #print(u)
            #print(v)
            #print(cov)
            #print(covGDOP)
            
            if not math.isnan(u):
                # populate Total Series
                totalData.loc[0] = u                                            # VELU
                totalData.loc[1] = v                                            # VELV
                totalData.loc[2] = np.sqrt(u**2 + v**2)                         # VELO
                totalData.loc[3] = (360 + np.arctan2(u,v) * 180/np.pi) % 360    # HEAD
                totalData.loc[4] = math.sqrt(cov[0,0])                          # UQAL
                totalData.loc[5] = math.sqrt(cov[1,1])                          # VQAL
                totalData.loc[6] = cov[0,1]                                     # CQAL
                totalData.loc[7] = math.sqrt(np.abs(covGDOP.trace()))           # GDOP
                totalData.loc[8] = len(contributions.index)                     # NRAD
            
    return totalData


# combineRadials
def combineRadials(rDF, grid, search, gRes, tStp):
    
    """
    this function generataes total vectors from radial measurements using the
    weighted Least Square method for combination (need at least 2 sites)
    
    INPUT: rDF = DataFrame containing input Radials; indices must be the site codes.
           grid = grid of all the lon / lat points
           search = search radius for combination in meters
           gRes = grid resoultion (in meters)
           tStp = timestamp in datetime format (YYYY-MM-DD hh:mm:ss)
        
    OUTPUT: T = Total object generated by the combination
    """

    # create empty total with grid
    T = Total(grid=grid)
    #print("\nTotal Data:\n")
    #print(T.data)
    #print("\nTotal Metadata:\n")
    #print(T.metadata)


    # fill site_source DataFrame with contributing radials information
    siteNum = 0    # initialization of site number
    for index, row in rDF.iterrows():
        #print(index)
        #print(row)
        #print(type(row))

        siteNum = siteNum + 1
        rad = row['Radial']
        thisRadial = pd.DataFrame(index=[index], columns=['#', 'Name', 'Lat', 'Lon', 'Coverage(s)', 'RngStep(km)', 'Pattern', 'AntBearing(NCW)'])
        thisRadial['#'] = siteNum
        thisRadial['Name'] = (rad.metadata['Site'].split()[0])
        thisRadial['Lat'] = float(rad.metadata['Origin'].split()[0])
        thisRadial['Lon'] = float(rad.metadata['Origin'].split()[1])
        thisRadial['Coverage(s)'] = float(rad.metadata['TimeCoverage'].split()[0])
        thisRadial['RngStep(km)'] = float(rad.metadata['RangeResolutionKMeters'].split()[0])
        thisRadial['Pattern'] = rad.metadata['PatternType'].split()[0]
        thisRadial['AntBearing(NCW)'] = float(rad.metadata['AntennaBearing'].split()[0])
        T.site_source = pd.concat([T.site_source, thisRadial])
    print(T.site_source)
            
    # insert timestamp
    T.time = tStp
    
    
    # fill Total with some metadata
    T.metadata['TimeZone'] = rad.metadata['TimeZone']
    T.metadata['AveragingRadius'] = str(search / 1000) + ' km'
    T.metadata['GridAxisOrientation'] = '0.0 DegNCW'
    T.metadata['GridSpacing'] = str(gRes / 1000) + ' km'

    # create Geod object according to the Total CRS
    g = Geod(ellps=T.metadata['GreatCircle'].split()[0])

    # create empty DataFrame for storing indices of radial bins within the search radius of each cell
    combineRadBins = pd.DataFrame(columns=range(len(T.data.index)))
    #print(combineRadBins)
    #print(combineRadBins.shape)
    
    # figure out which radial bins are within the spatthresh of each grid cell
    for index, row in rDF.iterrows():
        rad = row['Radial']         
        thisRadBins = T.data.loc[:,['LOND','LATD']].apply(lambda x: findBins(x, rad, search, g), axis=1)
        #print("\nthisRadBins:\n")
        #print(thisRadBins)
        #print(type(thisRadBins))
        combineRadBins.loc[index] = thisRadBins
    #print("\ncombineRadBins:\n")
    #print(combineRadBins)
    #print(type(combineRadBins))
    
    # loop over grid points and pull out contributing radial vectors
    combineRadBins = combineRadBins.T
    #print(combineRadBins)
    #print(type(combineRadBins))
    totData = combineRadBins.apply(lambda x: makeTotalVector(x, rDF), axis=1)
    
    # assign column names to the combination DataFrame
    totData.columns = ['VELU', 'VELV','VELO','HEAD','UQAL','VQAL','CQAL','GDOP','NRAD']
    #print(totData.dropna())

    # fill Total with combination results
    T.data[['VELU', 'VELV','VELO','HEAD','UQAL','VQAL','CQAL','GDOP','NRAD']] = totData
 
    # mask out vectors on land
    #T.mask_over_land(subset=True)
    
    #T = processTotals(T)
    
    return T


#-------------------------------------------------------------------------------

# qc_radial_file
def applyQC(T, dx, dy):
    
    """
    this function applies QC procedures to Total object
    
    INPUTS: T = total to be processed
            dx = lon limits for spatial median test
            dy = lat limits for spatial median test
        
    OUTPUTS: T = processed Total object
        
    """
    
    # check if Total object contains data
    if T.data.empty:
        print('total file is empty: no QC test applied')
        return T
        
    # initialize QC metadata
    T.initialize_qc()
    
    # DDNS
    T.qc_qartod_data_density_threshold()
    
    # CSPD
    T.qc_qartod_maximum_velocity()

    # Temporal Gradient
    '''
    prevHourTime = T.time-dt.timedelta(minutes=networkData.iloc[0]['temporal_resolution'])
    prevHourFileName = buildEHNtotalFilename(networkData.iloc[0]['network_id'],prevHourTime,'.ttl')
    prevHourTotFile = prevHourFolderPath + prevHourFileName     # previous hour total file
    if os.path.exists(prevHourTotFile):
        with open(prevHourTotFile, 'rb') as ttlFile:
            t0 = pickle.load(ttlFile)
    '''
    
    # GDOP
    T.qc_qartod_gdop_threshold()
    
    # Spatial Median
    T.qc_qartod_spatial_median(dx, dy)

    # Overall QC
    T.qc_qartod_primary_flag(True)
    
    return T


# processTotals
def processTotals(T, dx, dy):
    
    """
    this function processes the input total files and applies QC
    
    INPUTS: T = total to be processed
            dx = lon limits for spatial median test
            dy = lat limits for spatial median test
     
    OUTPUTS:

    """

    '''
    # Add metadata related to bounding box
    lonMin = networkData.iloc[0]['geospatial_lon_min']
    lonMax = networkData.iloc[0]['geospatial_lon_max']
    latMin = networkData.iloc[0]['geospatial_lat_min']
    latMax = networkData.iloc[0]['geospatial_lat_max']
    gridRes = networkData.iloc[0]['grid_resolution']
    dfTot['Total'] = dfTot['Total'].apply(lambda x: addBoundingBoxMetadata(x,lonMin,lonMax,latMin,latMax,gridRes))
    '''    
    
    T = applyQC(T, dx, dy)
    
    # it looks like NAN because a lot of empty grid cells
    #print(T.data)
    
    # get the indexes of grid cells without total vectors
    indexNoVec = T.data[T.data['VELU'].isna()].index
    
    # delete these row indexes from DataFrame
    T.data.drop(indexNoVec, inplace=True)
    T.data.reset_index(level=None, drop=False, inplace=True)    
        # set drop=True if the former indices are not necessary
    
    # add metadata related to bounding box
    #T = addBoundingBoxMetadata(T, lonMin, lonMax, latMin, latMax, gridResolution / 1000)
    
    # update is_combined attribute
    T.is_combined = True
          
    # Convert Total to standard data format (netCDF)
    #dfTot = dfTot.apply(lambda x: applyEHNtotalDataModel(x,networkData,stationData,vers,logger),axis=1) 
    
    #print(T.data)
    
    return T


#-------------------------------------------------------------------------------

# performRadialCombination - combine radials into totals
def performRadialCombination(paths, times, interpolate=False, gridRes=3000, searchRad=5000):
    
    """
    this function performs the least square combination of the input Radials and creates
    a Total object containing the resulting total current data
    
    the function creates a DataFrame containing the resulting Total object along with 
    related information
    
    INPUTS: paths = list of possible paths of radial data
            times = list of datetimes of the processing period (strings)
            interpolate = if radial data has been interpolated (default is False)
            gridRes = grid resolution (default is 3000 meters)
            searchRad = search radius (default is 5000 meters)

    OUTPUTS: T = Total object
        
    """

    # the format of bb = [lonMin, lonMax, latMin, latMax]

    rads, bb = selectRadials(paths, times)
    #print(rads)
    print(bb)
    
    if len(rads) < 2:
        print("Not enough radial data to create total.")
        return Total()
    
    # create the geographical grid
    grid, dx, dy = createGrid(bb[0], bb[1], bb[2], bb[3], gridRes)
    #print(grid.shape)
    
    # get the timestamp
    timeStamp = dt.datetime.strptime(str(rads.iloc[0]['datetime']),'%Y-%m-%d %H:%M:%S')
    
    # add Radial object to dataframe and interpolate
    processRadials(rads, interpolate)
    
    # generate the combined Total from the radial files
    T = combineRadials(rads, grid, searchRad, gridRes, timeStamp)
    
    # fill in some information for Total
    T.file_name = times[0]
    T.grid = grid
    #print(T.file_name)
    
    '''
    print(T.data)
    print(T.data.dropna())
    
    T = ti.interpolation(T, dx, dy, 2)
    
    print(T.data)
    print(T.data.dropna(subset=['VELU']))
    '''
    
    T = processTotals(T, dx, dy)
       
    # save file as pickle file
    pkl_name = 'total_' + times[0] + '.pkl'
    folder = '../total-data/'

    with open(folder + pkl_name, 'wb') as file:
        pickle.dump(T, file)
        
    total_plots = '../total-plots/'
    #T.plot(show=False, shade=True, save=True, save_dir=total_plots, interpolated=interpolate)
    T.plot_cartopy(show=False, save=True, save_dir=total_plots)
    
    return T
    



############################# CALLING FUNCTIONS ################################



# paths = [radial_dir1, radial_dir2, radial_dir3]
# times = ['2024_04_20_0700']
paths = [
    "../radial-data/processed/HATY/",
    "../radial-data/processed/DUCK/",
    "../radial-data/processed/CORE/",
]
# times = ['2014_11_03_0100']
times = [f"2014_11_03_{str(i).zfill(2)}00" for i in range(24)]

# calls the function to turn the radials into a total and also plots the total

for t in times:
    print(f"Processing time: {t}")
    T = performRadialCombination(paths, [t], interpolate=True)
# T = performRadialCombination(paths, times, interpolate=False)

