
# Created Sep 2024 by Cathleen Qiao

# This file is for interpolating radial data, using scipy and a bilinear 
# interpolation method. 

# Look at these repositories for more information:
# https://github.com/LorenzoCorgnati/EU_HFR_NODE_pyHFR/tree/main
# https://github.com/rucool/hfradarpy/tree/master


# import libraries
from radials import Radial

import glob
import os
import xarray as xr

# read in the data of a radial file
#radial_dir1 = '../radials_clean/MARA/'
#radial_dir2 = '../radials_clean/WEST/'
#radial_dir3 = '../radials_clean/JEFF/'
          
# save directory
#save_dir = './post_int_images'

# use glob to find radial files (*
#files = sorted(glob.glob(os.path.join(radial_dir1, '*.ruv')))

# select file
#R = Radial(files[70])

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from statistics import mean

''' 
FUNCTIONS IN THIS FILE:
    interpolation(R, d)
    plot(x, y, u, v, colors)
'''

################################################################################

def interpolation(R, d):

    '''
    
    this function interpolates the radial data based on the limit set

    Input: R = Radial object
           d = limit of interpolation
           
    Output: R = interpolated Radial object

    '''

    # get dataframe for radial data
    df = R.data
    #print(df)

    # separate the data to have specific variables (put into numpy arrays)
    lon_orig = df['LOND'].to_numpy()
    lat_orig = df['LATD'].to_numpy()

    rg = df['SPRC'].to_numpy()

    u_orig = df['VELU'].to_numpy()
    v_orig = df['VELV'].to_numpy()

    ranges = df['RNGE'].to_numpy()
    bearings = df['BEAR'].to_numpy()

    vel_orig = df['VELO'].to_numpy()
    std_orig = df['ETMP'].to_numpy()

    # getting dimensions for matrices
    maxr = np.max(rg)
    minr = np.min(rg)
    maxc = np.max(bearings)
    minc = np.min(bearings)

    # rows = based on range, cols = based on bearing
    rows = maxr - minr + 1
    cols = int((maxc - minc) / 2) + 1

    # creating matrices
    lon_matrix = np.zeros((rows, cols))
    lat_matrix = np.zeros((rows, cols))
    
    u_matrix = np.zeros((rows, cols))
    v_matrix = np.zeros((rows, cols))
    
    interpolated_matrix = np.zeros((rows, cols))
    
    velo_matrix = np.zeros((rows, cols))
    std_matrix = np.zeros((rows, cols))

    # putting in radial velocity into matrix
    for r, b, n, t, u, v, velo, std in zip(rg, bearings, lon_orig, lat_orig, u_orig, v_orig, vel_orig, std_orig):
        
        # r for the row
        r = r - minr
        # b for the column
        b = int((b - minc) / 2)
        
        # setting matrix element
        lon_matrix[r][b] = n
        lat_matrix[r][b] = t
        u_matrix[r][b] = u
        v_matrix[r][b] = v
        velo_matrix[r][b] = velo
        std_matrix[r][b] = std
        interpolated_matrix[r][b] = 1   # 1 means original clean data
   
    # getting interpolated u and v data
    u_int = []
    v_int = []

    # interpolated lon and lat
    lon_int = []
    lat_int = []
    
    # other interpolated data
    velo_int = []
    std_int = []
    head = []

    # dictionaries
    temp_lon = {}
    temp_lat = {}
    temp_bear = {}

    # separate list into dict by range (each range is a key)
    def create_dict(list1, dict1):
        for r, l in zip(ranges, list1):
            if r in dict1:
                dict1[r].append(l)
            else:
                dict1[r] = [l]
            
    create_dict(lon_orig, temp_lon)
    create_dict(lat_orig, temp_lat)
    create_dict(bearings, temp_bear)

    # goes through each key / range
    for n, t, b, r in zip(temp_lon, temp_lat, temp_bear, range(rows)):

        # start of cluster
        start = 0
        
        # going through each bearing
        for i in range(1, len(temp_bear[b])):
        
            # checking if next point is too far or at the end
            if temp_bear[b][i] - temp_bear[b][i - 1] > 12 or i == len(temp_bear[b]) - 1:
                if i == len(temp_bear[b]) - 1:
                    i = i + 1
                cluster = temp_bear[b][start:i]
                
                # checking if cluster has enough points
                if len(cluster) > 2:
                    # +2 to finish the interpolation at the end
                    new_angles = np.arange(cluster[0], cluster[-1] + 2, 2)
                    if_lon = interp1d(np.array(cluster), np.array(temp_lon[n][start:i]), kind='quadratic', fill_value="extrapolate")
                    if_lat = interp1d(np.array(cluster), np.array(temp_lat[t][start:i]), kind='quadratic', fill_value="extrapolate")
                    i_lon = if_lon(new_angles)
                    i_lat = if_lat(new_angles)

                    # adding each interpolated lon and lat to the matrix
                    for o, a, p in zip(i_lon, i_lat, new_angles):
                        
                        # getting the index for bearing
                        c = int((p - minc) / 2)

                        if c < len(interpolated_matrix[r]) and interpolated_matrix[r][c] != 1:
                            lon_matrix[r][c] = o
                            lat_matrix[r][c] = a
                            interpolated_matrix[r][c] = 2
                    
                # updating start of cluster
                start = i

    # check for row or column interpolation (have neighbors)   
    def check_next(r, c, rl, cl):
        if (interpolated_matrix[r][c-1] == 1 and interpolated_matrix[r][c+1]==1):
            rl.append(r)
            rl.append(r)
            cl.append(c-1)
            cl.append(c+1)
        if (interpolated_matrix[r-1][c] == 1 and interpolated_matrix[r+1][c] == 1):
            rl.append(r-1)
            rl.append(r+1)
            cl.append(c)
            cl.append(c)
        if (len(rl) > 0):
            return True
        return False
        
        
    def calculate_mean(rl, cl, m, l):
        t = []
        for r, c in zip(rl, cl):
            t.append(m[r][c])
        result = mean(t)
        m[r][c] = result
        l.append(result)

    # checking if grid is there
    def check_grid(r, c, a):
        if (interpolated_matrix[r + a[0]][c - a[2]] != 1 or 
            interpolated_matrix[r - a[1]][c - a[2]] != 1 or
            interpolated_matrix[r + a[0]][c + a[3]] != 1 or
            interpolated_matrix[r - a[1]][c + a[3]] != 1):
                return False
        return True


    # possible grids
    def get_grid(r, c, a):
        
        # check [1, 1, 1, 1]
        if check_grid(r, c, a):
            return True

        # check when one entry is 2
        for e in a:
            e = 2
            if check_grid(r, c, a):
                return True
            e = 1
        
        # when two entries are 2
        for i in range(4):
            a[i] = 2
            for j in range(i + 1, 4):
                a[j] = 2
                if check_grid(r, c, a):
                    return True
                a[j] = 1
            a[i] = 1     
        
        return False
            
    # array: [dr1, dr2, dc1, dc2] 

    # bilinear interpolation formula
    def bilinear(m, l, r, c, a):

        b00 = m[r + a[0]][c - a[2]]
        b01 = m[r - a[1]][c - a[2]]
        b10 = m[r + a[0]][c + a[3]]
        b11 = m[r - a[1]][c + a[3]]
        
        b1 = (b00 * a[0] + b01 * a[1]) / (a[0] + a[1])
        b2 = (b10 * a[0] + b11 * a[1]) / (a[0] + a[1])
        
        result = (b1 * a[2] + b2 * a[3]) / (a[2] + a[3])
        
        m[r][c] = result
        l.append(result)
          
    def add_values(r, c):
        interpolated_matrix[r][c] = 3
        lon_int.append(lon_matrix[r][c])
        lat_int.append(lat_matrix[r][c])
        head.append(((c + 1) * 2 + minc + 180) % 360)
        

    # applying the interpolation
    for r in range(d, rows - d):
    
        for c in range(d, cols - d):
            
            a = [1, 1, 1, 1]
            
            # if lon + lat has been interpolated
            if interpolated_matrix[r][c] == 2:
                
                rl = []
                cl = []
            
                # for points that are close enough
                if (check_next(r, c, rl, cl)):
                    calculate_mean(rl, cl, u_matrix, u_int)
                    calculate_mean(rl, cl, v_matrix, v_int)
                    calculate_mean(rl, cl, velo_matrix, velo_int)
                    calculate_mean(rl, cl, std_matrix, std_int)
                    add_values(r, c)
            
                # add points if grid has been found
                elif get_grid(r, c, a):
                    bilinear(u_matrix, u_int, r, c, a)
                    bilinear(v_matrix, v_int, r, c, a)
                    bilinear(velo_matrix, velo_int, r, c, a)
                    bilinear(std_matrix, std_int, r, c, a)
                    add_values(r, c)  

    # turning into numpy arrays (combine both)
    lon = np.concatenate((lon_orig, lon_int)) 
    lat = np.concatenate((lat_orig, lat_int)) 
    
    def add_data():
        new = pd.DataFrame({
            'LOND': lon_int,
            'LATD': lat_int,
            'HEAD': head,
            'VELU': u_int,
            'VELV': v_int,
            'VELO': velo_int,
            'ETMP': std_int,
        })
        
        
        R.data = pd.concat([R.data, new]).reset_index(drop=True)
            
   
    add_data()
    
    return R

    # turning into numpy arrays (combine both) for plotting purposes
    #x = np.concatenate((lon_original, lon_lin, lon_bilin)) 
    #y = np.concatenate((lat_original, lat_lin, lat_bilin)) 
    #u = np.concatenate((u_original, u_lin, u_bilin))
    #v = np.concatenate((v_original, v_lin, v_bilin))
    #colors = np.concatenate([['c'] * len(u_original), ['tomato'] * len(u_lin), ['b'] * len(u_bilin)])
   
''' FOR TESTING 
print(R.data)
R = interpolation(R, 2)    
print(R.data)  
'''

# import plotting libraries
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import colors
from matplotlib.colors import TwoSlopeNorm
import matplotlib.lines as mlines


# plotting from the matrix (to monitor progress)
def plot(x, y, u, v, colors):

    # Intialize an empty subplot using cartopy
    fig, ax = plt.subplots(
        #figsize=(11, 8),
        subplot_kw=dict(projection=ccrs.Mercator())
    )

    # quiver configurations
    scale=None
    headwidth=3
    headlength=5
    headaxislength=5
    #velocity_min = -40
    #velocity_max = 40

    # plot details
    title = 'Radial Map: Original and Interpolated Data'
    plt.title(f'{title}\n')

    offset = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1) 

    # setting the extent
    extent = [x.min()-0.2, x.max()+0.2, y.min()-0.2, y.max()+0.2]
    ax.set_extent(extent)

    qargs = dict(scale=scale, headwidth=headwidth, headlength=headlength, headaxislength=headaxislength)
    qargs['transform'] = ccrs.PlateCarree()
    qargs['norm'] = offset

    # add legend
    legend_lines = [mlines.Line2D([], [], color='c', lw=3, label='Original Data'), 
                     mlines.Line2D([], [], color='tomato', lw=3, label='Interpolated Data'),
                     mlines.Line2D([], [], color='b', lw=3, label='Bilinear Data')]

    # plot arrows
    q1 = ax.quiver(
        x,
        y,
        u,
        v,
        color=colors,
        **qargs
    )

    # include legend
    #plt.scatter(x, y)
    ax.legend(handles=legend_lines)
    plt.show()

