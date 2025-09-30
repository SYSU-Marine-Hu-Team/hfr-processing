# import packages
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
import xarray as xr
import netCDF4
import pandas as pd
from shapely.geometry import Point
import re
import io
import os
from collections import OrderedDict
import fnmatch
import warnings
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import geopandas as gpd
from geopandas import GeoSeries
from oceans.ocfis import uv2spdir, spdir2uv
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# import from files
from common import fileParser, addBoundingBoxMetadata
from calc import evaluateGDOP

#import total_interpolation as ti

# Total Class
class Total(fileParser):
    
    """
    Totals Subclass

    this class should be used when loading CODAR (.tuv) total files
    """
    
    def __init__(self, fname='', replace_invalid=True, grid=gpd.GeoSeries(), empty_total=False):

        if not fname:
            empty_total = True
            replace_invalid = False
            
        super().__init__(fname)
        for key in self._tables.keys():
            table = self._tables[key]
            if 'LLUV' in table['TableType']:
                self.data = table['data']                
            elif 'src' in table['TableType']:
                self.diagnostics_source = table['data']
            elif 'CUR' in table['TableType']:
                self.cur_data = table['data']
                
        if 'SiteSource' in self.metadata.keys():
            table_data = u''
            for ss in self.metadata['SiteSource']:
                if '%%' in ss:
                    rep = {' comp': '_comp', ' Distance': '_Distance',' Ratio': '_Ratio',' (dB)': '_(dB)',' Width': '_Width', ' Resp': '_Resp', 'Value ': 'Value_','FOL ':'FOL_' }
                    rep = dict((re.escape(k), v) for k, v in rep.items())
                    pattern = re.compile('|'.join(rep.keys()))
                    ss_header = pattern.sub(lambda m: rep[re.escape(m.group(0))], ss).strip('%% SiteSource \n')
                else:
                    ss = ss.replace('%SiteSource:', '').strip()
                    ss = ss.replace('Radial', '').strip()
                    table_data += '{}\n'.format(ss)
            # use pandas read_csv because it interprets the datatype for each column of the csv
            tdf = pd.read_csv(
                io.StringIO(table_data),
                sep=' ',
                header=None,
                names=ss_header.split(),
                skipinitialspace=True
            )
            self.site_source = tdf
                
        # Evaluate GDOP for total files
        if hasattr(self, 'site_source'):
            # Get the site coordinates
            siteLon = self.site_source['Lon'].values.tolist()
            siteLat = self.site_source['Lat'].values.tolist()  
            # Create Geod object according to the Total CRS, if defined. Otherwise use WGS84 ellipsoid
            if self.metadata['GreatCircle']:
                g = Geod(ellps=self.metadata['GreatCircle'].split()[0].replace('"',''))                  
            else:
                g = Geod(ellps='WGS84')
                self.metadata['GreatCircle'] = '"WGS84"' + ' ' + str(g.a) + '  ' + str(1/g.f)
            self.data['GDOP'] = self.data.loc[:,['LOND','LATD']].apply(lambda x: evaluateGDOP(x,siteLon, siteLat, g),axis=1)                
        elif hasattr(self, 'data'):
            self.data['GDOP'] = np.nan
            
        # Evaluate the number of contributing radials (NRAD) for CODAR total files
        if hasattr(self, 'data'):
            self.data['NRAD'] = self.data.loc[:, self.data.columns.str.contains('S.*CN')].sum(axis=1)
            

        if replace_invalid:
            self.replace_invalid_values()
            
        if empty_total:
            self.empty_total()

        if not grid.empty:
            self.initialize_grid(grid)
            
    
    def empty_total(self):
        
        """
        create an empty Total object  by setting the geographical grid
        """

        self.file_path = ''
        self.file_name = ''
        self.full_file = ''
        self.metadata = ''
        self._iscorrupt = False
        self.time = []
        self.grid = GeoSeries()

        for key in self._tables.keys():
            table = self._tables[key]
            self._tables[key]['TableRows'] = '0'
            if 'LLUV' in table['TableType']:
                self.data.drop(self.data.index[:], inplace=True)
                self._tables[key]['data'] = self.data
            elif 'rads' in table['TableType']:
                self.diagnostics_radial.drop(self.diagnostics_radial.index[:], inplace=True)
                self._tables[key]['data'] = self.diagnostics_radial
            elif 'rcvr' in table['TableType']:
                self.diagnostics_hardware.drop(self.diagnostics_hardware.index[:], inplace=True)
                self._tables[key]['data'] = self.diagnostics_hardware
            elif 'RINF' in table['TableType']:
                self.range_information.drop(self.range_information.index[:], inplace=True)
                self._tables[key]['data'] = self.range_information
            elif 'CUR' in table['TableType']:
                self.cur_data.drop(self.cur_data.index[:], inplace=True)
                self._tables[key]['data'] = self.cur_data
                
        if not hasattr(self, 'data'):
            self.data = pd.DataFrame()
        
        if hasattr(self, 'site_source'):
            self.site_source.drop(self.site_source.index[:], inplace=True)
        else:
            self.site_source = pd.DataFrame()
            
    
    def initialize_grid(self, gridGS):
        
        """
        initialize the geogprahic grid for filling the LOND and LATD columns of the 
        Total object data DataFrame
        
        INPUT: gridGS = GeoPandas GeoSeries containing the longitude/latitude pairs of all
               the points in the grid
                
        OUTPUT: DataFrame with filled LOND and LATD columns
        """
        
        # initialize data DataFrame with column names
        self.data = pd.DataFrame(columns=['LOND', 'LATD', 'VELU', 'VELV', 'VELO', 'HEAD', 'UQAL', 'VQAL', 'CQAL', 'GDOP', 'NRAD'])
        
        # extract longitudes and latitude from grid GeoSeries and insert them into data DataFrame
        self.data['LOND'] = gridGS.x
        self.data['LATD'] = gridGS.y
        
        # add metadata about datum and CRS
        self.metadata = OrderedDict()
        self.metadata['GreatCircle'] = ''.join(gridGS.crs.ellipsoid.name.split()) + ' ' + str(gridGS.crs.ellipsoid.semi_major_metre) + '  ' + str(gridGS.crs.ellipsoid.inverse_flattening)
   
        
        
    # plot - plot totals -----------------------------------------------------------
    def plot(self, lon_min=None, lon_max=None, lat_min=None, lat_max=None, shade=False, show=True, save=False, save_dir = None, interpolated=False):
        
        """
        this function plots the current total velocity field (VELU and VELV components) on a 
        Cartesian grid using Basemap
        
        The grid is defined either from the input values or from the Total object
        metadata. If no input is passed and no metadata related to the bounding box are present, the
        grid is defined from data content (LOND and LATD values).
        
        If 'shade' is False (default), a quiver plot with color and magnitude of the vectors proportional to
        current velocity is produced. If 'shade' is True, a quiver plot with uniform vetor lenghts is produced,
        superimposed to a pseudo-color map representing velocity magnitude.
        
        INPUT: lon_min = min lon value (decimal degrees)
               lon_max = max lon value (decimal degrees)
               lat_min = min lat value (decimal degrees)
               lat_max = max lat value (decimal degrees)
                 (if None it is taken from Total metadata)
               shade = boolean for enabling/disabling shade plot (default False)
               show = boolean for enabling/disabling plot visualization (default True)
               save = boolean for saving the plot
               save_dir = directory to save the total plot
               interpolated = boolean for if the radial data was interpolated (append to plot name)
            
        OUTPUT:

        """
        
        # initialize figure
        fig = plt.figure(figsize=(24, 16),tight_layout = {'pad': 2})
        #fig = plt.figure()
        
        plt.subplots_adjust(top = 0.9, bottom = 0.1, left = 0.1, right = 0.9)
        
        # get the bounding box limits
        if not lon_min:
            lon_min = self.data.LOND.min() - 0.25
        if not lon_max:
            lon_max = self.data.LOND.max() + 0.25
        if not lat_min:
            lat_min = self.data.LATD.min() - 0.3
        if not lat_max:
            lat_max = self.data.LATD.max() + 0.4
                
        # evaluate lon and lat of the center of the map
        lon_center = (lon_max + lon_min) / 2
        lat_center = (lat_max + lat_min) / 2
            
        # set the background map
        m = Basemap(llcrnrlon=lon_min, llcrnrlat=lat_min, urcrnrlon=lon_max, urcrnrlat=lat_max, lon_0=lon_center, lat_0=lat_center, resolution = 'i', ellps='WGS84', projection='merc')        
        
        m.drawcoastlines()
        
        m.fillcontinents(lake_color='white')
        m.fillcontinents(color='lightgrey', lake_color='lightblue')
        m.drawparallels(np.arange(lat_min,lat_max))
        m.drawmeridians(np.arange(lon_min,lon_max))
        m.drawmapboundary(fill_color='white')
        
        m.bluemarble()
        
        # get station coordinates and codes
        siteLon = self.site_source['Lon'].to_numpy()
        siteLat = self.site_source['Lat'].to_numpy()
        siteCode = self.site_source['Name'].tolist()
        
        # compute the native map projection coordinates for the stations
        xS, yS = m(siteLon, siteLat)       
        
        print(siteCode)
        
        # plot radial stations
        m.plot(xS,yS,'rD')
        for label, xs, ys in zip(siteCode,xS,yS):
            plt.text(xs + 0.1,ys + 0.1,label,fontdict={'fontsize': 16, 'fontweight' : 'bold'})
        
        # plot velocity field
        if shade:
            '''
            self.to_xarray()
            
            # Create grid from longitude and latitude
            [longitudes, latitudes] = np.meshgrid(self.xdr['LONGITUDE'].data, self.xdr['LATITUDE'].data)
            
            # Compute the native map projection coordinates for the pseudo-color cells
            X, Y = m(longitudes, latitudes)

            # Create velocity variable in the shape of the grid
            V = abs(self.xdr['VELO'][0,0,:,:].to_numpy())
            # V = V[:-1,:-1]            
            
            # Make the pseudo-color plot
            warnings.simplefilter("ignore", category=UserWarning)
            c = m.pcolormesh(X, Y, V, shading='nearest', cmap=plt.cm.jet, vmin=0, vmax=1)
            '''
           
            
            #self.data = self.data[self.data['INTP'] == 2]
            
            #print(self.data)
            
            # Compute the native map projection coordinates for the vectors
            x, y = m(self.data.LOND, self.data.LATD)
            
            
            
            # Create the velocity component variables
            u = self.data.VELU / 100       # CODAR velocities are in cm/s
            v = self.data.VELV / 100       # CODAR velocities are in cm/s
            vel = abs(self.data.VELO) / 100
            
            angle, speed = uv2spdir(u.squeeze(), v.squeeze())
            u_norm, v_norm = spdir2uv(np.ones_like(speed), angle, deg=True)
            
            #color_clipped = np.clip(speed, 0, 1).squeeze()
            
            #print(len(u))
            
            # Make the quiver plot
            m.quiver(x, y, u * 0.75 + u_norm * 0.25, v * 0.75 + v_norm * 0.25, vel, cmap=plt.cm.jet, width=0.001, headwidth=4, headlength=4, headaxislength=4)
            #m.quiver(x, y, u, v, vel, cmap=plt.cm.jet, width=0.001, headwidth=4, headlength=4, headaxislength=4)
            
            #warnings.simplefilter("default", category=UserWarning)
            
        else:
            # Compute the native map projection coordinates for the vectors
            x, y = m(self.data.LOND, self.data.LATD)
            
            # Create the velocity variables
            u = self.data.VELU / 100                # CODAR velocities are in cm/s
            v = self.data.VELV / 100                # CODAR velocities are in cm/s
            vel = abs(self.data.VELO) / 100         # CODAR velocities are in cm/s                
            
            angle, speed = uv2spdir(u.squeeze(), v.squeeze())
            color_clipped = np.clip(speed, 0, 1).squeeze()
            
            # Make the quiver plot
            m.quiver(x, y, u, v, vel, cmap=plt.cm.jet, width=0.001, headwidth=4, headlength=4, headaxislength=4)
            
        # Add colorbar
        cbar = plt.colorbar(fraction=0.028, pad=0.02)
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label('m/s', fontsize=18)
        
        # Add title
        plt.title(self.file_name + ' Total Velocity Field', fontdict={'fontsize': 24, 'fontweight' : 'bold'})
                
        if show:
            plt.show()
        
        save_dir = '../total-plots/'
        photo_name = 'total_' + self.file_name + '.png'
            
        if interpolated:
            #print("INTERPOLATED")
            save_dir = save_dir + 'interpolated/'
            
        if save:
            #print("SAVE")
            print(save_dir)
            print(photo_name)
            fig.savefig(save_dir + photo_name)
        
        return fig

    def plot_cartopy(self, show=True, save=False, save_dir=None):
        
        """
        this function plots the current total velocity field (VELU and VELV components) on a 
        Cartesian grid using Cartopy
        
        The grid is defined from the Total data content.
        
        INPUT: show = boolean for enabling/disabling plot visualization (default True)
               save = boolean for saving the plot
               save_dir = directory to save the total plot
               interpolated = boolean for if the radial data was interpolated (append to plot name)
            
        OUTPUT:

        """
        
        # initialize figure
        fig = plt.figure(figsize=(24, 16))
        #fig = plt.figure()
        
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
        
        fig.tight_layout(pad=8)

        siteLon = self.site_source['Lon'].to_numpy()
        siteLat = self.site_source['Lat'].to_numpy()
        
        # get the bounding box limits
        lon_min = min(self.data.LOND.min(), siteLon.min()) - 0.25
        lon_max = max(self.data.LOND.max(), siteLon.max()) + 0.25
        lat_min = min(self.data.LATD.min(), siteLat.min()) - 0.3
        lat_max = max(self.data.LATD.max(), siteLat.max()) + 0.5
         
        # Set colors of the land. 
        edgecolor = 'black'
        landcolor = 'lightgrey'

        LAND = cfeature.NaturalEarthFeature(
            'physical', 'land', '10m',
            edgecolor='face',
            facecolor='tan'
        )

        state_lines = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none'
        )
        
        # Gridlines and grid labels
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=.5,
            color='black',
            alpha=0.25,
            linestyle='--'
        )

        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = {'size': 16, 'color': 'black'}
        gl.ylabel_style = {'size': 16, 'color': 'black'}
        gl.xlocator = mticker.MaxNLocator(integer=True)
        gl.ylocator = mticker.MaxNLocator(integer=True)
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        ax.tick_params(which='major',
                       direction='out',
                       bottom=True, top=True,
                       labelbottom=True, labeltop=False,
                       left=True, right=True,
                       labelleft=True, labelright=False,
                       length=5, width=2)

        ax.tick_params(which='minor',
                       direction='out',
                       bottom=True, top=True,
                       labelbottom=True, labeltop=False,
                       left=True, right=True,
                       labelleft=True, labelright=False,
                       width=1)

        
        x = self.data.LOND
        y = self.data.LATD
        
        extent = [lon_min, lon_max, lat_min, lat_max]
        ax.set_extent(extent)
        #print(extent)
        
        ax.add_feature(LAND, edgecolor=edgecolor, facecolor=landcolor)
        #ax.add_feature(cfeature.OCEAN)
        #ax.add_feature(cfeature.RIVERS)
        #ax.add_feature(cfeature.LAKES)
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(state_lines, zorder=11, edgecolor=edgecolor)
        
        # get station coordinates and codes
        siteLon = self.site_source['Lon'].to_numpy()
        siteLat = self.site_source['Lat'].to_numpy()
        siteCode = self.site_source['Name'].tolist()

        
        ax.scatter(siteLon, siteLat, color='red', s=50, transform=ccrs.PlateCarree())
        
        for label, xs, ys in zip(siteCode,siteLon,siteLat):
            ax.text(xs, ys + 0.05, label, fontsize=18, fontweight='bold', color='red', transform=ccrs.PlateCarree())
        
        # Create the velocity component variables

        u = self.data.VELU / 100       # CODAR velocities are in cm/s
        v = self.data.VELV / 100       # CODAR velocities are in cm/s
        vel = abs(self.data.VELO) / 100
        
        angle, speed = uv2spdir(u.squeeze(), v.squeeze())
        u_norm, v_norm = spdir2uv(np.ones_like(speed), angle, deg=True)
        
        #color_clipped = np.clip(speed, 0, 1).squeeze()

        # Make the quiver plot
        plt.quiver(x, y, u * 0.75 + u_norm * 0.25, v * 0.75 + v_norm * 0.25, vel, cmap=plt.cm.jet, width=0.001, headwidth=4, headlength=4, headaxislength=4, transform=ccrs.PlateCarree())

        # Add colorbar
        cbar = plt.colorbar(fraction=0.028, pad=0.02)
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label('m/s', fontsize=18)
        
        # Add title
        plt.title(self.file_name + ' Total Velocity Field', fontdict={'fontsize': 28, 'fontweight' : 'bold'})
                
        if show:
            plt.show()
        
        photo_name = 'total_' + self.file_name + '.png'

        if save:
            print(save_dir + photo_name)
            fig.savefig(save_dir + photo_name)
        
        return fig


    # def to_xarray_multidimensional
    def to_xarray(self, lon_min=None, lon_max=None, lat_min=None, lat_max=None, grid_res=None):
        
        """
        this function creates a dictionary of xarray DataArrays
          
        coordinate axes are set as (TIME, DEPTH, LATITUDE, LONGITUDE)
        
        INPUT: lon_min (decimal degrees)
               lat_max (decimal degrees)
               lat_min (decimal degrees)
               lat_max (decimal degrees)
               grid_res (meters)
        
            (otherwise info taken from metadata)
                
            
        OUTPUT: (generated dictionary is attached to Total object = xdr)

        """
        
        # initialize empty dictionary
        xdr = OrderedDict()
        
        '''    
        # longitude limits   
        if lon_min is None:
            lon_min = self.data.LOND.min()
        if lon_max is None:
            lon_max = self.data.LOND.max()
                
        # latitude limits   
        if lat_min is None:
            lat_min = self.data.LATD.min()
        if lat_max is None:
            lat_max = self.data.LATD.max()       
                
        # grid resolution
        if grid_res is None:                
            grid_res = float(3000)            
                
        # generate grid coordinates
        gridGS = createGrid(lon_min, lon_max, lat_min, lat_max, grid_res)
        '''   
                         
        # extract lons and lats from grid GeoSeries and insert them into numpy arrays
        lon_dim = np.unique(self.grid.x.to_numpy())
        lat_dim = np.unique(self.grid.y.to_numpy())
        
        # manage antimeridian crossing
        lon_dim = np.concatenate((lon_dim[lon_dim >= 0],lon_dim[lon_dim < 0]))         

        # create total grid from lons and lats
        [longitudes, latitudes] = np.meshgrid(lon_dim, lat_dim)

        # find grid indices from lon / lat grid (longitudes, latitudes)
        lat_map_idx = np.tile(np.nan, self.data['LATD'].shape)
        lon_map_idx = np.tile(np.nan, self.data['LOND'].shape)

        for i, line in enumerate(self.data['LATD']):
            lat_map_idx[i] = np.argmin(np.abs(lat_dim - self.data.LATD[i]))
            lon_map_idx[i] = np.argmin(np.abs(lon_dim - self.data.LOND[i]))
            
        # set X and Y coordinate mappings
        X_map_idx = lon_map_idx             # LONGITUDE is X axis
        Y_map_idx = lat_map_idx             # LATITUDE is Y axis
            
        # create dictionary containing variables from dataframe in the shape of total grid
        d = {key: np.tile(np.nan, longitudes.shape) for key in self.data.keys()}       
            
        # remap all variables
        for k, v in d.items():
            v[Y_map_idx.astype(int), X_map_idx.astype(int)] = self.data[k]
            d[k] = v

        # add extra dimensions for time (T) and depth (Z) - CF Standard: T, Z, Y, X -> T=axis0, Z=axis1
        d = {k: np.expand_dims(np.float32(v), axis=(0, 1)) for (k, v) in d.items()}            

        # drop LOND and LATD variables (they are set as coordinates of the DataSet)
        d.pop('LOND')
        d.pop('LATD')

        # scale velocities to be in m/s
        toMs = ['VELU', 'VELV', 'VELO', 'UQAL', 'VQAL','CQAL']
        for t in toMs:
            if t in d:
                d[t] = d[t] * 0.01

        # evaluate timestamp as number of days since 1950-01-01T00:00:00Z
        timeDelta = self.time - dt.datetime.strptime('1950-01-01T00:00:00Z','%Y-%m-%dT%H:%M:%SZ')
        ncTime = timeDelta.days + timeDelta.seconds / (60 * 60 * 24)
        
        # add all variables as xarray
        for k, v in d.items():
            xdr[k] = xr.DataArray(v, dims={'TIME': v.shape[0], 
                                           'DEPTH': v.shape[1], 
                                           'LATITUDE': v.shape[2], 
                                           'LONGITUDE': v.shape[3]}, 
                                     coords={'TIME': [ncTime], 
                                             'DEPTH': [0], 
                                             'LATITUDE': lat_dim, 
                                             'LONGITUDE': lon_dim})
        
        # add DataArray for coordinate variables
        xdr['TIME'] = xr.DataArray(ncTime,
                                   dims={'TIME': len(pd.date_range(self.time, periods=1))},
                                   coords={'TIME': [ncTime]})
        xdr['DEPTH'] = xr.DataArray(0,
                                    dims={'DEPTH': 1},
                                    coords={'DEPTH': [0]})
        xdr['LATITUDE'] = xr.DataArray(lat_dim,
                                       dims={'LATITUDE': lat_dim},
                                       coords={'LATITUDE': lat_dim})
        xdr['LONGITUDE'] = xr.DataArray(lon_dim,
                                        dims={'LONGITUDE': lon_dim},
                                        coords={'LONGITUDE': lon_dim})  
        
        # attach the dictionary to the Total object
        xdr = xr.Dataset(xdr)
        self.xdr =  xdr
        
        return xdr
    

    def initialize_qc(self):
        
        """
        initialize dictionary entry for QC metadata
        """
        
        # initialize dictionary entry for QC metadta
        self.metadata['QCTest'] = {}
        
        
        
        
    # qc_ehn_data_density_threshold - QC301    
    def qc_qartod_data_density_threshold(self, minContrRad=2):
        
        """
        this test labels total velocity vectors with a number of contributing radial velocities bigger 
        than the minimum number defined for normal operations with a “good data” flag
        
        INPUTS: minContrRad = min number of contributing radial velocities for normal operations                     
        """
        
        # set the test name
        testName = 'QC301'
        
        # Add new column to the DataFrame for QC data by setting every row as passing the test (flag = 1)
        self.data.loc[:,testName] = 1
    
        # set bad flag for velocities not passing the test
        if 'NRAD' in self.data.columns:
            self.data.loc[(self.data['NRAD'] < minContrRad), testName] = 4
            #print("DATA DENSITY THRESHOLD")
    
        self.metadata['QCTest'][testName] = 'Data Density Threshold QC Test - Test applies to each vector. ' \
            + 'Threshold=[' + f'minimum number of contributing radial velocities={minContrRad}]'
        
    
    
        # qc_ehn_threshold - QC302    
    def qc_qartod_gdop_threshold(self, maxGDOP=2):
        
        """
        this test labels total velocity vectors whose GDOP is smaller than a maximum GDOP threshold 
        with a “good data” flag
        
        INPUTS: maxGDOP = maximum allowed GDOP for normal operations                     
        """
        
        # set the test name
        testName = 'QC302'
        
        # add new column to the DataFrame, set every row as passing the test (flag = 1)
        self.data.loc[:,testName] = 1
    
        # set bad flag for velocities not passing the test
        self.data.loc[(self.data['GDOP'] > maxGDOP), testName] = 4
    
        self.metadata['QCTest'][testName] = 'GDOP Threshold QC Test - Test applies to each vector. ' \
            + 'Threshold=[' + f'GDOP threshold={maxGDOP}]'
        
     
     
        
    # qc_ehn_maximum_velocity - QC303    
    def qc_qartod_maximum_velocity(self, maxSpeed=250):
        
        """
        this test labels total velocity vectors whose module is smaller than a maximum velocity threshold 
        with a “good data” flag
        
        INPUTS: maxSpeed = maximum velocity in m/s for normal operations                     
        """
        
        # set the test name
        testName = "QC303"
        
        # make sure VELO is float
        self.data["VELO"] = self.data["VELO"].astype(float)
        
        # Add new column to the DataFrame for QC data by setting every row as passing the test (flag = 1)
        self.data.loc[:,testName] = 1
    
        # set bad flag for velocities not passing the test
        self.data.loc[(self.data['VELO'].abs() > maxSpeed), testName] = 4
    
        self.metadata['QCTest'][testName] = 'Velocity Threshold QC Test - Test applies to each vector. ' \
            + 'Threshold=[' + f'maximum velocity={maxSpeed} (cm/s)]'

       
    # QC304
    def qc_qartod_spatial_median(self, dx, dy, limit = 2, diff=25):
        
        testName = 'QC304'
        
        self.data.loc[:,testName] = 1
        
        #print(dx)
        #print(dy)
        
        velo = self.data['VELO'].to_numpy()
        #print(velo.shape)
        diffcol = self.data['VELO'].to_numpy() + np.nan
        
        velo = velo.reshape((dx, dy))
        #print(velo.shape)
        diffcol = diffcol.reshape((dx, dy))
        
        
        padded = np.pad(velo, pad_width=((limit, limit), (limit, limit)), mode='constant', constant_values=np.nan)
        diffcol = np.pad(velo, pad_width=((limit, limit), (limit, limit)), mode='constant', constant_values=np.nan)
        #print(padded.shape)

        for row in range(limit, dx + limit):
            for col in range(limit, dy + limit):
                temp = padded[row - limit: row + limit + 1, col - limit: col + limit + 1]
                median = np.nanmedian(temp)
                diffcol[row][col] = padded[row][col] - median
        
        diffcol = diffcol[limit:-limit, limit:-limit]
        diffcol = diffcol.flatten()
        
        boolean = np.abs(diffcol) > diff
        
        
        self.data[testName] = self.data[testName].where(~boolean, other=4)
        
        #print(self.data[self.data[testName] == 4])
        self.metadata['QCTest'][testName] = 'Spatial Median QC Test - Test applies to each vector. ' \
            + 'Thresholds=[' + f'current_difference={str(diff)} (cm/s)]'
        
    
    # qc_ehn_temporal_derivative - QC206    
    def qc_qartod_temporal_derivative(self, t0, tempDerThr=100):
        
        """
        this test compares the velocity of each total vector with the velocity of the total vector 
        measured in the previous timestamp at the same location.
        Each vector for which the velocity difference is smaller than the specified threshold for normal 
        operations (tempDerThr), is labeled with a "good data" flag.
        
        INPUTS: t0 = Total object of the previous timestamp
                tempDerThr = velocity difference threshold in m/s for normal operations
        """
        
        # Set the test name
        testName = 'QC206'
        
        # Check if the previous timestamp total file exists
        if not t0 is None:
            # Merge the data DataFrame of the two Totals and evaluate velocity differences at each location
            mergedDF = self.data.merge(t0.data, on=['LOND', 'LATD'], how='left', suffixes=(None, '_x'), indicator='Exist')
            velDiff = (mergedDF['VELO'] - mergedDF['VELO_x']).abs()

            # Add new column to the DataFrame for QC data by setting every row as passing the test (flag = 1)
            self.data.loc[:,testName] = 1

            # Set rows of the DataFrame for QC data as not evaluated (flag = 0) for locations existing in the current total but not in the previous one
            self.data.loc[mergedDF['Exist'] == 'left_only', testName] = 0

            # Set bad flag for vectors not passing the test
            self.data.loc[(velDiff > tempDerThr), testName] = 4         # velocity in cm/s (LLUV)

        else:
            # Add new column to the DataFrame for QC data by setting every row as not evaluated (flag = 0)
            self.data.loc[:,testName] = 0
        
        self.metadata['QCTest'][testName] = 'Temporal Derivative QC Test - Test applies to each vector. ' \
            + 'Threshold=[' + f'velocity difference threshold={str(tempDerThr)} (cm/s)]'
 
    
    # qc_ehn_overall_qc_flag    
    def qc_qartod_primary_flag(self, erase=False):
        
        """
        this QC test labels total velocity vectors with a ‘good_data” flag if all QC tests are passed
        
        INPUTS: erase = if True, then it will remove data with a bad flag
            
        """
        
        # Set the test name
        testName = 'PRIM'
        
        # Add new column to the DataFrame for QC data by setting every row as not passing the test (flag = 4)
        self.data[testName] = 4
        
        # Set good flags for vectors passing all QC tests
        self.data.loc[self.data.loc[:, self.data.columns.str.contains('QC')].eq(1).all(axis=1), testName] = 1

        self.metadata['QCTest'][testName] = 'Overall QC Flag - Test applies to each vector. Test checks if all QC tests are passed.'
        
        #print(self.data)
        
        if erase:
            indexes = self.data[self.data['PRIM'] == 4].index
            self.data.drop(indexes, inplace=True)
            self.data.reset_index(level=None, drop=False, inplace=True)
            #print("ERASED")
             

    def file_type(self):
        """
        Return a string representing the type of file this is.
        """
        return 'totals'
    
    


