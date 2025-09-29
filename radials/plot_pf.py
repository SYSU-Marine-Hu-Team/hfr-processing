
# This file is for saving the pass/fail plots of all the flagged radial data. 

# Look at hfradarpy repository for more information:
# https://github.com/rucool/hfradarpy


from hfradarpy.radials import Radial, qc_radial_file
import glob
import os
import xarray as xr

site = 'HATY/'

# Path to radial directory
radial_dir = '../radial-data/flagged/' + site
save_dir = '../radial-plots/flagged/' + site


# Use glob to find radial files (*
# files = sorted(glob.glob(os.path.join(radial_dir, '*.ruv')))
files = sorted(glob.glob(os.path.join(radial_dir, '*')))


def plot_pf(r):
    tds = r.to_xarray('gridded', enhance=True).squeeze()

    # Lets get rid of the single time dimension. It will cause problems during plotting
    tds = tds.squeeze()

    # Import matplotlib and cartopy
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    # Adjust some standard plotting settings to make them the size of a sheet of paper
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 8
    plt.rcParams["figure.figsize"] = fig_size

    # Set colors of the land. 
    edgecolor = 'black'
    landcolor = 'tan'

    # LAND = cfeature.NaturalEarthFeature(
    #     'physical', 'land', '10m',
    #     edgecolor='face',
    #     facecolor='tan'
    # )

    # state_lines = cfeature.NaturalEarthFeature(
    #     category='cultural',
    #     name='admin_1_states_provinces_lines',
    #     scale='50m',
    #     facecolor='none'
    # )

    extent = []

    dx = dy = 0.2  # Area around the point of interest.

    # make adjustments to fit the data and get consistent boundaries
    extent = [tds.lon.min()-dx, tds.lon.max()+dx, tds.lat.min()-dy, tds.lat.max()+dy]
    #extent = [-84.8567, -80.8899, 22.7965, 26.4564]
    #print(extent)

    # Create a re-usable function for map features that we can pass an axes to.
    def map_features(ax):
        # Axes properties and features
        ax.set_extent(extent)
        # ax.add_feature(LAND, edgecolor=edgecolor, facecolor=landcolor)
        ax.add_feature(cfeature.LAND, edgecolor=edgecolor, facecolor=landcolor)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.RIVERS)
        ax.add_feature(cfeature.LAKES)
        # ax.add_feature(cfeature.BORDERS)
        # ax.add_feature(state_lines, zorder=11, edgecolor=edgecolor)

        # Gridlines and grid labels
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=.5,
            color='black',
            alpha=0.25,
            linestyle='--'
        )

        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = {'size': 10, 'color': 'black'}
        gl.ylabel_style = {'size': 10, 'color': 'black'}
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

    # Intialize an empty subplot using cartopy
    fig, ax = plt.subplots(
        figsize=(11, 8),
        subplot_kw=dict(projection=ccrs.Mercator())
    )

    # Get the receiver location for plotting purposes
    receiver_location = [float(x) for x in tds.Origin.split('  ')]
    receiver_location.reverse()
    receiver_location

    import numpy.ma as ma
    import numpy as np

    time = tds.time
    lon = tds.coords['lon'].data
    lat = tds.coords['lat'].data
    u = tds['u'].data
    v = tds['v'].data

    u = ma.masked_invalid(u)
    v = ma.masked_invalid(v)

    from oceans.ocfis import uv2spdir, spdir2uv

    angle, speed = uv2spdir(u, v)  # convert u/v to angle and speed

    u, v = spdir2uv(  # convert angle and speed back to u/v, normalizing the arrow sizes
        np.ones_like(speed),
        angle,
        deg=True
    )

    from matplotlib import colors
    from matplotlib.colors import TwoSlopeNorm, Normalize
    
    scale=None
    headwidth=3
    headlength=5
    headaxislength=5
    sub=1
    velocity_min = -40
    velocity_max = 40
    cbar_step = 10
    offset = Normalize(vmin=velocity_min, vmax=velocity_max, clip=True)

    color_clipped = tds.primary_flag_qc.where(tds.primary_flag_qc == 1, other=-1).data  # PRIM == 1 where vectors pass qc
    
    #color_clipped = tds.vector_flag.where(tds.vector_flag == 0, other=-1).data
    
    offset = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    title = "Radial Map: QC Pass/Fail"
    cmap = colors.ListedColormap(['red', 'blue'])

    fig, ax = plt.subplots(
            figsize=(11, 8),
            subplot_kw=dict(projection=ccrs.Mercator())
        )

    # Plot title
    plt.title(f'{title}\n{tds.time.data}')


    qargs = dict(cmap=cmap, scale=scale, headwidth=headwidth, headlength=headlength, headaxislength=headaxislength)
    qargs['transform'] = ccrs.PlateCarree()
    qargs['norm'] = offset

    # plot arrows over pcolor
    h = ax.quiver(
        lon[::sub],
        lat[::sub],
        u[::sub],
        v[::sub],
        color_clipped,
        **qargs
    )
    
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    
    green = mpatches.Patch(color='blue', label='Pass')
    red = mpatches.Patch(color='red', label='Fail')
    
    map_features(ax)
    plt.legend(handles=[green, red])
    
    file_name = r.file_name[:-4:] + '_PF.png'

    print(f'{file_name}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f'Saved figure to {save_dir + file_name}')
    fig.savefig(save_dir + file_name, dpi=300)
    
for f in files:
    r = Radial(f)
    plot_pf(r)
