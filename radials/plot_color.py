
# This file is for saving the color plots of all the raw radial data. 

# Look at hfradarpy repository for more information:
# https://github.com/rucool/hfradarpy


from hfradarpy.radials import Radial, qc_radial_file
import glob
import os
import xarray as xr

site = 'MARA/'

# Path to radial directory
radial_dir = '../radial-data/raw/' + site
save_dir = '../radial-plot/raw/' + site


# Use glob to find radial files (*
# files = sorted(glob.glob(os.path.join(radial_dir, '*.ruv')))
files = sorted(glob.glob(os.path.join(radial_dir, '*')))


def save_radial(r):
    tds = r.to_xarray('gridded', enhance=True).squeeze()
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

    extent = []

    dx = dy = 0.2  # Area around the point of interest.

    extent = [tds.lon.min()-dx, tds.lon.max()+dx, tds.lat.min()-dy, tds.lat.max()+dy]

    # Create a re-usable function for map features that we can pass an axes to.
    def map_features(ax):
        # Axes properties and features
        ax.set_extent(extent)
        ax.add_feature(LAND, edgecolor=edgecolor, facecolor=landcolor)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.RIVERS)
        ax.add_feature(cfeature.LAKES)
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(state_lines, zorder=11, edgecolor=edgecolor)

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
    
    import cmocean
    from matplotlib.colors import TwoSlopeNorm, Normalize

    """
    Displays the direction and magnitude of the radials
    """
    cmap = cmocean.cm.balance
    scale=None
    headwidth=3
    headlength=5
    headaxislength=5
    sub=1
    velocity_min = -40
    velocity_max = 40
    cbar_step = 10
    offset = Normalize(vmin=velocity_min, vmax=velocity_max, clip=True)
    
    # Define arrow colors. Limited by velocity_min and velocity_max
    color_clipped = np.clip(
        tds.velocity.data[::sub],
        velocity_min,
        velocity_max
    ).squeeze()

    ticks = np.append(np.arange(velocity_min, velocity_max, cbar_step), velocity_max)
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Plot title
    plt.title(f'Colorized Raw Radial Plot\n{tds.time.data}')

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
    map_features(ax)

    # generate colorbar
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size='5%', pad=0.05, axes_class=plt.Axes)
    fig.add_axes(cax)

    cb = plt.colorbar(h, cax=cax, ticks=ticks)
    cb.ax.set_yticklabels([f'{s:d}' for s in ticks])
    cb.set_label('cm/s')

    file_name = r.file_name[:-4:] + '_raw_color.png'

    print(f'{file_name}')

    fig.savefig(save_dir + file_name)

for f in files:
    r = Radial(f)
    save_radial(r)
