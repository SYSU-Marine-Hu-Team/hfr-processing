
# Created Aug 2024 by Cathleen Qiao

# This file is for plotting totals using cartopy. I wanted to implement this to
# see which plotting method would look nicer. Additionally, cartopy is considered
# the successor to Basemap which is no longer being developed. 

# Note: I also couldn't plot the Florida Keys with Basemap.



# import plotting packages
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from oceans.ocfis import uv2spdir, spdir2uv
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    
fig = plt.figure(figsize=(24, 16))

def plot_cartopy(T, show=True, save=False):

    # initialize figure
    #fig = plt.figure()
    
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
    
    fig.tight_layout(pad=8)
    
    #plt.subplots_adjust(top = 0.9, bottom = 0.1, left = 0.1, right = 0.9)
    
    # get the bounding box limits
    lon_min = T.data.LOND.min() - 0.25
    lon_max = T.data.LOND.max() + 0.25
    lat_min = T.data.LATD.min() - 0.3
    lat_max = T.data.LATD.max() + 0.5
     
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

    
    x = T.data.LOND
    y = T.data.LATD
    
    extent = [lon_min, lon_max, lat_min, lat_max]
    print("extent: ", extent)
    ax.set_extent(extent)
    
    ax.add_feature(LAND, edgecolor=edgecolor, facecolor=landcolor)
    #ax.add_feature(cfeature.OCEAN)
    #ax.add_feature(cfeature.RIVERS)
    #ax.add_feature(cfeature.LAKES)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(state_lines, zorder=11, edgecolor=edgecolor)
    
    # get station coordinates and codes
    siteLon = T.site_source['Lon'].to_numpy()
    siteLat = T.site_source['Lat'].to_numpy()
    siteCode = T.site_source['Name'].tolist()

    
    ax.scatter(siteLon, siteLat, color='red', s=50, transform=ccrs.PlateCarree())
    
    for label, xs, ys in zip(siteCode,siteLon,siteLat):
        ax.text(xs,ys,label,fontsize=16,fontweight='bold')
    
    # Create the velocity component variables

    u = T.data.VELU / 100       # CODAR velocities are in cm/s
    v = T.data.VELV / 100       # CODAR velocities are in cm/s
    vel = abs(T.data.VELO) / 100
    
    angle, speed = uv2spdir(u.squeeze(), v.squeeze())
    u_norm, v_norm = spdir2uv(np.ones_like(speed), angle, deg=True)
    
    #color_clipped = np.clip(speed, 0, 1).squeeze()

    # Make the quiver plot
    q = ax.quiver(x, y, u * 0.75 + u_norm * 0.25, v * 0.75 + v_norm * 0.25, vel, cmap=plt.cm.jet, width=0.001, headwidth=4, headlength=4, headaxislength=4, transform=ccrs.PlateCarree())

    # Add colorbar
    cbar = fig.colorbar(q, fraction=0.028, pad=0.02)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('m/s', fontsize=18)
    
    # Add title
    ax.set_title(T.file_name + ' Total Velocity Field', fontdict={'fontsize': 28, 'fontweight' : 'bold'})
            
    if show:
        plt.show()
    
    save_dir = '../total-plots/'
    photo_name = 'total_' + T.file_name + '.png'
        

    if save:
        #print("SAVE")
        print(save_dir)
        print(photo_name)
        fig.savefig(save_dir + photo_name)
        fig.clear()
    
    return fig
    
    



       


