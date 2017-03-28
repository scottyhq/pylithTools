#!/usr/bin/env python
"""
Matplotlib plots of pylith h5 output

Usage:  import pylith_plotter as pp
        pp.plot_surface(surfaceFile)
"""
import numpy as np
import os

import scipy as sp
import itertools

#import osgeo.gdal as gdal
#import osgeo.osr as osr
#import osgeo.gdalconst as gdalconst
import gdal

import matplotlib.pyplot as plt
import mpl_toolkits.basemap as bm
from matplotlib.mlab import griddata, rms_flat
from matplotlib.transforms import Bbox, BboxTransform
from matplotlib.colors import Normalize, ListedColormap
from mpl_toolkits.axes_grid1 import ImageGrid #replace with subplots()?
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize

# Currently only works with python2
#from mayavi import mlab # for tomography animation

import roipy as rp #for comparing to los geo_unw files (could just copy relevant stuff..)
import roipy.models as m
#import pylith_utils as pu
import pylithTools.util as pu


#datadir = os.environ['DATA']
datadir = '/Volumes/OptiHDD/data/'


def plotMesh(verts,tris):
    """
    Plot mesh triangles on a given surface

    NOTE: triangles must consistently be labelled clockwise or anticlockwise...
    doesn't seem to be the case for pylith output...
    """
    x = verts[:,0]
    y = verts[:,1]

    plt.figure()
    plt.gca().set_aspect('equal')
    plt.triplot(x, y, tris, 'k-')
    plt.title('Unstructured Mesh')
    plt.xlabel('distance (m)')
    plt.ylabel('distance (m)')


def plotScatter(verts, data, coords=(1,2), comp=2):
    """
    Scatter plot of displacements at nodes
    coords = 0=x,1=y,2=z
    """
    z = data[:,:,comp].flatten()
    x = verts[:,coords[0]]
    y = verts[:,coords[1]]

    # NOTE: either scatter or pcolor should work
    plt.figure()
    compDict = {0:'X',1:'Y',2:'Z'}
    #plt.gca().set_aspect('equal')
    plt.scatter(x, y, c=z, s=80, cmap=plt.cm.bwr)
    plt.title( compDict[comp] + ' Displacement' )
    plt.xlabel(compDict[coords[0]] + ' Distance [m]')
    plt.ylabel(compDict[coords[1]] + ' Distance [m]')
    cb = plt.colorbar()
    cb.set_label('[m]')


def plotComponents(verts,data,type='scatter'):
	''' show all three components together
	type=scatter or type=contour
	'''
	x = verts[:,0]
	y = verts[:,1]
	labels = {0:'X',1:'Y',2:'Z'}
	ms = 100
	fig, grid = plt.subplots(1,3, subplot_kw=dict(aspect=1.0, adjustable='box-forced'),
	sharex=True, sharey=True, figsize=(17,8.5))

	if type == 'scatter':
		for i,ax in enumerate(grid.flat):
			sc = ax.scatter(x,y, c=data[:,:,i], s=ms, cmap=plt.cm.bwr) #norm=MidpointNormalize(midpoint=0),
			ax.set_title(labels[i] + ' Displacement')
			ax.set_xlabel('X [km]')
			ax.set_ylabel('Y [km]')
			cb = plt.colorbar(sc,ax=ax, orientation='horizontal', pad=0.1, ticks=MaxNLocator(nbins=5)) #5 ticks only)
			cb.set_label('m')

	if type == 'contour':
		# ensure rectangular grid:
		xi = np.linspace(x.min(), x.max(), x.size)
		yi = np.linspace(y.min(), y.max(), y.size)

	for i,ax in enumerate(grid.flat):
		z = data[:,:,i].flatten()
		print(x.shape, y.shape, z.shape)
		zi = griddata(x,y,z,xi,yi, interp='nn')

		cs = ax.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
		csf = ax.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
		ax.set_title( labels[i] + ' Displacement')
		ax.set_xlabel('Distance [m]')
		ax.set_ylabel('Distance [m]')
		cb = plt.colorbar(csf,ax=ax, orientation='horizontal', pad=0.1, ticks=MaxNLocator(nbins=5))
		cb.set_label('[m]')
	#plt.show() #not needed in ipython


def plotContour(verts, data, comp=2):
    """
    Contours of surface deformation
    # Also see matplotlib tricontour and tricontourf functions!
    """
    z = data[:,:,comp].flatten()
    x = verts[:,0]
    y = verts[:,1]

    xi = np.linspace(x.min(), x.max(), x.size)
    yi = np.linspace(y.min(), y.max(), y.size)
    zi = griddata(x,y,z, xi,yi, interp='nn') #'nn' #NOTE: for irregularly spaced data

    plt.figure()
    #plt.gca().set_aspect('equal')
    compDict = {0:'X',1:'Y',2:'Z'}
    C = compDict[comp]

    CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
    CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
    plt.title( C + ' Displacement')
    plt.xlabel('Distance [m]')
    plt.ylabel('Distance [m]')
    cb = plt.colorbar()
    cb.set_label('[m]')



def plotSurface(surfaceFile, comp=2, points=False, tris=False,
                profile=False, ax=None, annotate=True, norm=None,xscale=1, yscale=1):
    """
    Plot interpolated image of surface displacements, optionally show original points
    """
    verts,data,tris = load_h5(surfaceFile)

    if comp==3: #radial displacements
        z = np.hypot(data[:,:,0], data[:,:,1]).flatten()
    else:
        z = data[:,:,comp].flatten()
    #z = data[:,:,comp].flatten()
    x = verts[:,0] / xscale
    y = verts[:,1] / yscale

    #NOTE: need to change grid for linear spacing to work properly
    xi = np.linspace(x.min(), x.max(), x.size)
    yi = np.linspace(y.min(), y.max(), y.size)
    zi = griddata(x,y,z, xi,yi, interp='nn') #'nn'

    #NOTE: getting error message here...
    # linear interpolation requires exactly the same limits
    #xi=np.arange(-15000.0,15000.0+1e-14,30000.0/x.size)
    #yi=np.arange(-15000.0,15000.0+1e-14,30000.0/x.size)
    #zi = griddata(x,y,z, xi,yi, interp='linear') #'nn'
    #ValueError: output grid must have constant spacing when using interp='linear'

    if ax==None:
        plt.figure()
    else:
        ax = plt.axes(ax)

    #plt.pcolor(xi, yi, zi, cmap=plt.cm.jet) #Very slow...
    x1, x2, y1, y2 = [x.min(), x.max(), y.min(), y.max()]
    im = plt.imshow(zi, cmap=plt.cm.jet, norm=norm, extent=[x1, x2, y1, y2])

    if annotate:
        compdict = {0:'Ux',1:'Uy',2:'Uz',3:'Ur'}
        plt.title('{} Displacement'.format(compdict[comp]))
        plt.xlabel('Distance [m]')
        plt.ylabel('Distance [m]')
        cb = plt.colorbar()
        cb.set_label('[m]')

    if points:
        plt.plot(x,y,'k.')

    if type(tris) is np.ndarray:
        plt.triplot(x, y, tris, 'k-')

    # EW profile line through the x-axis
    if profile:
        plt.axhline(linewidth=2, color='r')
        Zi = zi[x.size/2,:]
        plt.figure()
        plt.plot(xi, Zi, 'b.-')
        plt.title('Profile')
        plt.xlabel('Distance [m]')
        plt.ylabel('{} Displacement [m]'.format(compdict[comp]))

    return im


def imshow2array(imx,imy):
    """
    Convert axes coordinates in figure to image indicies

    #eg. -15000, 0 in figure corresponds to # print tr.transform_point((-15000, 0)) # zi[0,0]
    """
    # NOTE: would be more efficient to not reconstruct 'tr' transform object every time...
    # NOTE: different order comp to 'extent' keyword in imshow
    bbox_in = Bbox.from_extents([x1, y1, x2, y2])
    # NOTE: lower left corner always -0.5,-0.5 by deafult with imshow
    bbox_out = Bbox.from_bounds(-0.5, -0.5, zi.shape[1], zi.shape[0])
    # transform from data coordinate into image coordinate.
    tr = BboxTransform(bbox_in, bbox_out)
    arrXY = tr.transform_point((imx, imy)).astype(int)

    return arrXY



def plot_maxdisp_time(pointsh5, xscale=1e3, yscale=1e-2, tscale=3.1536e7,
                      adjustRadial=False):
    """
    For time-dependent runs plot maximum displacements versus time
    """
    coords,data,number,times = pu.load_h5_visco(pointsh5)
    x = coords[:,0]
    ur = np.hypot(data[:,:,0], data[:,:,1])
    uz = data[:,:,2]

    # Convert units & extract maximums for each timestep
    x = x / xscale
    ur = np.max(ur,1) / yscale
    uz = np.max(uz,1) / yscale #cm
    #times = times / 8.64e4 #days
    #times = times / 31536000 #years
    times = times / tscale

    plt.figure()
    line, = plt.plot(times, uz, 'b.-', lw=2, label='Uz')
    plt.plot(times, ur, ls='dashed', lw=2, marker='.', color=line.get_color(), label='Ur')
    plt.title('Maximum displacements')
    plt.ylabel('Displacement [{}]'.format(get_unit(yscale)))
    plt.xlabel('Time [{}]'.format(get_unit(tscale)))
    plt.show()
    plt.legend(loc='best')
    plt.grid()



def plot_visco_profiles(pointsh5, skip=slice(None,None,1), xscale=1e3, yscale=1e-2, tscale=3.1536e7, adjustRadial=False, benchmark=[], title=None):
	"""
	Profiles of surface displacement at each timestep

	NOTE: also would be interesting to plot uz_ur ratio versus time
	"""
	plt.figure()

	coords,data,number,times = pu.load_h5_visco(pointsh5)

	#x = 1e3*np.loadtxt(points,usecols=[0]) # output_points2.txt
	#y = np.zeros_like(x)
	x = coords[:,0]
	y = np.zeros_like(x)

	# NOTE: plot elastic solution by passing dictionary as showelastic
	# Plot analytic elastic solution (t=0)
	#print(benchmark)
	if len(benchmark)>=1:
		ur = zeros_like(x)
		uz = np.zeros_like(x)
		for b in benchmark:
			uri,uzi = m.calc_mogi_dp(x,y,**params)
			ur += uri
			uz += uzi
		plt.plot(x*xscale,uz*yscale,'ko',label='benchmark')

	# Convert units
	#ur = np.hypot(data[:,:,0], data[:,:,1]) #assume progiles are along EW profile
	ur = data[:,:,0]
	uz = data[:,:,2]
	x = x / xscale
	ur = ur / yscale #cm
	uz = uz / yscale #cm
	times = times / tscale
	#times = times / 8.64e4 #days
	#times = times / 31536000 #years

	#plots = np.arange(0,times.size,skip)
	#print(plots.size)
	#way to cycle through markers if plotting many lines
	#marker = itertools.cycle(['o','^','s','D']) #plot(marker=marker.next() iterates list)
	#way to use gradually changing colors from a colormap
	#color = plt.cm.jet(1.0*i/plots.size)
	indplots = np.arange(times.size-1)
	print(indplots)
	indplots = indplots[skip]
	print(indplots)
	for i in indplots:
		line, = plt.plot(x, uz[i], color=plt.cm.jet(1.0*i/indplots[-1]), label='{:.1f}'.format(times[i]))
		plt.plot(x, ur[i], ls='dashed', color=line.get_color())
	#print uz[i]
	#print uz[i-1]

	if title:
		plt.title(title)
	else:
		plt.title(pointsh5)

	plt.axhline(color='k',linestyle='dashed')
	plt.xlabel('Distance [{}]'.format(get_unit(xscale)))
	plt.ylabel('Displacement [{}]'.format(get_unit(yscale)))
	plt.show()
	plt.legend(title='{}'.format(get_unit(tscale)))
	plt.grid()



def plot_profile(outdir, xval='x', xscale=1, yscale=1, comp2los=False, adjustRadial=False,
                 fig=True):
    """
    Plot vertical and radial surface displacement profile
    if x='r', calculate radial distance from x & y
    """
    #Load data
    path = os.path.join(outdir,'points.h5')
    x,y,z,ux,uy,uz = pu.extract_points(path)

    Y = uz / yscale
    if xval == 'x':
        X = x / xscale
        Y1 = ux / yscale
    elif xval == 'r':
        X = np.hypot(x,y) / xscale
        ur = np.hypot(ux,uy)
        Y1 = ur / yscale
        if adjustRadial: #fix sign from hypot square root
            ur = pu.radial2negative(Y1)

    if fig:
        plt.figure()
    # otherwise profile added to active plot

    #plt.plot(X,uy/yscale,'r.-',label='Uy') #should be zero along EW axis
    de = 90e3 / xscale #eastern data extent
    if comp2los != False:
        data_extents = (X<=de)
        if comp2los == 'west': #switch sign of radial profile
            #ux = -ux #move to comp2los function
            X = -X
            Y1 = -Y1
            de = -de
            data_extents = (X>=de)

        los = pu.comp2los(x,ux,uy,uz,track=comp2los)
        plt.plot(X, los/yscale, 'k-', lw=2, label='Ulos_' + comp2los)
        plt.fill_between(X,los/yscale, where=data_extents, color='gray',alpha=0.5)

    plt.plot(X, Y, 'b-', lw=2, label='Uz')
    plt.plot(X, Y1, 'b--',lw=2, mfc='None',label='U{0}'.format(xval))

    # Annotate
    plt.title(outdir)
    plt.xlabel('Distance [{}]'.format(get_unit(xscale)))
    plt.ylabel('Uz [{}]'.format(get_unit(yscale)))
    plt.axhline(color='k')
    plt.axvline(de,color='k', linestyle='dashed', label='EW data extent') #EW extent of InSAR coverage
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()



def comp2profile(output):
    """
    Compare FEM output to vertical and radial displacements from inversion of
    multiple LOS InSAR datasets
    """

    # Load components
    data = '/home/scott/data/insar/cvz/los2xyz/components.txt_backup'
    lon,lat,x,ur,uz = np.loadtxt(data,unpack=True)

    # Load FEM output
    points = '/home/scott/research/models/pylith/3d/fialko2012/model3_agu/output_points.txt'
    pointsh5 = '/home/scott/research/models/pylith/3d/fialko2012/model3_agu/output/elastic/apmb/points.h5'
    x_fem, ur_fem, uz_fem = pu.extract_points(pointsh5)
    #NOTE: in this case need radial distance b/c not on x-axis
    #x = np.sqrt(2)*1e3*np.loadtxt(points,usecols=[0])
    # NOTE: cludge fix for negative radial displacements in dipole model
    ur_fem[x_fem >= 40000] = -ur_fem[x_fem >= 40000]
    #ur_fem = np.abs(ur_fem)

    #Convert to cm units for plotting
    x_fem = x_fem/1000
    ur_fem = ur_fem * 100.0
    uz_fem = uz_fem * 100.0

    #NOTE: must interpolate either insar data or FEM to compare at values at same radial distance
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(x, uz, 'b.-', lw=2, label='data')
    plt.plot(x, ur, 'g.-', lw=2)
    plt.plot(x_fem, uz_fem, marker='o', ls='None', lw=2, mec='b', mfc='None',label='FEM')
    plt.plot(x_fem, ur_fem, marker='o', ls='None', lw=2, mec='g', mfc='None')
    plt.axhline(color='k')
    plt.title('FEM vs. Data')
    plt.xlabel('Radial distance [km]')
    plt.ylabel('Displacement [cm]')
    plt.grid(True)
    plt.xlim(0,150) #only show out to 150km
    plt.legend()


def data_model_residual(surface, dem, unw, incidence):
    """
    Convert Pylith output to InSAR LOS and plot data, model in map view
    """
    los,fem_los,residual = pu.los2pylith(surface,dem,unw,incidence)

    # Using image_grid
    fig = plt.figure()
    grid = ImageGrid(fig, 111, # similar to subplot(111)
                    nrows_ncols = (1, 3),
                    direction="row",
                    axes_pad = 0.05,
                    add_all=True,
                    label_mode = "1",
                    share_all = True,
                    cbar_location="top",
                    cbar_mode="each", #"single"
                    cbar_size="5%",
                    cbar_pad=0.05,
                    )
    #grid[0].set_xlabel("X")
    #grid[0].set_ylabel("Y")
    #grid2[0].set_xticks([-2, 0])
    #grid2[0].set_yticks([-2, 0, 2])

    #NOTE: could find global min/max from three arrays here
    norm = Normalize(vmin=np.nanmin(los), vmax=np.nanmax(los))
    #for ax,data in zip(grid,[los,fem_los,residual]):
    im = grid[0].imshow(los,origin='upper',norm=norm,cmap=plt.cm.jet)
    grid[0].axhline(100,color='m') #show profile
    cax = grid.cbar_axes[0]
    cax.colorbar(im)
    grid[1].axhline(100,color='k') #show profile
    im1 = grid[1].imshow(fem_los,origin='upper',norm=norm,cmap=plt.cm.jet)

    cax = grid.cbar_axes[1]
    cax.colorbar(im1)

    im2 = grid[2].imshow(residual,origin='upper',cmap=plt.cm.jet)
    cax = grid.cbar_axes[2]
    cax.colorbar(im2)

    # Add letter labels
    for ax, label in zip(grid,['A', 'B', 'C']):
        ax.text(0.05, 0.95, label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top')

    # Annotate
    # NOTE: way too high!
    #plt.suptitle('FEM Results')

    # Add profile
    # NOTE: for now EW, but would be easy to do arbitrary line, and convert to km
    fig = plt.figure()
    #x = arange(los.shape[0])
    plt.axhline(color='k',ls='--')
    plt.plot(los[100],'m.',label='data')
    plt.plot(fem_los[100],'k-',lw=2,label='model')
    plt.xlabel('Distance [km]')
    plt.ylabel('Distance [km]')
    plt.legend(loc='upper left')

    plt.show()


def plot_timeDB(timeDB, xunits='yr', yunits='MPa', skip=8, P0=33.0):
    """
    Defaults set-up for pressurization history
    """
    time, pressure = np.loadtxt(timeDB, skiprows=skip, unpack=True)
    pressure = pressure * P0

    #if xunits == 'yr':
    #    time = time / 31536000.0
    #elif xunits == 'day':
    #    time = time / 86400.0

    plt.figure()
    plt.plot(time,pressure,'b.-',lw=3,label='pressure')
    plt.xlabel('Time [{}]'.format(xunits))
    plt.ylabel('Pressure [{}]'.format(yunits))
    plt.title('Time History')
    plt.show()




def plot_directory_numex(path, vals, param='density', outname=None, show=True,
                         xscale=1e-3,yscale=1e2):
    """
    plot results for numerical experiment folder
    1) surface profiles for each parameter
    2) uz_max versus parameter values
    3) residual versus parameter values
    """
    #vals = arange(2300.0, 2800.0, 50.0)
    outdirs = np.sort(os.listdir(path))
    plt.figure()

    # Plot surface profiles for each parameter
    for val,outdir in zip(vals,outdirs):
        pointsFile = os.path.join(path, outdir, 'points.h5')
        print(pointsFile)
        x_fem, ur_fem, uz_fem = pu.extract_points(pointsFile, output=True, adjustRadial=True)
        x_fem = x_fem / xscale
        ur_fem = ur_fem / yscale
        uz_fem = uz_fem / yscale
        l, = plt.plot(x_fem,uz_fem,'.-',label=str(val))
        plt.plot(x_fem,ur_fem,'.-',color=l.get_color())

    # Annotate
    plt.axhline(color='k') #zero displacement line
    plt.title(param)
    plt.xlabel('Distance [{}]'.format(get_unit(xscale)))
    plt.ylabel('Displacement [{}]'.format(get_unit(yscale)))
    plt.legend()

    if outname: plt.savefig(outname)
    if show: plt.show()


def plot_directory_surface_rmse(path, vals, param='Vp'):
    """
    plot RMSE versus parameter value for numerical experiment
    RMSE is calculated based on full fem output converted to LOS
    """

    unw = '/home/scott/data/insar/cvz/t2282/geo/geo_stack282_8rlks.unw'
    incidence = '/home/scott/data/insar/cvz/t2282/aux_files/geo_incidence_8rlks.unw'
    dem = '/home/scott/data/dems/cgiar/uturuncu_1000_1000.tif'
    #surface = '/home/scott/research/models/pylith/3d/uturuncu_layered/output/step01/surface.h5'
    #geosurface = '/home/scott/research/models/pylith/scripts/geo_fem_Uz.tif'

    plt.figure()
    outdirs = np.sort(os.listdir(path))

    # Calculate rmse
    rmse = np.zeros_like(vals)
    for i,outdir in enumerate(outdirs):
        surface = os.path.join(path,outdir,'surface.h5')
        los,fem_los,residual = pu.los2pylith(surface, dem, unw, incidence)
        rmse[i] = rms_flat(residual[np.isfinite(residual)])

    plt.plot(vals,rmse,'b.-')

    # Annotate
    #plt.title(param)
    plt.xlabel('{} [km]'.format(param))
    plt.ylabel('RMSE [cm]')
    plt.legend()
    plt.show()


def get_unit(scale):
    """
    Convert scale term to unit label
    """
    scale2unit = { 1e-9: 'nm',
                   1e-6: u'\N{MICRO SIGN}m', #or hex id (lookup): u'\u00B5'
                   1e-3: 'mm',
                   0.01: 'cm',
                   0.1:'dm',
                   1:'m',
                   1000:'km',
                   # time
                   8.6400e4:'day',
                   3.1536e7:'yr',
                   3.1536e10:'ka',
                   3.1536e13:'Ma',
                   #Pressure
                   1e9: 'GPa',
                   1e6: 'MPa',
                   }
    return scale2unit[scale]



def compare_ratios(path='/Volumes/OptiHDD/data/pylith/3d/agu2013/output',
				steps=['step01','step02'],
				#labels='',
				show=True,
				xscale=1e3,
				yscale=1e-2):
	"""
	Plot vertical and radial profiles for specified output directories.
	NOTE: normalization value is hardcoded...
	"""
	plt.figure()
	#path = '/Users/scott/Desktop/elastic'

	# Deep source
	labels = ['no APMB', 'APMB']
	deep = {}
	uzmax = 0.824873455364
	# NOT sure why hardcoded...
	uzmax = 1
	for i,outdir in enumerate(steps):
		pointsFile = os.path.join(path, outdir, 'points.h5')

		x,y,z,ux,uy,uz = pu.extract_points(pointsFile)

		X = x / xscale
		Y1 = ux / yscale

		x_fem = X #/ xscale #double scaling!
		ur_fem = Y1 #/ yscale
		uz_fem = uz / yscale

		#print(pointsFile)
		print(ur_fem.min(), ur_fem.max(), uz_fem.min(), uz_fem.max(), uz_fem.max() / ur_fem.max())

		#normalize
		uz_fem = uz_fem / uzmax
		ur_fem = ur_fem / uzmax
		x_fem = x_fem / 30.0

		l, = plt.plot(x_fem,uz_fem,'o-',ms=4,lw=4,label=labels[i])
		plt.plot(x_fem,ur_fem,'o--',ms=4,lw=4,color=l.get_color()) #mfc='none' transparent
		deep[outdir] = uz_fem/uz_fem


	# Shallow Source
	shallow = {}
	uzmax = 0.949652827795 # Why?
	for i,outdir in enumerate(['step11','step12']):
		pointsFile = os.path.join(path, outdir, 'points.h5')

		x,y,z,ux,uy,uz = pu.extract_points(pointsFile)

		X = x / xscale
		Y1 = ux / yscale

		x_fem = X #/ xscale #double scaling!
		ur_fem = Y1 #/ yscale
		uz_fem = uz / yscale

		#print(pointsFile)
		print(ur_fem.min(), ur_fem.max(), uz_fem.min(), uz_fem.max(), uz_fem.max() / ur_fem.max())

		#normalize
		uz_fem = uz_fem / uzmax
		ur_fem = ur_fem / uzmax
		x_fem = x_fem / 20.0

		l, = plt.plot(x_fem,uz_fem,'.-', mfc='w', lw=4,label=labels[i])
		plt.plot(x_fem,ur_fem,'.--',lw=4, mfc='w',color=l.get_color()) #mfc='none' transparent

		shallow[outdir] = uz_fem/ur_fem

	# Annotate
	plt.axhline(color='k',lw=0.5)
	#plt.xlabel('Distance [{}]'.format(get_unit(xscale)))
	#plt.ylabel('Displacement [{}]'.format(get_unit(yscale)))
	plt.legend()
	plt.grid()
	#plt.ylim(-0.5, 3.5)
	#plt.savefig('deep.png',bbox_inches='tight')
	#plt.savefig('shallow.png',bbox_inches='tight')

	# normalized
	plt.ylim(-0.5, 4)
	plt.xlim(0,10)
	plt.xlabel('Normalized Radial Distance  [R / D]')
	plt.ylabel('Normalized Displacement  [U / Uz_max]')
	#plt.savefig('normalized_deep.png',bbox_inches='tight')
	plt.savefig('normalized_shallow.png',bbox_inches='tight')


	# Plot ratios of uz versus NOTE: this plot is confusing,,, just keep ratio of uz_max to ur_max
	'''
	plt.figure()
	plt.plot(x_fem, deep['step01'], label='Deep no APMB')
	plt.plot(x_fem, deep['step02'], label='Deep w/ APMB')
	plt.plot(x_fem, shallow['step11'], label='Shallow no APMB')
	plt.plot(x_fem, shallow['step12'], label='Shallow w/ APMB')
	plt.xlabel('Distance [km]') #NOTE: maybe plot normailzed X-axis (R-d)
	#plt.xlabel('Normalized Distance [R/d]')
	plt.ylabel('Ratio [Uz/Ur]')
	plt.title('Ratio of vertical to radial displacement')
	plt.legend()
	plt.show()
	'''


def plot_ratios(path='/Volumes/OptiHDD/data/pylith/3d/agu2014/output',
				steps=['step01','step02'],
				#labels='',
				show=True,
				xscale=1e3,
				yscale=1e-2):
	"""
	Plot vertical and radial profiles for specified output directories.
	NOTE: normalization value is hardcoded...
	"""
	plt.figure()
	#path = '/Users/scott/Desktop/elastic'

	# Deep source
	#labels = ['no APMB', 'APMB']
	#if labels == '':
	labels = steps
	deep = {}
	#uzmax = 0.824873455364
	# NOT sure why hardcoded...
	uzmax = 1
	for i,outdir in enumerate(steps):
		pointsFile = os.path.join(path, outdir, 'points.h5')
		print(pointsFile)
		x,y,z,ux,uy,uz = pu.extract_points(pointsFile)

		X = x / xscale
		Y1 = ux / yscale

		x_fem = X #/ xscale #double scaling!
		ur_fem = Y1 #/ yscale
		uz_fem = uz / yscale

		#print(pointsFile)
		print(ur_fem.min(), ur_fem.max(), uz_fem.min(), uz_fem.max(), uz_fem.max() / ur_fem.max())

		#normalize
		uz_fem = uz_fem / uzmax
		ur_fem = ur_fem / uzmax
		x_fem = x_fem / 30.0

		l, = plt.plot(x_fem,uz_fem,'o-',ms=4,lw=4,label=labels[i])
		plt.plot(x_fem,ur_fem,'o--',ms=4,lw=4,color=l.get_color()) #mfc='none' transparent
		deep[outdir] = uz_fem/uz_fem

	'''
	# Shallow Source
	shallow = {}
	uzmax = 0.949652827795
	for i,outdir in enumerate(['step11','step12']):
		pointsFile = os.path.join(path, outdir, 'points.h5')

		x,y,z,ux,uy,uz = pu.extract_points(pointsFile)

		X = x / xscale
		Y1 = ux / yscale

		x_fem = X #/ xscale #double scaling!
		ur_fem = Y1 #/ yscale
		uz_fem = uz / yscale

		#print(pointsFile)
		print(ur_fem.min(), ur_fem.max(), uz_fem.min(), uz_fem.max(), uz_fem.max() / ur_fem.max())

	#normalize
	uz_fem = uz_fem / uzmax
	ur_fem = ur_fem / uzmax
	x_fem = x_fem / 20.0

		l, = plt.plot(x_fem,uz_fem,'.-', mfc='w', lw=4,label=labels[i])
		plt.plot(x_fem,ur_fem,'.--',lw=4, mfc='w',color=l.get_color()) #mfc='none' transparent

		shallow[outdir] = uz_fem/ur_fem
	'''

	# Annotate
	plt.axhline(color='k',lw=0.5)
	#plt.xlabel('Distance [{}]'.format(get_unit(xscale)))
	#plt.ylabel('Displacement [{}]'.format(get_unit(yscale)))
	plt.legend()
	plt.grid()
	#plt.ylim(-0.5, 3.5)
	#plt.savefig('deep.png',bbox_inches='tight')
	#plt.savefig('shallow.png',bbox_inches='tight')

	# normalized
	plt.ylim(-0.5, 4)
	plt.xlim(0,10)
	plt.xlabel('Normalized Radial Distance  [R / D]')
	plt.ylabel('Normalized Displacement  [U / Uz_max]')
	#plt.savefig('normalized_deep.png',bbox_inches='tight')
	plt.savefig('normalized_shallow.png',bbox_inches='tight')


	# Plot ratios of uz versus NOTE: this plot is confusing,,, just keep ratio of uz_max to ur_max
	'''
	plt.figure()
	plt.plot(x_fem, deep['step01'], label='Deep no APMB')
	plt.plot(x_fem, deep['step02'], label='Deep w/ APMB')
	plt.plot(x_fem, shallow['step11'], label='Shallow no APMB')
	plt.plot(x_fem, shallow['step12'], label='Shallow w/ APMB')
	plt.xlabel('Distance [km]') #NOTE: maybe plot normailzed X-axis (R-d)
	#plt.xlabel('Normalized Distance [R/d]')
	plt.ylabel('Ratio [Uz/Ur]')
	plt.title('Ratio of vertical to radial displacement')
	plt.legend()
	plt.show()
	'''



def plot_directory_profiles(path, outname=None, show=True, xscale=1, yscale=1,
                            xval='x', adjustRadial=True):
    """
    Plot profiles for each output/step0X folder on same figure
    """
    outdirs = np.sort(os.listdir(path))
    plt.figure()

    #labels=['homogeneous','1D layering', '3D tomography'] #xscale=1e-3, yscale=1e2
    for i,outdir in enumerate(outdirs):
        pointsFile = os.path.join(path, outdir, 'points.h5')
        #print(pointsFile)
        #x_fem, ur_fem, uz_fem = pu.extract_points(pointsFile, output='cyl',adjustRadial=adjustRadial)
        #x_fem, ur_fem, uz_fem = pu.extract_points(pointsFile)
            #Load data

        x,y,z,ux,uy,uz = pu.extract_points(pointsFile)
        #Y = uz / yscale
        if xval == 'x':
            X = x / xscale
            Y1 = ux / yscale
        elif xval == 'r':
            X = np.hypot(x,y) / xscale
            ur_fem = np.hypot(ux,uy)
            Y1 = ur_fem / yscale
            if adjustRadial: #fix sign from hypot square root
                ur_fem = pu.radial2negative(Y1)

        x_fem = X #/ xscale #double scaling!
        ur_fem = Y1 #/ yscale
        uz_fem = uz / yscale

        #print(pointsFile)
        print(ur_fem.min(), ur_fem.max(), uz_fem.min(), uz_fem.max())

        l, = plt.plot(x_fem,uz_fem,'.-',lw=3,label=outdir)
        #l, = plt.plot(x_fem,uz_fem,'.-',lw=2,label=labels[i]) #for 3d heterogeneity example
        plt.plot(x_fem,ur_fem,'.--',lw=3, mfc='w',color=l.get_color()) #mfc='none' transparent

    # Annotate
    plt.axhline(color='k',lw=0.5)
    plt.xlabel('Distance [{}]'.format(get_unit(xscale)))
    plt.ylabel('Displacement [{}]'.format(get_unit(yscale)))
    plt.legend()

    #NOTE: custom annotations for 3d heterogeneity
    #plt.title('Elastic Heterogeneity Effects')
    #plt.legend([l1,l2,l3],['homogeneous','1D layering', '3D tomography'])

    if outname: plt.savefig(outname)
    if show: plt.show()


def plot_directory_surface(path,figsize=(17,11), comp=2, nrow=1, norm=None,
    cbar='each', cloc='top', outname=None, labels='1', show=True):
    """
    Plot grid of surface displacement maps from each output/step* folder
    if normalize=True, use step01 colorbar for all images
    """
    outdirs = np.sort(os.listdir(path))
    nplots = len(outdirs)
    ncol = np.ceil(nplots/nrow).astype(np.int)
    fig = plt.figure(figsize=figsize)

    grid = ImageGrid(fig, 111, # similar to subplot(111)
                    nrows_ncols = (nrow, ncol),
                    direction="row",
                    axes_pad = 0.25,
                    add_all=True,
                    label_mode = labels, #'all', 'L', '1'
                    share_all = True,
                    cbar_location=cloc, #top,right
                    cbar_mode=cbar, #each,single,None
                    cbar_size=0.1,#"7%",
                    cbar_pad=0.0#,"1%",
                    )

    #NOTE: if cbar='single',cloc='right', a way to automatically normalize by grid[0]
    #if normalize:
    #    verts,data,tris = load_h5(os.path.join(path,'step01/surface.h5'))
    #    if comp==3: #radial displacements
    #        z = np.hypot(data[:,:,0], data[:,:,1]).flatten()
    #    else:
    #        z = data[:,:,comp].flatten()
    #    norm = Normalize(vmin=np.nanmin(z), vmax=np.nanmax(z))
    #else:
    #    norm=None

    for i,outdir in enumerate(outdirs):
        ax = grid[i]
        print(outdir)
        im = plotSurface(os.path.join(path,outdir,'surface.h5'), comp=comp, ax=ax,
                    points=False, tris=False, profile=False, annotate=False, norm=norm,
                    xscale=1e-3, yscale=1e-3)

        # colorbar settings, not sure what's up with set_xticks...
        ax.cax.colorbar(im)
        #cmin = np.nanmin(data)
        #cmax = np.nanmax(data)
        #ax.cax.set_xticks([cmin,0,cmax])

        # label upper left
        ax.text(0.05,0.95,outdir,
                weight='bold',
                ha='left',
                va='top',
                bbox=dict(facecolor='white'),
                transform=ax.transAxes)
        #ax.set_ylabel(outdir)
        #ax.tick_params(labelbottom=0,labeltop=0,labelleft=0,labelright=0,
        #                bottom=0,top=0,left=0,right=0)

    #if cbar=='single':
    #    grid[0].cax.colorbar(im)

    # Annotate Plot
    # don't show grid frames without data...
    Nextra = grid.ngrids - nplots
    if Nextra > 0:
        for ax in grid[-Nextra:]:
            #print(ax)
            ax.set_visible(False)
            ax.cax.set_visible(False)
    fig.suptitle(path, fontsize=14, fontweight='bold')
    if outname: plt.savefig(outname)
    if show: plt.show()



def ind2latlon(index, filePath):
    """
    Use gdal/osr to get latlon point location from georeferenced array indices
    """
    # Load georeferencing
    ds = gdal.Open(filePath)
    proj = ds.GetProjection()
    gt = ds.GetGeoTransform()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)

    x0 = gt[0] #top left longitude
    y0 = gt[3] #top left latitude
    dx = gt[1] #pixel width
    dy = gt[5] #pixel height

    # Convert row,col of array to projected coords
    row, col = index
    x = x0 + (col * dx)
    y = y0 + (row * dy)

    # Convert projected coords to latlon
    trs = osr.SpatialReference()
    trs.ImportFromEPSG(4326)
    ct = osr.CoordinateTransformation(srs, trs)

    (lon, lat, height) = ct.TransformPoint(x, y) #note could add elevation
    #gdal.DecToDMS(lat, 'Lat', 2)

    return lon, lat



def uturuncu_map(surfaceFile,dem,comp=2):
    """
    Plot transparent fem result on high-res srt hillshade with summit
    and stations, etc
    comp=0,1,2,3 --> ux, uy, uz, ur
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #print(datadir)
    #print(dem)
    geosurface = pu.surface2geotiff(dem,surfaceFile,outname=None,comp=comp,nanval=-9999)

    #load georeferenced fem output from pu.surface2geotiff
    #geosurface = '/home/scott/research/models/pylith/scripts/geo_fem_Uz.tif'
    data,geotrans,proj = pu.load_gdal(geosurface)
    data = data*100 # N-up, units=m
    nLat, nLon = data.shape


    #NOTE: are basmap ll and ur corner or center pixel locations??
    bmap = bm.Basemap(projection='tmerc', #NOTE: if changing to 'merc' have to use latlon=True
                   resolution='i',
                   lon_0=-67.18,
                   lat_0=-22.27,
                   width=200000.0,
                   height=200000.0,
                   suppress_ticks=True, #set to true if using drawmeridians
                   ax=ax)

    # Set map background
    #dem = '/home/scott/data/dems/cgiar/uturuncu_1000_1000.tif'
    # full res
    dem = os.path.join(datadir,'dems/cgiar/srtm_23_17.tif')
    bmap.background(style='srtm', file=dem, zscale=1.5)

    # Annotate
    bmap.drawcountries(linewidth=1,color='k')
    bmap.drawcoastlines(linewidth=1,color='k')
    meridians = np.arange(-68,-65,1)
    md = bmap.drawmeridians(meridians, labels=[0,0,0,1])
    parallels = np.arange(-24,-20,1)
    pl = bmap.drawparallels(parallels, labels=[1,0,0,0])

    # Overlay FEM result
    compdict = {0:'Ux',1:'Uy',2:'Uz',3:'Ur'}
    im = bmap.imshow(data, origin='upper', alpha=0.7) #alternatively flipud(data)
    cb = bmap.colorbar(im)
    cb.set_label('{} [cm]'.format(compdict[comp]))

    # Uturunu Summit
    bmap.plot(-67.18, -22.27, 'r^', latlon=True,label='Uturuncu')

    # Location of maximum uplift
    # NOTE: multiple coordinate transforms needed here
    maxval = np.nanmax(data)
    indflat = np.nanargmax(data)
    ind = np.unravel_index(indflat, data.shape) #NOTE row,col --> (y,x)
    lon,lat = ind2latlon(ind, geosurface)
    bmap.plot(lon,lat,'y*',latlon=True,label='Uz_max')
    print('Maximum={} at ({:.2f},{:.2f})\n'.format(maxval, lon, lat))

    # PLUTONS seismometers
    path = os.path.join(datadir,'vector/uturuncu_plutons_seis')
    sm = bmap.readshapefile(path,'seis',drawbounds=False)
    x,y = np.hsplit(np.array(bmap.seis),2)
    bmap.plot(x,y,'wv', mec='k', markersize=10, mew=2, label='3T')

    # Continuous GPS
    path = os.path.join(datadir,'vector/uturuncu_contGPS')
    bmap.readshapefile(path,'cGPS',drawbounds=False)
    x,y = np.hsplit(np.array(bmap.cGPS),2)
    bmap.plot(x,y,'go', mec='k', markersize=10, mew=2, label='cGPS')

    # Scalebar
    length = 50 #km
    # Scale in lower left
    lon = bmap.llcrnrlon + (length/2.0/100) + (bmap.lonmax - bmap.lonmin)*0.05 #pad by 5% of length, also add 1/2 length of scale length
    lat = bmap.llcrnrlat + (bmap.latmax - bmap.latmin)*0.05
    # Upper Right (todo)
    scale = bmap.drawmapscale(lon, lat, bmap.projparams['lon_0'],bmap.projparams['lon_0'],
                  length, #50km
                  barstyle='fancy',
                  #barstyle='simple',
                  fontsize=14)

    # More Annotations
    plt.legend(loc='upper right',numpoints=1)
    plt.title('FEM Model Output')
    #plt.savefig('map_fem.png',bbox_inches='tight')
    plt.show()


def plot_powerlawDB(timeDB, xunits='yr', yunits='MPa', skip=8, P0=33.0):
    """
    Defaults set-up for pressurization history
    """
    workdir = '/home/scott/research/models/pylith/powerlaws/iavcei_diorite'

    # Plot geotherm
    z, T = np.loadtxt('geotherm.txt', usecols=(2,3), unpack=True)
    plt.figure()
    plt.plot(time,pressure,'b.-',lw=3,label='pressure')
    plt.xlabel('Time [{}]'.format(xunits))
    plt.ylabel('Pressure [{}]'.format(yunits))
    plt.title('Time History')
    plt.show()

    # Print lab data used (last line of powerlaw_params.spatialdb)
    # NOTE: can specify these at various depths to have different mineralogies
    with open('powerlaw_params.spatialdb') as f:
        lines = f.readlines()
    vals = lines[-1].split()
    A = float(vals[3]) #flow constant
    R = float(vals[4]) #activation energy
    n = float(vals[5]) #exponent
    Q = float(vals[6]) #activation energy multiplier

    # Get strain rate from powerlaw_gendb.cfg
    with open('powerlaw_gendb.cfg') as f:
        lines = f.readlines()
    line = [l for l in lines if l.startswith('reference_strain_rate')]
    rate = float(line[0].split('=')[-1].rstrip('/s\n'))

    #NOTE: use functino in IAVCEI_plots.py to plot effective strength or viscosity versus depth


def contour_stresses(matFile, infoFile, ax=0, esize=100):
    """
    Plot stress contours by extracting nearest stresses resolved on a point
    for 4 vertices per cell (tetrahedra)
    """
    # NOTE: some bug to work out here
    vertices, cells, moduli, stress, strain = pu.load_h5_material(matFile, infoFile)

    # NOTE: could get list of all elements that have a vertex on a particular surface
    # or get list of all cells that have a centroid within a certain distane of the surface
    centroids = 0.25 * (vertices[cells[:,0]] +
                        vertices[cells[:,1]] +
                        vertices[cells[:,2]] +
                        vertices[cells[:,3]] )

    # get list of cell centroids that are within a certain distance to x-plane
    ind = (np.abs(centroids[:,0]) <= esize) # X=0 plane

    # yz location of centroids
    a = centroids[ind,1] / 1e3 #y-axis points [km]
    b = centroids[ind,2] / 1e3 #z-axis points

    pointStresses = stress[ind] / 1e6 #report in MPa

    sigma_mean = []
    tau_max = []
    for tensor in pointStresses:
        #sm, tm = pu.stress_analysis(tensor)
        sm, tm = pt.util.stress_analysis(tensor)
        sigma_mean.append(sm)
        tau_max.append(tm)
    sigma_mean = np.array(sigma_mean)
    tau_max = np.array(tau_max)

    #z = sigma_mean
    z = tau_max

    # Figure after 7.7 in segall 2010
    # NOTE: set axis is equal?
    #f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True) #not sure about this
    plt.figure()
    plt.gca().set_aspect('equal')

    # contour
    #A,B = np.meshgrid(a,b)
    ai = np.linspace(a.min(), a.max(), a.size)
    bi = np.linspace(b.min(), b.max(), b.size)
    zi = griddata(a,b,z,ai,bi)
    plt.pcolormesh(ai,bi,zi)

    plt.scatter(a,b,c=sigma_mean) #show actual points
    cb = plt.colorbar()
    cb.set_label('MPa')

    plt.xlabel('Y-axis')
    plt.ylabel('Z-axis')
    plt.title('Max Shear Stress Contours on X=0 plane')


def plot_powerlaw_output(timeDB, xunits='yr', yunits='MPa', skip=8, P0=33.0):
    """
    Read from powerlaw database input files
    """




if __name__ == '__main__':
    print('Should be imported as a module\nimport pylith_plotter.py as pp')
