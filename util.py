"""
Utilities for working scripting pylith and visualizing output
"""
import numpy as np
import h5py
from matplotlib.mlab import griddata

import osgeo.gdal as gdal
import osgeo.osr as osr
import osgeo.gdalconst as gdalconst

import roipy as rp


def load_h5(h5file, field='displacement'):
	""" Load h5 file. use h5dump to see file structure (h5dump points.h5)
	
	nodes, data, elmts = load_h5('./output/step01/points.h5')
	"""
	# NOTE: _info files have stated variables (stress/strain)
	# other material files have (mu, lambda, density, maxwell_time...)
	# use load_h5_material function instead...
	
	# for fault output, field='slip', u'traction_change'
	# FAULT_INFO output changes for forward or inverse model runs:
	# Forward 'dip_dir', u'final_slip', u'normal_dir', u'strike_dir' (see pylith manual for all possibilities)
	# Inverse [u'area', u'impulse_amplitude']
	
	# Same for point output $points.h5 (both w/ 'displacement' name)
	# Forward (1, 49, 3) array
	# Inverse (40, 49, 3) arrays (nfault n_odes, n_outpoints, n_comp), NOTE: time values junk (1 'timestep' per slip impulse)
	print(h5file)
	
	solution = h5py.File(h5file, "r")
	nodes = solution['geometry/vertices'][:]
	data = solution['vertex_fields/' + field][:]
	elmts = solution['topology/cells'][:]
	elmts = elmts.astype(int)
	solution.close()
	
	return nodes, data, elmts



def load_h5_visco(h5file,tstep=-1):
    """ Load h5 file. use h5dump to see file structure
    eg. h5dump. This includes 'time', if tstep=-1 return full arrays """
    
    print(h5file)
    
    solution = h5py.File(h5file, "r")
    nodes = solution['geometry/vertices'][:]
    data = solution['vertex_fields/displacement'][:]
    elmts = solution['topology/cells'][:]
    elmts = elmts.astype(int)
    time = solution['time'][:].flatten() #time in seconds
    solution.close()
    
    # Return solution at specific time step
    if tstep != -1:
        print(tstep)
        data = data[tstep]
        time = time[tstep]
    
    return nodes, data, elmts, time


def extract_surface(vertices, cells, ax=0, elmntsize=100):
    """
    Get stress information for all elements in NS plane
    """
    ind = (np.abs(vertices[:,0]) <= elmtsize) # X=0 plane



def get_closest_cell(vertices,cells,point):
    """
    for example stress resolved at a point=(x,y,z)
    """
    # extract centroid points of all elements
    centroids = 0.25 * (vertices[cells[:,0]] +
                        vertices[cells[:,1]] +
                        vertices[cells[:,2]] +
                        vertices[cells[:,3]] )
    
    # distance from centroids to target point
    dist = np.sqrt((centroids[:,0] - point[0])**2 +
                    (centroids[:,1] - point[1])**2 +
                    (centroids[:,2] - point[2])**2 )
    
    # index of shortest distance
    ind = np.argmin(dist)
    
    # note: get stress tensor at that point with:
    #pointStress = stress[ind]
    #cellRho = density[ind]
    
    # return index and centroid location
    return ind, centroids[ind]


def stress_analysis(stress):
    """
    return meaningful manipulations of stress tensor (mean stress, deviatoric stress, principal)
    NOTE: negative axial stress=compression,  postive=tension
    NOTE: calculations assume 3D stress tensor (stress at a point)
    NOTE: also see about modifying princaxes.py in Pylith playpen
    """
    # 3x3 form of stress matrix:
    sigma = np.array( [ [stress[0], stress[3], stress[5]],
                        [stress[3], stress[1], stress[4]],
                        [stress[5], stress[4], stress[2]] ])
    
    # Isometric (volume change)
    sigma_iso = (1.0/3.0) * np.trace(sigma) * np.eye(3)
    
    # Deviatoric (distortion)
    # Recall differential stress = (sig1 - sig3)
    # But DEVIATORIC stress is a matrix http://www3.geosc.psu.edu/~jte2/references/link100.pdf
    sigma_dev = sigma - sigma_iso 
    
    # Compute principal stresses
    #NOTE: eigvalsh used b/c stress tensor always symmetric
    eigvals = np.linalg.eigvalsh(sigma)
    #NOTE: eignvalues automatically sorted as sig1, sig2, sig3 since negative=compression
    # could multiply by -1 for engineering convention (+ = compression)
    sig1,sig2,sig3 = eigvals
    
    # NOTE: review how to calculate principal stress axes unit vectors
    
    # Stress Invariants
    I1 = np.trace(sigma)
    I2 = sig1*sig2 + sig2*sig3 + sig3*sig1
    I3 = sig1*sig2*sig3
    
    # Mean Stress
    sigma_mean = (1/3.0) * I1
    
    # Max Shear
    # NOTE: usual representation of crustal strength due to mohr-coloumb failure
    # or *differential* stress (sig1 - sig3)
    #tau_max = np.abs(sig1 - sig3) / 2 #assumes sig1 > sig2 > sig3
    tau_max = np.abs(eigvals.max() - eigvals.min()) / 2
    
    # Stress-Deviator Invariantes
    #J1 = 0
    J2 = (1/3.0)*I1**2 - I2
    J3 = (2/27.0)*I1**3 - (1/3.0)*I1*I2 + I3
    #J2 = 0.5 * np.trace(np.dot(sigma_dev, sigma_dev))
    #J3 = (1/3.0) * np.trace( np.dot(sigma_dev, np.dot(sigma_dev,sigma_dev)))
    # Von Mises Stress (aka 'equivalent stress')
    sigma_VM = np.sqrt(3 * J2)
    
    # for now, just return these two to reproduce plot in segall
    return sigma_mean, tau_max
    
    

def load_h5_material(matFile, infoFile):
    """
    Load physical properties (info) and states variables (mat) from material file
    """
    #elastic = ['mu','lambda','density']
    #linear_maxwell = elastic + ['maxwell_time']
    #general_maxwell = linear_maxwell + ['shear_ratio']
    #powerlaw = elastic + ['reference_strain_rate', 'reference_stress', 'power_law_exponent']
    
    solution = h5py.File(infoFile, "r")
    vertices = solution['geometry/vertices'][:] #nodes
    cells = solution['topology/cells'][:].astype('int') #elements
    physical = solution['cell_fields']
    
    rho = physical['density'][:].flatten()
    G = physical['mu'][:].flatten()
    lam = physical['lambda'][:].flatten()
    
    # get dictionary of all elastic moduli (NOTE: big. could do for individual cells too)
    moduli = get_elastic_moduli({'G':G, 'lam':lam, 'rho':rho})    
    #E = moduli['E']
    # Convert to Vp/Vs
    #velocities = mod2vel(moduli)
    #Vp = velocities['Vp']
    
    elastic = ['total_strain', 'stress']
    #linear_maxwell = elastic + ['viscous_strain']
    #general_maxwell = linear_maxwell + ['viscous_strain_1', 'viscous_strain_2', 'viscous_strain_3']
    #powerlaw = elastic + ['viscous_strain']
    
    solution = h5py.File(matFile, "r")
    #vertices = solution['geometry/vertices'][:] #nodes
    #cells = solution['topology/cells'][:].astype('int') #elements #same as above!
    state = solution['cell_fields']
    #3D: xx,yy,zz,xy,yz,xz
    #2D: xx,yy,xy
    stress = state['stress'][0]
    strain = state['total_strain'][0]

    return vertices, cells, moduli, stress, strain
    
  
def extract_points(pointsFile):
    """
    return flattened arrays of coordinates and displacements from h5 file
    """
    coords,data,number = load_h5(pointsFile)
    x,y,z = np.hsplit(coords,3)
    displacements = data[0]
    ux,uy,uz = np.hsplit(displacements,3)
    

    return (x.flatten(), y.flatten(), z.flatten(),
            ux.flatten(), uy.flatten(), uz.flatten() )
  

def extract_points_visco(pointsFile,tstep=-1):
    """
    extract user-specified surface points for visco-elastic models
    comp: x(0), y(1), z(2), r(3) at specified time step.
    """
    coords,data,number,time = load_h5_visco(pointsFile,tstep)
    x,y,z = np.hsplit(coords,3)

    ux,uy,uz = np.hsplit(data,3)
    ur = np.hypot(ux,uy)
  
    #Return as vectors
    return x.flatten(), ur.flatten(), uz.flatten(), time.flatten()


def extract_profile_axisym(verts, data, elmtsize=500):
    """ plot profile along axis: x(0) or y(1)
    and component of deformation x(0), y(1), z(2), r(3)
    NOTE: this is quite slow... how to speed up?
    """
    ux = data[:,:,0].flatten()
    uy = data[:,:,1].flatten()
    ur = np.hypot(ux,uy)
    uz = data[:,:,2].flatten()
    
    # extract values from unstructured mesh
    x = verts[:,0]
    y = verts[:,1]
    ind = (abs(y)<= elmtsize) #within 1 element size of y=0
    X = x[ind]
    Y = y[ind]
    R = ur[ind]
    Z = uz[ind]
    
    # sort in order of increasing r for line plots
    sort = np.argsort(X)
    X = X[sort]
    R = R[sort]
    Z = Z[sort]
    
    return X, R, Z  


def extract_points2D(pointsFile):
    """
    for 2D FEM problems
    """
    x,y = np.hsplit(coords,2)
    displacements = data[0]
    ur,uz = np.hsplit(displacements,2)
    #Return as vectors
    return x.flatten(), ur.flatten(), uz.flatten()


def extract_profile(verts, data, imxy=(0,0), axis=0, comp=2, shift=False):
    """ plot profile along axis: x(0) or y(1)
    and component of deformation x(0), y(1), z(2), r(3)
    NOTE: this is quite slow... 
    """
    if comp == 3:
      x = data[:,:,0].flatten()
      y = data[:,:,1].flatten()
      z = np.hyopt(x,y)
    else:  
      z = data[:,:,comp].flatten()
    x = verts[:,0]
    y = verts[:,1]
    
    # interpolate data onto regular grid
    xi = np.linspace(x.min(), x.max(), x.size)
    yi = np.linspace(y.min(), y.max(), y.size)
    zi = griddata(x,y,z, xi,yi, interp='nn')
    
    # extract points for plotting
    arrx, arry = imshow2array(imxy)
    if axis == 0:
        Xi = xi
        Zi = zi[arrx,:]
    elif axis == 1:
        Xi = yi
        Zi = zi[:,arry]
    
    return Xi, Zi 


def radial2negative(ur_fem):
    """
    Radial output transition for + to - for complex or dipole source but ur=np.hypot(ux,uy)
    only returns positive root. Find 2nd inflection point and switch sign
    """
    #print ur_fem
    du_dx = np.diff(ur_fem.flatten())
    inflections = np.diff(np.sign(du_dx)).nonzero()[0] + 1
    #local_mins = (diff(sign(du_dx)) > 0).nonzero()[0] + 1
    #local_maxs = (diff(sign(du_dx)) < 0).nonzero()[0] + 1
    ind = inflections[1]
    ur_fem[ind:] = -ur_fem[ind:] 
    return ur_fem


def get_elastic_moduli(inputDict):
    """ Pass dictionary of 2 elastic moduli, return dictionary with the rest
    NOTE make sure units are consistent!
    example:
    moduli = get_elastic_moduli({'E':55.0,'G':36.0})
    """
    #print inputDict
    
    if ('K' in inputDict) and ('E' in inputDict):
        K = inputDict['K']
        E = inputDict['E']
        lam = ((3*K)*(3*K - E)) / (9*K - E)
        G = (3*K*E) / (9*K - E)
        nu = (3*K - E) / (6*K)
        
    elif 'K' in inputDict and 'G' in inputDict:
        K = inputDict['K']
        G = inputDict['G']
        E = (9*K*G) / (3*K + G)
        lam = K - (2*G/3)
        nu = (3*K - 2*G) / (2*(3*K+G))
        
    elif 'K' in inputDict and 'nu' in inputDict:
        K = inputDict['K']
        nu = inputDict['nu']
        E = 3*K*(1-2*nu)
        lam = (3*K*nu) / (1+nu)
        G = (3*K*(1-2*nu)) / (2*(1+nu))
    
    elif 'E' in inputDict and 'G' in inputDict:   
        E = inputDict['E']
        G = inputDict['G']
        K = (E*G) / (3*(3*G - E))
        lam = (G*(E - 2*G)) / (3*G - E)
        nu = E/(2*G) - 1
        
    elif 'E' in inputDict and 'nu' in inputDict:  
        E = inputDict['E']
        nu = inputDict['nu']
        K = E / (3*(1 - 2*nu))
        lam = (E*nu) / ((1+nu)*(1-2*nu))
        G = E / (2*(1+nu))
            
    elif 'lam' in inputDict and 'G' in inputDict:  
        lam = inputDict['lam']
        G = inputDict['G']
        K = lam + 2*G/3
        E = G*(3*lam + 2*G) / (lam + G)
        nu = lam / (2*(lam+G))

    elif 'G' in inputDict and 'nu' in inputDict:  
        G = inputDict['G']
        nu = inputDict['nu']
        K = (2*G*(1 + nu)) / (3*(1 - 2*nu))
        E = 2*G*(1 + nu)
        lam = (2*G*nu) / (1 - 2*nu)
    
    moduli = dict(K=K, E=E, lam=lam, G=G, nu=nu)
    
    if 'rho' in inputDict:
        moduli['rho'] = inputDict['rho']

    return moduli


# NOTE pylith elastic defaults give G=30GPa,E=75GPa,v=0.25,K=50GPa
def vel2mod(velocities):
    '''
    Input: dict(Vp=5000.0, Vs=3000.0, rho=2700.0)
    Output: dictionary of elastic parameters
    '''
    Vp = velocities['Vp']
    Vs = velocities['Vs']
    rho = velocities['rho']
    
    
    # From standard elastic wave formulas:
    G = rho*Vs**2
    lam = rho*Vp**2 - 2*G

    moduli = get_elastic_moduli(dict(G=G,lam=lam,rho=rho))
    #moduli['rho'] = rho #kept in get_elastic_moduli function

    return moduli


def calc_timestep(G,eta,mu1=0.5,nu=0.25):
    """
    pylith's definition of a stable timestep according to brad: 1/5 relaxtion time
    apparently of the maximum relaxation time for viscoelastic materials
    """
    tau0,tau1,tau2 = calc_relaxation_times(G,eta,mu1=0.5,nu=0.25)
    relax = max([tau0,tau1,tau2]) #should be min...
    relax = min([tau0,tau1,tau2])
    ts = 0.2 * relax
    print 'stable timestep < {:.3f}s'.format(ts)
    

def calc_relaxation_times(G,eta,mu1=0.5,nu=0.25):
    '''
    for standard linear solid formulation, there are three relaxtion times
    1) material 2) long, 3) short
    G = total shear modulus
    mu1 = fraction of shear modulus in spring that is in series w/ dashpot (G1 = mu1*G)
    eta = viscosity of dashpot
    
    Note if mu1=0 linear solid, if mu1=1 linear maxwell
    '''
    K = (2.0*G*(1+nu)) / (3*(1-(2*nu)))
    mu0 = 1.0 - mu1
    alpha = (3.0*K) + G #recurring terms
    beta = (3.0*K) + (G*mu0)
    try:
        tau0 = eta / (G*mu1)
    except:
        tau0 = np.inf #mu1=0 case (no relaxtion b/c just spring!)
    tau1 = (alpha / beta) * tau0
    
    try:
        tau2 = tau0 / mu0
    except:
        tau2 = np.inf
    
    #print('sec_per_day= 8.6400e4')
    #print('sec_per_yr=3.1536e7')
    print('relaxation times:')
    print('days:', np.array([tau0,tau1,tau2]) / 8.64e4)
    print('years:', np.array([tau0,tau1,tau2]) / 3.1536e7)
    
    return tau0, tau1, tau2 


def mod2vel(moduli):
    '''
    Input: dict(E=45e9,nu=0.25) NOTE: any two moduli will do
    Output: density, Vp, Vs
    '''
    moduli = get_elastic_moduli(moduli) #in case dictionary of just two parameters passed
    print(moduli)
    rho = moduli['rho']
    lam = moduli['lam']
    G = moduli['G']

    Vs = np.sqrt(G/rho)
    Vp = np.sqrt((lam + 2*G)/rho)

    #print 'Density= %f' % rho
    #print 'Vs\n%.3e\n' % Vs
    #print 'Vp\n%.3e\n' % Vp
    #return Vp,Vs,rho

    #return dictionary to be consistent with vel2mod
    velocities = {'Vs':Vs, 'Vp':Vp, 'rho':rho}
    return velocities


def vp2rho(vp):
    """
    Empirical relation between vp and vs from Currenti et al 2008
    based on 3rd-order polynomial from Gardner et al 1974, Christensen and
    Mooney 1995, Brocher 2005
    """
    rho = 1.2861 + 0.5498*vp - 0.0930*vp**2 + 0.007*vp**3
    return rho


def vs2vp(vs):
    """
    Empirical relation from Brocher 2005 valid for non-mafic 0<vs<4.5
    """
    vp = 0.9409 + 2.0947*vs - 0.8206*vs**2 + 0.2683*vs**3 - 0.0251*vs**4
    return vp


def vpvs2nu(vp=5773.0,vs=3333.0):
    """
    Poisson's ratio doesn't depend on density!, just Vp & Vs
    """
    nu = (0.5*vp**2 - vs**2) / (vp**2 - vs**2)
    return nu


def nu2vpvs(nu=0.25):
    ''' note pure quarts has nu~0.1 (Christensen 2001) '''
    vpvs = np.sqrt( (2*nu-2) / (2*nu -1) )
    return vpvs


def vpnu2vs(vp=5773.0, nu=0.25):
    """
    vs given vp and nu
    """
    vs = vp * np.sqrt((nu-0.5)/(nu-1))
    return vs


def vsmu2rho(vs,mu=40e9):
    """
    density given vs and mu
    """
    return mu/(vs**2)

def vp2vs(vp,ref='brocher',mafic=False):
    """
    Good fit all lithologies for 1.5<vp<8.0 EXCEPT Ca-rich, mafic, gabbros, serpentines (overpredics)
    """
    if mafic: # 5.25<vp<7.25
        vs = 2.88 + 0.52*(vp - 5.25)
    else:
        vs = 0.7858 - 1.2344*vp + 0.7949*vp**2 - 0.1238*vp**3 + 0.0064*vp**4
    return  vs


def calc_lithostat(z,p=2700.0,g=9.80665):
    """
    Lithostatic stress according to Pylith's definition of g=9.80665
    """
    litho = p*g*z
    print('%.3e' % litho)
    return litho


def calc_volume(P,a,nu=0.25,G=30e9):
    """
    given pressure and radius what is equivalent volume?
    """
    V = (P * np.pi * a**3) / G
    print('V={} km^3'.format(V/1e9))
    return V

def calc_pressure(V=None,uzmax=None,a=2e3,d=10e3,nu=0.25,G=30e9):
    """
    Estimate pressure from dV or uzmax in a halfspace model
    Input: a=radius, d=depth [m]
    Output: [MPa]
    """
    if V:
        P = (V*G) / (np.pi*a**3)
    else:
        P = (uzmax*G*d**2) / ((1-nu)*a**3)
    return P / 1e6

    
def xyz2file(vectorTuple,name='output_points.txt', fmt='%5.3f'):
    """
    Numpy vectors to ascii columns
    note fmt is limited to 5 colums, precision is 3 decimal numbers
    """
    #NOTE: could easily generalize to list of vectors
    # see savetxt for format options (eg. justification, sigfigs)
    np.savetxt(name, np.transpose(vectorTuple), fmt=fmt)
    
    
def make_surface_grid(x=(-100,100),y=(-100,100),dx=1.0,dy=1.0):
    """
    Use Pylith output points to output solution on a uniform grid as opposed
    to triangular surface elements. UNITS=km
    """
    x = np.arange(x[0],x[1]+dx,dx)
    y = np.arange(y[0],y[1]+dy,dy)
    
    X,Y = np.meshgrid(x,y)
    X = X.flatten()
    Y = Y.flatten()
    Z = np.zeros_like(X)
    xyz2file(X,Y,Z,'surface_grid_points.txt')


def calc_scaling_factor(lat0,lon0,k0=0.9996):
    """ scale factor for transverse mercator maps... confusing topic!
    http://forums.esri.com/Thread.asp?c=93&f=984&t=277480
    http://lists.maptools.org/pipermail/proj/2006-July/002419.html
    http://lists.maptools.org/pipermail/proj/2008-September/003743.html
    see section 8-11 of Snyder's Map Projections book
    """
    print('not implemented')
    

def calc_stats(x,ur,uz,ur_fem,uz_fem):
    """
    Calculate maximum and RMSE errors for benchmarking FEM against Mogi solution,
    requires that FEM and analytic output points are at the same location
    """
    n = x.size
    
    # Uz
    residuals = uz - uz_fem
    point = residuals.argmax()
    max_err = ((uz[point] - uz_fem[point]) / uz[point]) * 100
    mse = np.sum(residuals**2) / n
    rmseZ = np.sqrt(mse)
    print('Uz Max % Error = {:.3f}%  (at x={})'.format(max_err, x[point]))
    print('MSE = {:.4e}'.format(mse))
    print('RMSE = {:.4e}'.format(rmseZ))
    print('\n')
    
    # Ur
    residuals = ur - ur_fem
    point = residuals.argmax()
    max_err = ((ur[point] - ur_fem[point]) / ur[point]) * 100
    mse = np.sum(residuals**2) / n
    rmseR = np.sqrt(mse)
    print('Ur Max % Error = {:.3f}%  (at x={})'.format(max_err, x[point]))
    print('MSE = {:.4e}'.format(mse))
    print('RMSE = {:.4e}'.format(rmseR))
    print('\n')
    
    return rmseR, rmseZ

    
def load_gdal(path):
    #path = '/home/scott/data/dems/cgiar/uturuncu_1000_1000.tif'
    ds = gdal.Open(path)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    #ncol = ds.RasterXSize
    #nrow = ds.RasterYSize
    data = ds.GetRasterBand(1).ReadAsArray()
    
    return data,gt,proj
    

def comp2los(x,ux,uy,uz,track='east'):
    """
    Convert pylith cartesian output to LOS based on average viewing geometry for track
    """
    
    
    track2los = {'282': (22.8, -77.7),
                 '10': (22.0, -77.5),
                 '3': (20.0, 76.3),
                 '89': (41.1, 77.0),
                 '318': (23.1, 76.5)}

    # NOTE: load profile from merged dataset
    #NOTE: merged raster extends approximately 90km to East & West of maximum uplift
    
    #if track == 'merged':
    #    x,dlos,look,head = np.loadtxt('/home/scott/research/models/pylith/scripts/insarEWprofile.txt')
    
    #NOTE: for now do cludge fix noting that 282 & 10 profiles incidence varies 7 degrees from min to edge of scene
    #EW edge of data is ~90km from uplift source
    if track == 'east': #t10 incidence angle
        look = np.linspace(24.3, 19.3, uz.size)
        look[x>90e3] = 19.3
        head = np.ones_like(uz.size)*-77.6
    elif track == 'west':
        ux = -ux
        look = np.linspace(22.0, 27.0, uz.size)
        look[x>90e3] = 27.0
        head = np.ones_like(uz.size)*-77.6
    else:
        #print('NOTE that using mean incidence angle is not accurate: for example, consider Eastern EW profile through Uturuncu (this is near-range of t282, but far-range of t10)')
        look = track2los[track][0] #just use average incidence value
        head = track2los[track][1]
    
    #print uz[0],look[0],head[0]
    #print uz,look,head
    look = np.radians(look)
    head = np.radians(head)
    
    # uplift positive convention
    los = -(np.sin(look)*np.sin(head)*ux + np.sin(look)*np.cos(head)*uy -np.cos(look)*uz)
    #print('los_max = {}'.format(los.max()))
    
    return los
   
    
def los2pylith(surface, dem, unw, incidence):
    """ Convert InSAR LOS, incidence, heading to same grid as pylith output
    Input: paths to files
    Output: los, fem_los, residual arrays (coregistered)
    1) put georeferenced pylith output onto common grid with geo_incidence file
    2) convert pylith output to LOS
    3) 4x4 figure with fem_surface
    """
    #unw = '/home/scott/data/insar/cvz/t2282/geo/geo_stack282_8rlks.unw'
    #incidence = '/home/scott/data/insar/cvz/t2282/aux_files/geo_incidence_8rlks.unw'
    #dem = '/home/scott/data/dems/cgiar/uturuncu_1000_1000.tif'
    #surface = '/home/scott/research/models/pylith/3d/uturuncu_layered/output/step01/surface.h5'
    #geosurface = '/home/scott/research/models/pylith/scripts/geo_fem_Uz.tif'
    
    # Save georeferenced versions of ux,uy,uz surface output
    uxgeo = surface2geotiff(dem,surface,0)
    uygeo = surface2geotiff(dem,surface,1)
    uzgeo = surface2geotiff(dem,surface,2)
    
    # Match projection and grid of unw's to pylith output
    losgeo = coregister(unw, uzgeo)
    lookgeo = coregister(incidence, uzgeo, half=1, outfile='look.tif')
    headgeo = coregister(incidence, uzgeo, half=2, outfile='head.tif')
    
    # Load arrays
    ux,gt,proj = load_gdal(uxgeo)
    uy,gt,proj = load_gdal(uygeo)
    uz,gt,proj = load_gdal(uzgeo)
    los,gt,proj = load_gdal(losgeo)
    look,gt,proj = load_gdal(lookgeo)
    head,gt,proj = load_gdal(headgeo)
    
    # Get rid of zeros added by gdalwarp
    los[los==0] = np.nan
    look[look==0] = np.nan
    head[head==0] = np.nan
    
    # Convert stack to m, angles to radians
    los = los / 100.0
    look = np.radians(look)
    head = np.radians(head)
    
    # Convert fem to radar LOS
    fem_los = -1 * (np.sin(look)*np.sin(head)*ux + np.sin(look)*np.cos(head)*uy -np.cos(look)*uz)
    residual = los - fem_los
    
    return los,fem_los,residual
    
    

def coregister(sourceFile, targetFile, half=2, outfile=None):
    """Save geounw format to geotiff with grid matching fem ouput"""
    # Load unw as gdal format
    #unw = '/home/scott/data/insar/cvz/t2282/geo/geo_stack282_8rlks.unw'
    ig = rp.data.Geogram(sourceFile)
    
    src_filename = rp.tools.save_gdal(ig, half=half, outfile=outfile)
    src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()
    src_geotrans = src.GetGeoTransform()
    
    # We want a section of source that matches this:
    #geosurface = '/home/scott/research/models/pylith/scripts/geo_fem_Uz.tif'
    match_filename = targetFile
    match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize
    
    # Output / destination
    outfile = src_filename.replace('tif','_reproj.tif')
    dst = gdal.GetDriverByName('GTiff').Create(outfile, wide, high, 1, gdalconst.GDT_Float32)
    dst.SetGeoTransform(match_geotrans)
    dst.SetProjection(match_proj)
    
    # Reproject (equivalent to command line call of gdalwarp)
    # NOTE: odd addition of data on the right hand side (outside of image extents)
    # NOTE: could probably assign nan value to avoid this...
    gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)
    
    del dst # Flush
    return outfile
    
    
    
def surface2geotiff(demFile,surfaceFile,comp=2,nanval=-9999,outname=None,):
    """
    save pylith output as geotiff (for uturuncu layered model)
    component=0(x) 1(y) 2(z)
    nanval from cgial srtm tif -32768
    NOTE: to get in latlon
    gdalwarp -t_srs '+proj=longlat +datum=WGS84' geo_fem.tif geo_fem_latlon.tif
    """
    # Load surface DEM
    data,gt,proj = load_gdal(demFile)
    nrow,ncol = data.shape

    # Put FEM output onto same grid    
    #surfaceFile = '/home/scott/research/models/pylith/3d/uturuncu_layered/output/step01/surface.h5'
    verts,data,tris = load_h5(surfaceFile)
    
    if comp==3: #radial displacements
        z = np.hypot(data[:,:,0], data[:,:,1]).flatten()
    else:
        z = data[:,:,comp].flatten()
    x = verts[:,0]
    y = verts[:,1]

    # Affine transform to pixel upper left corner
    #xgeo_ul = gt[0]+ np.arange(ncol)*gt[1] + np.arange(nrow)*gt[2]
    #ygeo_ul = gt[3]+ np.arange(ncol)*gt[4] + np.arange(nrow)*gt[5]

    # Affine transform to pixel centers (add half pixel to start)
    xgeo_c = (gt[0]+ gt[1]/2.0)  + np.arange(ncol)*gt[1] + np.arange(nrow)*gt[2]
    ygeo_c = (gt[3]+ gt[1]/2.0)  + np.arange(ncol)*gt[4] + np.arange(nrow)*gt[5]
    
    #Xi,Yi = np.meshgrid(xi,yi)
    zi = griddata(x,y,z, xgeo_c,ygeo_c, interp='nn') #'nn','linear',etc
    #zi = zi.filled(nanval) #replace nan's with -9999
    zi = zi.data.astype('float32')
    
    # Save as gdal-recognized file
    compdict = {0:'Ux',1:'Uy',2:'Uz',3:'Ur'}
    outname = 'geo_fem_{}.tif'.format(compdict[comp])
    #NOTE: can probably just swap array in ds from DEM above...
    driver = gdal.GetDriverByName('GTiff')
    nd = driver.Register()
    outDataset = driver.Create(outname, ncol, nrow, 1, gdal.GDT_Float32)
    
    outBand = outDataset.GetRasterBand(1)
    #outBand.SetNoDataValue(nanval)
    #zi[np.isnan(zi)] = nanval
    
    result = outBand.WriteArray(zi, 0, 0) 
    result = outDataset.SetGeoTransform(gt)
    outSR = osr.SpatialReference()
    #proj4str = '+proj=tmerc +ellps=WGS84 +datum=WGS84 +lon_0=-67.18 +lat_0=-22.27 +units=m'
    #result = outSR.ImportFromProj4(proj4str)
    #outWkt = outSR.ExportToWkt()
    result = outDataset.SetProjection(proj)
    
    #Write to disk & Close
    outDataset.FlushCache()
    del outDataset
    
    return outname
   


def tomo2spatialdb(tomoFile, density=2700.0, vpvs=1.75, maxdepth=10.1,
                   ave_elev=4.630, vpfill=5100, vsfill=3000):
    """
    Convert Matt Haney's matlab surface wave tomography data to simpledb file
    ave_elev=  match cubit mesh
    maxdepth= only use data down to specified depth
    (BELOW sea level end w/0.1 b/c cell centers) [km]
    """
    templatedir = '/home/scott/research/models/pylith/scripts/templates/'
    outdir = '/home/scott/research/models/pylith/3d/uturuncu_layered/spatialdb/'
    matfile = '/home/scott/research/haney_tomography/vel_utu_depm2.mat'
    
    # Load Tomography Matlab file
    mat_dict = sp.io.loadmat(matfile)
    Vs = mat_dict['vel3dz']
    # Set-up 3d grid cropped & oriented 3D mesh (+x=E,+y=N,+z=up)
    xcs = np.arange(-28.875,31.876,2.25) #full grid
    ycs = np.arange(-25.875,25.876,2.25)
    zcs = np.arange(0.1,30.0,0.2)
    X,Y,Z = np.mgrid[-25.875:25.876:2.25, -28.875:31.876:2.25, 0.1:maxdepth:0.2]
    # Shift z-values by average elevation
    Z = ave_elev - Z
    
    # Crop to depth
    zind = np.argwhere(np.abs(zcs - maxdepth)<0.001)
    Vs = np.rot90(Vs[:,:,:zind]) #NOTE: switch in X and Y below
    Vp = vpvs * Vs
    density = density * np.ones_like(Vs)

    
    # Fix velocities below max depth due to poor model resolution
    # ----------------
    #Vs[Z <= (ave_elev-maxdepth)] = 3500.0
    #Vs[Z == Z.min()] = 3500.0 # fix bottom plane for Upper Crust to 
    # Fix values more than 25km away from summit
    R = np.hypot(X,Y)
    ind = R>=25.0
    Vs[ind] = vsfill #note: search and replace this number w/ whatever is appropriate for layer
    Vp[ind] = vpfill #these are values applt above sea level
    # For example, take values from mike west's model for upper crust vp = 5600, vs=3300
    
    # Concatenate vectors for saving
    data = np.hstack([X.reshape(-1,1),
                      Y.reshape(-1,1),
                      Z.reshape(-1,1),
                      Vp.reshape(-1,1),
                      Vs.reshape(-1,1),
                      density.reshape(-1,1)])
    
    # NOTE: need double {{ in format string to escape character
    with open(templatedir + 'simple.spatialdb', 'r') as template:
        string = template.read()
        header = string.format(npoints=Vs.size)
    
    # Save spatialdb
    outpath = outdir + 'haney_tomo_simple.spatialdb'
    with open(outpath, 'w') as f:
        f.write(header)
        np.savetxt(f,data,delimiter=' ',fmt='%1.3f') #can also try %g
        
    print('Saved {}'.format(outpath))
    
    
    
def tomo2griddb(tomoFile, ave_elev=4.630):
    """
    Convert Matt Haney's matlab surface wave tomography data to spatialdb
    material property file needed by pylith
    NOTE: should probably shift
    """
    path = '/home/scott/research/models/pylith/scripts/templates/'
    #tomoFile = 'haney_tomo.dat'
    
    # Load Tomography
    tomo = np.loadtxt(path + tomoFile)#, unpack=True)
    x,y,z,vs = np.hsplit(tomo,4)
    
    # shift z-values by average elevation
    z = ave_elev - z
    
    xvals = np.unique(x)
    yvals = np.unique(y)
    zvals = np.unique(z)
    
    # Fix velocities below sea level (~5km below surface) 
    vs[z<=0.0] = 3500.0
    # Fix values more than 25km away from summit
    r = np.hypot(x,y)
    vs[r>=25.0] = 3500.0
    
    # Array to string of numbers
    #tmp = str(xvals).strip('[]') #note keeps '\n' chartacters wherever printed
    #xvals_str = tmp.remove('\n','')
    #xvals_str = ' '.join(map(str, xvals.tolist()))
    #xvals_str = ' '.join([str(x) for x in xvals.tolist()])
    # NOTE: easier to use svaetxt for big arrays (see below)
    
    # Make dictionary of values
    # NOTE 'vals arrays prob won't work too well here?
    values = dict(nx=xvals.size,
                  ny=yvals.size,
                  nz=zvals.size,
                  nvals=3,
                  ndim=3,
                  names='vp vs density',
                  units='m/s  m/s kg/m**3',
                  scale=1000.0,#cubit units in meters
                  xvals=' '.join(map(str, xvals.tolist())),
                  yvals=' '.join(map(str, yvals.tolist())),
                  zvals=' '.join(map(str, zvals.tolist())),
                  )
    
    # Add vp and desnsity
    vp = 1.75*vs
    density = 2700*np.ones_like(vs)
    data = np.hstack([x,y,z,vp,vs,density])
    
    # Load
    # NOTE: need double {{ in format string to escape character
    name = 'grid.spatialdb'
    with open(path + name, 'r') as template:
        string = template.read()
        header = string.format(**values)
    
    # Save spatialdb
    name = 'haney_tomo_grid.spatialdb'
    with open(path + name, 'w') as f:
        f.write(header)
        np.savetxt(f,data,delimiter=' ',fmt='%1.3f') #can also try %g
        
    print('Saved {}'.format(name))
  
  
# NOTES: simplest way to create a spatialDB file with python script? :
'''
# See /home/scott/software/pylith-1.9.0/src/pylith/examples/2d/subduction/afterslip_tractions.py
# for example of creating spatialDB with background + surface tractions on fault


# Create coordinate system for spatial database
from spatialdata.geocoords.CSCart import CSCart
cs = CSCart()
cs._configure()
cs.setSpaceDim(2)


# Create writer for spatial database file
from spatialdata.spatialdb.SimpleIOAscii import SimpleIOAscii
writer = SimpleIOAscii()
writer.inventory.filename = "afterslip_tractions.spatialdb"
writer._configure()
writer.write({'points': vertices,
              'coordsys': cs,
              'data_dim': 1,
              'values': [{'name': "traction-shear",
                          'units': "Pa",
                          'data': tractions_shear},
                         {'name': "traction-normal",
                          'units': "Pa",
                          'data': tractions_normal}]})



'''
