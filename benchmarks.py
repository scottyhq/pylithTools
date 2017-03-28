"""
Make figures of FEM benchmark tests
"""

#import pylith_plotter3D as pp
import roipy.models as m
import matplotlib.pyplot as plt
import numpy as np

import pylithTools.util as pu
import pylithTools.plot as pp

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


def plot_result(x,ur,uz,ur_fem,uz_fem,params=None,plotcomp=False,plotlos=False,xlim=100):
    """
    Compare FEM to analytic profile. Relative & Absolute Error
    """
    x = x/1000.0 #convert to km
    # load uz and ur from inversion, make all points positve

    fig = plt.figure(figsize=(17,11))
    ax = fig.add_subplot(131)
    plt.plot(x, uz*100, 'b-', lw=2, label='analytic')
    plt.plot(x, ur*100, 'g-', lw=2)
    plt.plot(x, uz_fem*100, marker='o', ls='None', lw=2, mec='b', mfc='None',label='FEM')
    plt.plot(x, ur_fem*100, marker='o', ls='None', lw=2, mec='g', mfc='None')
    
    if plotcomp:
        lon, lat, x_data, ur_data, uz_data = np.loadtxt('profiles.txt',unpack=True)
        uz_data[uz_data==0] = np.nan
        ur_data[ur_data==0] = np.nan
        indWest = x_data < 0
        indEast = x_data >= 0
        uz_west = uz_data[indWest]
        uz_east = uz_data[indEast]
        ur_west = -1 * ur_data[indWest]
        ur_east = ur_data[indEast]
        x_west = -1 * x_data[indWest]
        x_east = x_data[indEast]
        plt.plot(x_east, uz_east, 'b.', ls='None',label='InSAR')
        plt.plot(x_east, ur_east, 'g.', ls='None')
    
    if plotlos:
        xlos, ulos, stdlos = np.loadtxt('collapsed_profile.txt',unpack=True)
        plt.plot(xlos, ulos, 'k.', ls='None',label='InSAR')
    
    
    plt.title('FEM vs. Analytic Solution')
    plt.xlabel('Radial distance [km]')
    plt.ylabel('displacement [cm]')
    plt.grid(True)
    plt.axhline(color='k')
    plt.xlim(0,100)
    plt.legend()
    
    # Plot Residual
    #plt.figure()
    ax = fig.add_subplot(132)
    plt.plot(x, (uz-uz_fem)/1000, 'b-', lw=2, label='uz')
    plt.plot(x, (ur-ur_fem)/1000, 'g-', lw=2, label='ur')
    plt.title('Absolute Error')
    plt.xlabel('Radial distance [km]')
    plt.ylabel('residual [mm]')
    plt.grid(True)
    #ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    #pyplot method for the above:
    plt.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    plt.xlim(0,100)
    plt.legend()
    
    # Plot Relative Error
    ax = fig.add_subplot(133)
    #plt.figure()
    plt.plot(x, (uz-uz_fem)/uz*100, 'b-', lw=2, label='uz')
    plt.plot(x, (ur-ur_fem)/ur*100, 'g-', lw=2, label='ur')
    plt.title('Relative Error')
    plt.xlabel('Radial distance [km]')
    plt.ylabel('Relative Error (%)')
    plt.grid(True)
    plt.xlim(0,100)
    plt.legend()
    
    if params:
        titlestr = 'depth={d:g}, radius={a:g}, dP={dP:2.2e}, mu={mu:2.2e}, nu={nu:g}'.format(**params)
        plt.suptitle(titlestr)


def mogi_currenti():
    """
    Reproduce Gilda Currenti's Mogi source benchmark using a Cubit Mesh
    instead of LaGrit Mesh. Domain 100x100x50km, source at 3km depth
    """
    surfaceh5 = '/home/scott/research/models/pylith/3d/mogi_currenti/output/surface.h5'
    pointsh5 = '/home/scott/research/models/pylith/3d/mogi_currenti/output/points.h5'
    
    #Get analytic solution
    #x = np.linspace(0,35e3,1e2) #100 points
    #x = np.arange(36)*1e3 #simple output_points
    x = 1e3*np.loadtxt('/home/scott/research/models/pylith/3d/mogi_currenti/output_points.txt',usecols=[0])
    y = np.zeros_like(x)
    params = dict(xoff = 0,
                  yoff = 0,
                  d = 3e3,
                  dP = 10e6,
                  a = 500.0,
                  mu = 4e9,
                  nu = 0.25)
    ur,uz = m.calc_mogi_dp(x,y,**params)
    
    # Get FEM solution
    #verts, data, tris = load_h5(surfaceh5)
    #X, R, Z = extract_profile_axisym(verts, data, elmtsize=500)
    x_fem, ur_fem, uz_fem = pp.extract_points(pointsh5)
    
    # Print out statistics
    calc_stats(x,ur,uz,ur_fem,uz_fem)
    
    plot_result(x,ur,uz,ur_fem,uz_fem,params)


def mogi_uturuncu_moat():
  """
  Mogi benchmark for a deep source in a larger domain. Domain 200x200x100km,
  source at 10km. Can be used for Dipole Source Model.
  """
  surfaceh5 = '/home/scott/research/models/pylith/3d/uturuncu_moat/output/elastic1/surface.h5'
  pointsh5 = '/home/scott/research/models/pylith/3d/uturuncu_moat/output/elastic1/points.h5'
  
  #Get analytic solution
  #x = np.linspace(0,35e3,1e2) #100 points
  #x = np.arange(36)*1e3 #simple output_points
  x = 1e3*np.loadtxt('/home/scott/research/models/pylith/3d/uturuncu/output_points.txt',usecols=[0])
  y = np.zeros_like(x)
  
  # Different Mogi formulations, same result
  #params = dict(xoff = 0,
  #              yoff = 0,
  #              d = 10e3,
  #              dV = 41e6,
  #              nu = 0.25)
  #ur,uz = m.calc_mogi(x,y,**params)
  
  #params = dict(xoff = 0,
  #              yoff = 0,
  #              d = 25e3,
  #              dP = 53e6,
  #              a = 1e3,
  #              mu = 4e9, #volcanic regions
  #              nu = 0.25)
  #ur,uz = m.calc_mogi_dp(x,y,**params)
  
  params = dict(xoff = 0,
                yoff = 0,
                d = 10e3, #5e3
                dP = 100e6,
                a = 1e3, #0.7e3
                mu = 30e9, #volcanic regions
                nu = 0.25)
  ur,uz = m.calc_mogi_dp(x,y,**params)
  
  
  # Get FEM solution
  #verts, data, tris = load_h5(surfaceh5)
  #X, R, Z = extract_profile_axisym(verts, data, elmtsize=500)
  x_fem, ur_fem, uz_fem = pp.extract_points(pointsh5)
  
  # Statistics & Graph
  calc_stats(x,ur,uz,ur_fem,uz_fem)
  plot_result(x,ur,uz,ur_fem,uz_fem,params,plotlos=True,plotcomp=False)
  
  
def mogi_DN2009_elastic():
    """
    Reproduce elastic model from Del Negro 2009 for viscoelastic model. Domain is
    50x50x35, source at 4km depth. Then use same source in slightly bigger mesh
    for uturuncu UPLIFT modeling (Domain 100x100x50)
    """
    #DEL NEGRO MESH
    #surfaceh5 = '/home/scott/research/models/pylith/3d/delnegro2009/output/surface.h5'
    #pointsh5 = '/home/scott/research/models/pylith/3d/delnegro2009/output/points.h5'
    #points = '/home/scott/research/models/pylith/3d/delnegro2009/output_points.txt'
    #UTURNCU MESH
    surfaceh5 = '/home/scott/research/models/pylith/3d/uturuncu/output_DN2009/surface.h5'
    pointsh5 = '/home/scott/research/models/pylith/3d/uturuncu/output_DN2009/points.h5'
    points = '/home/scott/research/models/pylith/3d/uturuncu/output_points.txt'
    
    #Get analytic solution
    #x = np.linspace(0,25e3,1e2)
    #y = np.zeros_like(x)
    x = 1e3*np.loadtxt(points,usecols=[0])
    y = np.zeros_like(x)
    params = dict(xoff = 0,
                  yoff = 0,
                  d = 4e3,
                  dP = 100e6,
                  a = 700,
                  mu = 30e9,
                  nu = 0.25)
    #ur,uz = m.calc_mogi_dp(x,y,d=4e3,dP=100e6,a=700,mu=30e9,nu=0.25)
    ur,uz = m.calc_mogi_dp(x,y,**params)
    
    # Get FEM solution
    #verts, data, tris = load_h5(surfaceh5)
    #X, R, Z = extract_profile_axisym(verts, data, elmtsize=500)
    x_fem, ur_fem, uz_fem = pp.extract_points(pointsh5)
    
    # Statistics & Graph
    calc_stats(x,ur,uz,ur_fem,uz_fem)
    plot_result(x,ur,uz,ur_fem,uz_fem,params)
    
    
def mogi_LZ2003_layered():
    """
    Fully 3D model for investigating Uplift source in the mid-crust. Used to
    investigate layering proposed by Leidig & Zandt 2003, and seismic tomography
    from Matt Haney. Domain 150x100 (cylinder), depth=25km. NOTE: also a version
    of mesh that includes topography.
    """
    #UTURNCU MESH
    surfaceh5 = '/home/scott/research/models/pylith/3d/uturuncu_layered/output/step01/surface.h5'
    pointsh5 = '/home/scott/research/models/pylith/3d/uturuncu_layered/output/step01/points.h5'
    points = '/home/scott/research/models/pylith/3d/uturuncu_layered/output_points.txt'
    
    #Get analytic solution
    #x = np.linspace(0,25e3,1e2)
    #y = np.zeros_like(x)
    x = 1e3*np.loadtxt(points,usecols=[0])
    y = np.zeros_like(x)
    
    params = dict(xoff = 0.0, #-2150.0,
                  yoff = 0.0, #250.0,
                  d = 26e3,
                  dP = 33.0e6,
                  a = 2000.0,
                  mu = 30e9,
                  nu = 0.25)
    #ur,uz = m.calc_mogi_dp(x,y,d=4e3,dP=100e6,a=700,mu=30e9,nu=0.25)
    ur,uz = m.calc_mogi_dp(x,y,**params)
    
    # Get FEM solution
    #verts, data, tris = load_h5(surfaceh5)
    #X, R, Z = extract_profile_axisym(verts, data, elmtsize=500)
    x_fem, ur_fem, uz_fem = pp.extract_points(pointsh5)
    
    # Statistics & Graph
    calc_stats(x,ur,uz,ur_fem,uz_fem)
    plot_result(x,ur,uz,ur_fem,uz_fem,params)


def mogi_tomography():
    """
    Compare homogeneous solution w/ Wa Domain 150x100 (cylinder), depth=25km. NOTE: also a version
    of mesh that includes topography.
    """
    #UTURNCU MESH
    surfaceh5 = '/home/scott/research/models/pylith/3d/uturuncu_layered/output/step01/surface.h5'
    pointsh5 = '/home/scott/research/models/pylith/3d/uturuncu_layered/output/step01/points.h5'
    points = '/home/scott/research/models/pylith/3d/uturuncu_layered/output_points.txt'
    
    #Get analytic solution
    #x = np.linspace(0,25e3,1e2)
    #y = np.zeros_like(x)
    x = 1e3*np.loadtxt(points,usecols=[0])
    y = np.zeros_like(x)
    
    params = dict(xoff = 0.0, #-2150.0,
                  yoff = 0.0, #250.0,
                  d = 26e3,
                  dP = 33.0e6,
                  a = 2000.0,
                  mu = 30e9,
                  nu = 0.25)
    #ur,uz = m.calc_mogi_dp(x,y,d=4e3,dP=100e6,a=700,mu=30e9,nu=0.25)
    ur,uz = m.calc_mogi_dp(x,y,**params)
    
    # Get FEM solution
    #verts, data, tris = load_h5(surfaceh5)
    #X, R, Z = extract_profile_axisym(verts, data, elmtsize=500)
    x_fem, ur_fem, uz_fem = pp.extract_points(pointsh5)
    
    # Statistics & Graph
    calc_stats(x,ur,uz,ur_fem,uz_fem)
    plot_result(x,ur,uz,ur_fem,uz_fem,params)


def mogi_dipole(track='west'):
    """
    Based on Tom's inversion for dipole source
    """
    outdir = '/home/scott/research/models/pylith/3d/agu2013_noapmb/output/step01'
    pointsh5 = '/home/scott/research/models/pylith/3d/agu2013_noapmb/output/step01/points.h5' 

    pp.plot_profile(outdir,comp2los=track,adjustRadial=True,yscale=1e-2,xscale=1e3)
    
    # Get FEM solution arrays (displacements should be exact at nodal points!)
    x_fem, y_fem, z_fem, ux_fem, uy_fem, uz_fem = pu.extract_points(pointsh5)
    r_fem = np.hypot(x_fem,y_fem)
    ur_fem = np.hypot(ux_fem,uy_fem)
    ur_fem = pu.radial2negative(ur_fem) 

    # Inflation Source
    params = dict(xoff = 0,
                  yoff = 0,
                  d = 30e3, #increasing depth is positive
                  dP = 20e6,
                  a = 3e3,
                  mu = 30e9,
                  nu = 0.25)
    ur1,uz1 = m.calc_mogi_dp(x_fem,y_fem,**params)
    
    # Deflation Source
    ur2,uz2 = m.calc_mogi_dp(x_fem,y_fem,d=70e3,a=4e3,mu=30e9,dP=-20e6)

    # Superposition of analytic solutions
    ur = ur1 + ur2
    uz = uz1 + uz2

    #pu.calc_stats(r_fem, ur,uz,ur_fem,uz_fem)
    
    if track == 'west':
        r_fem = -r_fem
        ur = -ur
    
    # Plot elastic benchmark solution on top of active plot!
    #plt.figure()
    #plt.gcf()
    every=2 #NOTE: this is a bit cleaners, avoid too much clutter...
    plt.plot(r_fem/1e3, uz*1e2, marker='o', markevery=every, ls='None', lw=2, ms=5, mew=1, mec='b',mfc='None',label='analytic')
    plt.plot(r_fem/1e3, ur*1e2, marker='s', markevery=every, ls='None', lw=2, ms=5, mew=1, mec='b',mfc='None')
    
    datalos = '/home/scott/research/models/pylith/scripts/collapsed_profile.txt'
    xlos, ulos, stdlos = np.loadtxt(datalos, unpack=True) #points in cm & in km from center!
    if track == 'west':
        xlos = -xlos
    plt.plot(xlos, ulos, 'k.', ls='None',label='InSAR')
    
    plt.show()
    

    
def mogi_agu_elastic():
    """
    Large domain mesh that exploited reflection symmetry of large domain for
    simulating stacked Dipole source. Domain 600x200km (1/4 r=300km)  
    """
    #DEL NEGRO MESH
    #surfaceh5 = '/home/scott/research/models/pylith/3d/delnegro2009/output/surface.h5'
    #pointsh5 = '/home/scott/research/models/pylith/3d/delnegro2009/output/points.h5'
    #points = '/home/scott/research/models/pylith/3d/delnegro2009/output_points.txt'
    #UTURNCU MESH
    surfaceh5 = '/home/scott/research/models/pylith/3d/fialko2012/model3_agu/output/surface.h5'
    pointsh5 = '/home/scott/research/models/pylith/3d/fialko2012/model3_agu/output/elastic/apmb/points.h5'
    points = '/home/scott/research/models/pylith/3d/fialko2012/model3_agu/output_points.txt'
    
    #Get analytic solution
    #x = np.linspace(0,25e3,1e2)
    #y = np.zeros_like(x)
    #NOTE: in this case need radial distance b/c not on x-axis
    x = np.sqrt(2)*1e3*np.loadtxt(points,usecols=[0])
    y = np.zeros_like(x)
    params = dict(xoff = 0,
                  yoff = 0,
                  d = 25e3,
                  dP = 50e6,
                  a = 2e3,
                  mu = 30e9,
                  nu = 0.25)
    
    # Superposition of inflation and deflation source
    #ur,uz = m.calc_mogi_dp(x,y,d=4e3,dP=100e6,a=700,mu=30e9,nu=0.25)
    ur1,uz1 = m.calc_mogi_dp(x,y,**params)
    ur2,uz2 = m.calc_mogi_dp(x,y,d=75e3,a=5e3,mu=30e9,dP=-10e6)
    
    # Get FEM solution
    #verts, data, tris = load_h5(surfaceh5)
    #X, R, Z = extract_profile_axisym(verts, data, elmtsize=500)
    x_fem, ur_fem, uz_fem = pp.extract_points(pointsh5)
    # NOTE: cludge fix for negative radial displacements in dipole model
    ur_fem[x >= 60000] = -ur_fem[x >= 60000]
    #void-space
    #ur_fem[x >= 1000] = -ur_fem[x >= 1000]
    
    #Convert units for plotting (done in plotting routine)
    #x = x/1000.0
    #ur1 = ur1 * 100.0
    #uz1 = uz1 * 100.0
    #ur2 = ur2 * 100.0
    #uz2 = uz2 * 100.0
    #ur_fem = ur_fem * 100.0
    #uz_fem = uz_fem * 100.0
    
    # Superposition of analytic solutions
    ur = ur1 + ur2
    uz = uz1 + uz2

    
    # Statistics & Graph
    calc_stats(x,ur,uz,ur_fem,uz_fem)
    plot_result(x,ur,uz,ur_fem,uz_fem,params,plotlos=True)
  
  
def agu_convergence():
    """
    Mesh Refinement demonstration for 1/4 cylinder large domain mesh
    """
    
    points = '/home/scott/research/models/pylith/3d/fialko2012/model3_agu/output_points.txt'
    
    outputs = ['/home/scott/research/models/pylith/3d/fialko2012/model3_agu/output/elastic/supercoarse/points.h5',
               '/home/scott/research/models/pylith/3d/fialko2012/model3_agu/output/elastic/coarse/points.h5',
               '/home/scott/research/models/pylith/3d/fialko2012/model3_agu/output/elastic/medium/points.h5',
               '/home/scott/research/models/pylith/3d/fialko2012/model3_agu/output/elastic/fine/points.h5',
               '/home/scott/research/models/pylith/3d/fialko2012/model3_agu/output/elastic/test/points.h5',
               '/home/scott/research/models/pylith/3d/fialko2012/model3_agu/output/elastic/superfine/points.h5',
              ]
    
    #number of elements
    #supercoarse 40000/1000/1000 = 300360
    #coarse 30000/1000/500 = 445841
    #medium 25000/750/300 = 642087
    #fine 20000/500/250 = 832400
    #test? 30000/300/150 = 1170025
    #superfine 10000/250/100 = 1955701
    #max? 20000/250/50 = 2840658  #NOTE: Fialko max # elements=2.5mil
    nel = np.array([300360,445841,642087,832400,1170025,1955701])
    RMSE = []
    for pointsh5 in outputs:
        x = np.sqrt(2)*1e3*np.loadtxt(points,usecols=[0])
        y = np.zeros_like(x)
        params = dict(xoff = 0,
                      yoff = 0,
                      d = 25e3,
                      dP = 50e6,
                      a = 2e3,
                      mu = 30e9,
                      nu = 0.25)
        
        # Superposition of inflation and deflation source
        ur1,uz1 = m.calc_mogi_dp(x,y,**params)
        ur2,uz2 = m.calc_mogi_dp(x,y,d=75e3,a=5e3,mu=30e9,dP=-10e6)
        ur = ur1 + ur2
        uz = uz1 + uz2
    
        x_fem, ur_fem, uz_fem = pp.extract_points(pointsh5)
        ur_fem[x >= 60000] = -ur_fem[x >= 60000]
        
        rmseR, rmseZ = calc_stats(x,ur,uz,ur_fem,uz_fem)
        RMSE.append(rmseR + rmseZ)
  
    RMSE = np.array(RMSE)
  
    plt.figure()
    plt.plot(nel/10000,RMSE*100,'k.-',lw=2)
    plt.title('Benchmark Mesh Accuracy')
    plt.xlabel('# Elements [x1e4]')
    plt.ylabel('RMSE [mm]')
    plt.grid()
  
  
def agu_apmb_weakness():
    """
    Test different material properties of APMB layer in AGU mesh, comment
    outputs to select different material property settings
    """
    
    points = '/home/scott/research/models/pylith/3d/fialko2012/model3_agu/output_points.txt'
    
    homogeneous = '/home/scott/research/models/pylith/3d/fialko2012/model3_agu/output/elastic/medium/points.h5'
    
    outputs = [#'/home/scott/research/models/pylith/3d/fialko2012/model3_agu/output/elastic/weak/points.h5',
               '/home/scott/research/models/pylith/3d/fialko2012/model3_agu/output/elastic/weaker/points.h5',
               '/home/scott/research/models/pylith/3d/fialko2012/model3_agu/output/elastic/weakest/points.h5',
              ]
    
    xe, ure, uze = pp.extract_points(homogeneous)
    #ur_fem[xe >= 60000] = -ur_fem[xe >= 60000] #or limit view...
    
    normZ = uze.max()
    normR = ure.max()
    
    plt.figure()
    plt.plot(xe,uze/normZ,'k-')
    plt.plot(xe,uze/normR,'k--')
    
    Gs = np.array([30,20,10,0.01])
    colors = ['b','g','r']
    for G,c,pointsh5 in zip(Gs,colors,outputs):    
        x_fem, ur_fem, uz_fem = pp.extract_points(pointsh5)
        ur_fem[xe >= 60000] = -ur_fem[xe >= 60000]
    
    plt.plot(xe,uze/normZ,ls='-',color=c, label=G)
    plt.plot(xe,uze/normR,ls='--',color=c)
    
    plt.title('Buffering Effect of APMB')
    plt.xlabel('Radial Distance [km]')
    plt.ylabel('Normalized Displacement')
    plt.grid()
    
  
def mogi_DN2009_visco():
    """
    Reproduce VISCOELASTIC model from Del Negro 2009 for viscoelastic model. Domain is
    50x50x35, source at 4km depth.
    """
    surfaceh5 = '/home/scott/research/models/pylith/3d/delnegro2009/output_visco/surface.h5'
    pointsh5 = '/home/scott/research/models/pylith/3d/delnegro2009/output_visco/points.h5'
    
    #Get analytic solution
    x = 1e3*np.loadtxt('/home/scott/research/models/pylith/3d/delnegro2009/output_points.txt',usecols=[0])
    y = np.zeros_like(x)
    params = dict(xoff = 0,
                    yoff = 0,
                    d = 4e3, #m
                    dP = 100e6, #Pa
                    a = 700, #m
                    nu = 0.25,
                    G = 30e9, #Pa
                    eta = 2e16, #Pa*s
                    mu1 = 0.5)
    
    fig = plt.figure()
    plt.suptitle('DelNegro2009 Fig 2', fontsize=16)
    timesteps = [0,5,10,14]
    # Loop through output points
    for i,tstep in enumerate(timesteps):
        print(i, tstep)
        x_fem, ur_fem, uz_fem,time = pp.extract_points_visco(pointsh5,tstep)
        tau0 = params['eta'] / (params['G']*params['mu1'])
        time_days = time / 8.64e4
        time_maxwell = time / tau0
        
        ur,uz = m.calc_mogi_genmax(x,y,time,**params)
        
        ax = fig.add_subplot(2,2,i+1)
        ax.plot(x, uz, 'b-', lw=2, label='analytic')
        ax.plot(x, ur, 'g-', lw=2)
        ax.plot(x, uz_fem, marker='o', ls='None', lw=2, mec='b', mfc='None',label='FEM')
        ax.plot(x, ur_fem, marker='o', ls='None', lw=2, mec='g', mfc='None')
        #ax.set_ylim(-0.01,0.07)
        ax.set_ylim(0,0.1)
        ax.set_xlim(0,10e3)
        ax.set_title('{:g} days, t/tau={:g}'.format(time_days,time_maxwell))
        ax.grid(True)
        
        if i==2:
            ax.set_xlabel('Radial distance [m]')
            ax.set_ylabel('displacement [m]')
        
        # Statistics & Graph
        print("time={}\n".format(time_days))
        calc_stats(x,ur,uz,ur_fem,uz_fem)
    
    # Residual plots for last timestep
    plot_result(x,ur,uz,ur_fem,uz_fem,params)   

    #plt.xlim(0,35000)
    plt.legend()
    

def mogi_PS2004_layered(depth=12e3):
    """
    Compare Uturuncu uplift source basic domain (100x100x50) to
    Pritchard & Simons 2004 elastic layering calculations propagator matrix
    calculations
    """
        #UTURNCU MESH
    if depth == 12e3:
        h5elastic = '/home/scott/research/models/pylith/3d/uturuncu/output_PS2004/points.h5'
        h5m1 = '/home/scott/research/models/pylith/3d/uturuncu/output_PS2004_M1/points.h5'
        h5m2 = '/home/scott/research/models/pylith/3d/uturuncu/output_PS2004_M2/points.h5'
    elif depth == 18e3:
        h5elastic = '/home/scott/research/models/pylith/3d/uturuncu/output_PS2004_18/points.h5'
        h5m1 = '/home/scott/research/models/pylith/3d/uturuncu/output_PS2004_M1_18/points.h5'
        h5m2 = '/home/scott/research/models/pylith/3d/uturuncu/output_PS2004_M2_18/points.h5'
        
    points = '/home/scott/research/models/pylith/3d/uturuncu/output_points.txt'
    
    #Get analytic solution
    x = 1e3*np.loadtxt(points,usecols=[0])
    y = np.zeros_like(x)
    params = dict(xoff = 0,
                  yoff = 0,
                  d = depth, #12e3,18e3
                  dP = 100e6,
                  a = 1000.0,
                  mu = 33e9,
                  nu = 0.25)
    ur,uz = m.calc_mogi_dp(x,y,**params)
    
    x_e, ur_e, uz_e = pp.extract_points(h5elastic)
    x_m1, ur_m1, uz_m1 = pp.extract_points(h5m1)
    x_m2, ur_m2, uz_m2 = pp.extract_points(h5m2)
    
    # Statistics & Graph for elastic solution
    calc_stats(x,ur,uz,ur_e,uz_e)
    plot_result(x,ur,uz,ur_e,uz_e,params)
    
    #redo figure 3
    if depth == 12e3: norm=uz_m2.max()
    elif depth == 18e3: norm=uz_m1.max()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x/1000, uz_e/norm, 'k-', lw=2, label='homogeneous')
    ax.plot(x/1000, uz_m1/norm, 'b-',lw=2, label='M1')
    ax.plot(x/1000, uz_m2/norm, 'r-',lw=2, label='M2')
    plt.xlabel('Distance [km]')
    plt.ylabel('Normalized Displacement')
    plt.title('Elastic Layer Test; Depth = {}km'.format(depth/1e3))
    plt.legend()
    plt.show()
  
  
  
def mogi_DN2009_shell():
    """
    DelNegro2009 Mogi source surrounded by viscoelastic shell in an elastic
    halfspace, compared to analytic solution in Segall 2010 Ch.7
    """
    surfaceh5 = '/home/scott/research/models/pylith/3d/delnegro2009/output_shell2/surface.h5'
    pointsh5 = '/home/scott/research/models/pylith/3d/delnegro2009/output_shell2/points.h5'
    
    #Get analytic solution
    x = 1e3*np.loadtxt('/home/scott/research/models/pylith/3d/delnegro2009/output_points.txt',usecols=[0])
    y = np.zeros_like(x)
    params = dict(xoff = 0,
                    yoff = 0,
                    d = 4e3, #m
                    dP = 100e6, #Pa
                    a = 700, #m
                    nu = 0.25,
                    mu = 30e9)
    
    fig = plt.figure()
    plt.suptitle('DelNegro Fig 4', fontsize=16)
    # NOTE: 0,1,2,3 because output skip=3
    #timesteps = [0,1,2,3]
    timesteps = [0,10,20,40]
    # Loop through output points
    for i,tstep in enumerate(timesteps):
        print(i, tstep)
        x_fem, ur_fem, uz_fem,time = pp.extract_points_visco(pointsh5,tstep)
        tau0 = 2e16 / (0.5*30e9)
        time_days = time / 8.64e4
        time_maxwell = time / tau0
        
        ur,uz = m.calc_mogi_dp(x,y,**params)
        
        ax = fig.add_subplot(2,2,i+1)
        ax.plot(x, uz, 'b-', lw=2, label='analytic')
        ax.plot(x, ur, 'g-', lw=2)
        ax.plot(x, uz_fem, marker='o', ls='None', lw=2, mec='b', mfc='None',label='FEM')
        ax.plot(x, ur_fem, marker='o', ls='None', lw=2, mec='g', mfc='None')
        #ax.set_ylim(-0.01,0.07)
        ax.set_ylim(0,0.1)
        ax.set_title('{:g} days, t/tau={:g}'.format(time_days,time_maxwell))
        ax.grid(True)
        
        if i==2:
            ax.set_xlabel('Radial distance [m]')
            ax.set_ylabel('displacement [m]')
        
        # Statistics & Graph
        print("time={}\n".format(time_days))
        calc_stats(x,ur,uz,ur_fem,uz_fem)
    
    # Residual plots for last timestep
    #plot_result(x,ur,uz,ur_fem,uz_fem)   

    #plt.xlim(0,35000)
    plt.legend()



def mogi_uturuncu_moat_shell():
    """
    Test viscoelastic shell in large basic uturuncu domain (200x200x100km, source
    at 10km)
    """    
    surfaceh5 = '/home/scott/research/models/pylith/3d/uturuncu_moat/output/shell/surface.h5'
    pointsh5 = '/home/scott/research/models/pylith/3d/uturuncu_moat/output/shell/points.h5'
    
    #Get analytic solution
    x = 1e3*np.loadtxt('/home/scott/research/models/pylith/3d/uturuncu_moat/output_points.txt',usecols=[0])
    y = np.zeros_like(x)
    # WARNING: a & b have to be floats!
    params = dict(xoff = 0,
                    yoff = 0,
                    d = 10e3, #m
                    dP = 100e6, #Pa
                    a = 1000.0, #m
                    b = 1200.0,
                    nu = 0.25,
                    mu = 30e9,
                    eta = 2e16)
    
    fig = plt.figure()
    plt.suptitle('Viscoelastic Shell FEM vs. Analytic Solution', fontsize=16)
    # NOTE: 0,1,2,3 because output skip=3
    #timesteps = [0,1,2,3]
    timesteps = [0,5,10,15]
    # Loop through output points
    for i,tstep in enumerate(timesteps):
        print(i, tstep)
        x_fem, ur_fem, uz_fem,time = pp.extract_points_visco(pointsh5,tstep)
        tau0 = params['eta'] / params['mu']
        time_days = time / 8.64e4
        time_maxwell = time / tau0
        
        ur, uz = m.calc_mogi_viscoshell(x,y,time,**params)
        
        ax = fig.add_subplot(2,2,i+1)
        ax.plot(x, uz, 'b-', lw=2, label='analytic')
        ax.plot(x, ur, 'g-', lw=2)
        ax.plot(x, uz_fem, marker='o', ls='None', lw=2, mec='b', mfc='None',label='FEM')
        ax.plot(x, ur_fem, marker='o', ls='None', lw=2, mec='g', mfc='None')
        #ax.set_ylim(-0.01,0.07)
        ax.set_title('{:g} days, t/tau={:g}'.format(time_days,time_maxwell))
        ax.grid(True)
        
        if i==2:
            ax.set_xlabel('Radial distance [m]')
            ax.set_ylabel('displacement [m]')
        
        # Statistics & Graph
        print("time={}\n".format(time_days))
        calc_stats(x,ur,uz,ur_fem,uz_fem)
    
    # Residual plots for last timestep
    #plot_result(x,ur,uz,ur_fem,uz_fem)   

    #plt.xlim(0,35000)
    plt.legend()
    
    

def uturuncu_uplift(step,depth,dP):
    """
    Show benchmark accuracy for variety of depths
    """
    #UTURNCU MESH
    surfaceh5 = '/home/scott/research/models/pylith/3d/uturuncu_uplift/output/step0{}/surface.h5'.format(step)
    pointsh5 = '/home/scott/research/models/pylith/3d/uturuncu_uplift/output/step0{}/points.h5'.format(step)
    points = '/home/scott/research/models/pylith/3d/uturuncu_uplift/output_points.txt'
    
    #Get analytic solution
    #x = np.linspace(0,25e3,1e2)
    #y = np.zeros_like(x)
    x = 1e3*np.loadtxt(points,usecols=[0])
    y = np.zeros_like(x)
    
    params = dict(xoff = 0.0, #-2150.0,
                  yoff = 0.0, #250.0,
                  d = depth,
                  dP = dP,
                  a = 2000.0,
                  mu = 30e9,
                  nu = 0.25)
    #ur,uz = m.calc_mogi_dp(x,y,d=4e3,dP=100e6,a=700,mu=30e9,nu=0.25)
    ur,uz = m.calc_mogi_dp(x,y,**params)
    
    # Get FEM solution
    #verts, data, tris = load_h5(surfaceh5)
    #X, R, Z = extract_profile_axisym(verts, data, elmtsize=500)
    x_fem, ur_fem, uz_fem = pp.extract_points(pointsh5)
    
    # Statistics & Graph
    calc_stats(x,ur,uz,ur_fem,uz_fem)
    plot_result(x,ur,uz,ur_fem,uz_fem,params,plotcomp=False)
    
    

    
def mogi_reservoir_depth(step,depth,dP):
    """
    Show benchmark accuracy for variety of depths
    """
    #UTURNCU MESH
    surfaceh5 = '/home/scott/research/models/pylith/3d/uturuncu_cylinder/output/step0{}/surface.h5'.format(step)
    pointsh5 = '/home/scott/research/models/pylith/3d/uturuncu_cylinder/output/step0{}/points.h5'.format(step)
    points = '/home/scott/research/models/pylith/3d/uturuncu_cylinder/output_points.txt'
    
    #Get analytic solution
    #x = np.linspace(0,25e3,1e2)
    #y = np.zeros_like(x)
    x = 1e3*np.loadtxt(points,usecols=[0])
    y = np.zeros_like(x)
    
    params = dict(xoff = 0.0, #-2150.0,
                  yoff = 0.0, #250.0,
                  d = depth,
                  dP = dP,
                  a = 2000.0,
                  mu = 30e9,
                  nu = 0.25)
    #ur,uz = m.calc_mogi_dp(x,y,d=4e3,dP=100e6,a=700,mu=30e9,nu=0.25)
    ur,uz = m.calc_mogi_dp(x,y,**params)
    
    # Get FEM solution
    #verts, data, tris = load_h5(surfaceh5)
    #X, R, Z = extract_profile_axisym(verts, data, elmtsize=500)
    x_fem, ur_fem, uz_fem = pp.extract_points(pointsh5)
    
    # Statistics & Graph
    calc_stats(x,ur,uz,ur_fem,uz_fem)
    plot_result(x,ur,uz,ur_fem,uz_fem,params,plotcomp=True)
    
