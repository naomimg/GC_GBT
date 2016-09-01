import numpy as np
import pylab
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from matplotlib import rc
from scipy.signal import resample
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D

rc('mathtext',default='regular')

def get_xy_data(filename=None):
    """
    Utility function to retrieve x,y vectors form an ASCII data file.
    It return the first column as x, the second as y.
    Usage:
        x,y = get_xy_data("mydata.txt")
    """
    data = np.loadtxt(filename, usecols=(3,4,5))
    return data

# Read in the data
dat = get_xy_data("clouds_GBT_coord.txt")
print np.shape(dat)
lon = dat[:,0]
lat = dat[:,1]
vel = dat[:,2]
for i in range(len(lon)):
    if lon[i]>180.0:
        lon[i]=lon[i]-360.0


n, bins, patches = pylab.hist(vel, 30, facecolor='blue')
pylab.show()
print n, bins



nclouds = len(lon)
dist_index = np.zeros(nclouds)
derived_cloud_R = np.zeros(nclouds)
derived_cloud_z = np.zeros(nclouds)
derived_cloud_x = np.zeros(nclouds)
derived_cloud_y = np.zeros(nclouds)
derived_cloud_theta = np.zeros(nclouds)
derived_VLSR = np.zeros(nclouds)

dist_max = 25.0
R0 = 8.5
Vw = 350.
V0 = -220.0
vtheta = 0.0

for i in range(len(lon)):
    longr = np.radians(lon[i])
    latr = np.radians(lat[i])
    VLSR = vel[i]
    #vtheta = clouds_vtheta[i]
    distances = np.arange(0.05,dist_max,0.10)
    zs = distances * np.sin(latr)
    rp = distances * np.cos(latr)
    xs = rp *np.cos(longr) - R0
    ys = rp * np.sin(longr)
    thetas = np.arctan2(ys,xs)
    theta_deg = np.degrees(thetas)
    Rs = np.sqrt(xs**2 + ys**2)
    phis = np.arctan2(zs,Rs)
    phi_deg = np.degrees(phis)
    VRs = Vw * np.cos(phis)
    Vzs = Vw * np.sin(phis)
    VLSRs = (np.cos(latr)*( (R0*np.sin(longr))*(V0/R0-vtheta/Rs) + VRs*np.cos(longr-thetas)) + Vzs*np.sin(latr))
    dist_index = min(range(len(VLSRs)), key = lambda j: abs(VLSRs[j]-VLSR))
    derived_VLSR[i] = VLSRs[dist_index]
    derived_cloud_R[i] = Rs[dist_index]
    derived_cloud_x[i] = xs[dist_index]
    derived_cloud_y[i] = ys[dist_index]
    derived_cloud_z[i] = zs[dist_index]
    derived_cloud_theta[i] = thetas[dist_index]
    print i,abs(VLSR-VLSRs[dist_index])


fig=pylab.figure(1)
fig.clf()
ax = Axes3D(fig)
ax.scatter(derived_cloud_x,derived_cloud_y,derived_cloud_z,marker='o',s=40, c=vel)