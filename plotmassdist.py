#from MCMCfit import fitfunc
from astropy.io import ascii
from astropy import units
import numpy
import corner
import pylab as plt
import numpy as np
import numpy.ma as ma
from numpy import sum
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator 
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import griddata
import sys
#import seaborn as sns
from matplotlib import pyplot



Rsun = units.Rsun.to(units.cm) #6.957e10 #cm
parsec = units.pc.to(units.cm) #3.086e18 #cm

#Eclipsing binary distance (Meibom+2009)
#dobs = 1770.0 #pc
#dobserr = 75.0

#Gaia DR2 distance from Christian Knigge
dobs = 1942.0 #pc
dobserr	= 15.0 

#Photometric distance (Sarajedini+2001)
#dobs = 1940.0 #pc
#dobserr = 71
print("Reading in posterior...")
g1, t1, g2, t2 = np.loadtxt('posterior_setdist.txt', unpack=True)
t1 = np.log10(t1)
#t2 = np.log10(t2)

print("Reading in WD tables...")
temp, logg, Mo, Ro = np.loadtxt('../Natalie/wdtable.txt', unpack=True)
t_He, logg_He, Mo_He, Ro_He, age = np.loadtxt('../Natalie/mygrid.txt', unpack=True)

print("Creating grids of WD parameters...")
#let's make some tables...
glist = [7.0, 7.5, 8.0, 8.5, 9.0, 9.5]
tlist = temp[0:60]

#I think we'll only be looking up either M or R at a time. 
tableMo = np.zeros([len(glist), len(tlist)])
tableRo = np.zeros([len(glist), len(tlist)])

for i in range(0, len(glist)):
	for j in range(0, len(tlist)):
		tableMo[i][j] = Mo[i*(len(tlist))+j]
		tableRo[i][j] = Ro[i*(len(tlist))+j]

interpMo = RegularGridInterpolator((glist,tlist), tableMo, bounds_error=True)#, fill_value=None)
interpRo = RegularGridInterpolator((glist,tlist), tableRo, bounds_error=True)#, fill_value=None)

#temp_He = 10.0**(t_He)
temp_He = t_He
tlist_He = temp_He[0:299]
glist_He = logg_He[0::300]

tableMo_He = np.zeros([len(glist_He), len(tlist_He)])
tableRo_He = np.zeros([len(glist_He), len(tlist_He)])

for i in range(0, len(glist_He)):
	for j in range(0, len(tlist_He)):
		tableMo_He[i][j] = Mo_He[i*(len(tlist_He))+j]
		tableRo_He[i][j] = Ro_He[i*(len(tlist_He))+j]

interpMo_He = RegularGridInterpolator((glist_He,tlist_He), tableMo_He, bounds_error=True)#, fill_value=None)
interpRo_He = RegularGridInterpolator((glist_He,tlist_He), tableRo_He, bounds_error=True)#, fill_value=None)

masslist1 = np.zeros(len(g1))
radlist1 = np.zeros(len(g1))
masslist2 = np.zeros(len(g2))
radlist2 = np.zeros(len(g2))

#print("Finding corresponding mass and radius values for He-core WDs...")
#for i in xrange(0, len(g1)):
#	wdvals1 = np.array([g1[i],t1[i]])
#	Mval1 = interpMo_He(wdvals1)[0]
#	Rval1 = interpRo_He(wdvals1)[0]
#	masslist1[i] = Mval1
#	radlist1[i] = Rval1
	
print("Finding corresponding mass and radius values for CO-core WDs...")
for i in range(0, len(g2)):
	wdvals2 = np.array([g2[i],t2[i]])
	Mval2 = interpMo(wdvals2)[0]
	#Mval2 = interpMo_He(wdvals2)[0]
	Rval2 = interpRo(wdvals2)[0]
	#Rval2 = interpRo_He(wdvals2)[0]
	masslist2[i] = Mval2
	radlist2[i] = Rval2

#print masslist1
#print masslist2

print("Creating grid for He-core WD values...")
#t_He, logg_He, Mo_He, Ro_He, age = np.loadtxt('../Natalie/mygrid.txt', unpack=True)

zi = griddata((t_He, logg_He), Mo_He, (t1, g1), method='linear')
zimass = np.array(list(zip(zi)))
gridmass = zimass.flatten()
'''
filename = "massdist_newdist.pdf"

fsize = 12

plt.subplot(2,1,1)
plt.subplots_adjust(hspace=.5)
plt.xlabel('Mass (Msun)', fontsize=fsize)
plt.ylabel('Distance (pc)', fontsize=fsize)
plt.plot(masslist1, dist, '.', color="black", linestyle="None")

plt.subplot(2,1,2)
plt.xlabel('Mass (Msun)', fontsize=fsize)
plt.ylabel('Distance (pc)', fontsize=fsize)
plt.plot(masslist2, dist, '.', color="black", linestyle="None")

plt.savefig(filename)
'''

pyplot.rc('font', size=22)          # controls default text sizes
pyplot.rc('axes', titlesize=22)     # fontsize of the axes title
pyplot.rc('axes', labelsize=26)    # fontsize of the x and y labels
pyplot.rc('xtick', labelsize=22)    # fontsize of the tick labels
pyplot.rc('ytick', labelsize=22)    # fontsize of the tick labels
pyplot.rc('legend', fontsize=14)    # legend fontsize
pyplot.rc('figure', titlesize=30)  # fontsize of the figure title

print("Plotting mass histogram for WD 1...")

fig1 = pyplot.figure(figsize=(6,5))
f1name = "masshist1_setdist.pdf"
ax1 = fig1.add_subplot(111)
#ax1 = sns.distplot(masslist1)
#ax1.hist(masslist1, bins=25)
ax1.hist(gridmass[~np.isnan(gridmass)], bins=25, color='#8c939e')
ax1.axvline(x=(np.nanpercentile(gridmass, 16, axis=0)),ls='--',color='#3a516c')
ax1.axvline(x=(np.nanpercentile(gridmass, 50, axis=0)),ls='--',color='#3a516c')
ax1.axvline(x=(np.nanpercentile(gridmass, 84, axis=0)),ls='--',color='#3a516c')
#ax1.axvline(x=0.42, ymin=0, ymax=1, color='black')
#ax1.axvline(x=(0.42+0.01), ymin=0, ymax=1, color='black', ls='--')
#ax1.axvline(x=(0.42-0.03), ymin=0, ymax=1, color='black', ls='--')
ax1.set_xlabel("WD Mass (solar masses)")
ax1.set_ylabel("N")
pyplot.tight_layout()
fig1.savefig(f1name, dpi=300)

print("mass1, .16, .5, .84:")
print((np.nanpercentile(gridmass, 50, axis=0) - np.nanpercentile(gridmass, 16, axis=0)))
print((np.nanpercentile(gridmass, 50, axis=0)))
print((np.nanpercentile(gridmass, 84, axis=0) - np.nanpercentile(gridmass, 50, axis=0)))

print("mass2, .16, .5, .84:")
print((np.nanpercentile(masslist2, 50, axis=0) - np.nanpercentile(masslist2, 16, axis=0)))
print((np.nanpercentile(masslist2, 50, axis=0)))
print((np.nanpercentile(masslist2, 84, axis=0) - np.nanpercentile(masslist2, 50, axis=0)))

print("Plotting mass histogram for WD 2...")

fig2 = pyplot.figure(figsize=(6,5))
f2name = "masshist2_setdist.pdf"
ax2 = fig2.add_subplot(111)
#ax2 = sns.distplot(masslist2)
ax2.hist(masslist2, bins=25)
ax2.axvline(x=(np.nanpercentile(masslist2, 16, axis=0)),ls='--',color='#3a516c')
ax2.axvline(x=(np.nanpercentile(masslist2, 50, axis=0)),ls='--',color='#3a516c')
ax2.axvline(x=(np.nanpercentile(masslist2, 84, axis=0)),ls='--',color='#3a516c')
ax2.set_xlabel("WD Mass (solar masses)")
ax2.set_ylabel("N")
pyplot.tight_layout()
fig2.savefig(f2name, dpi=300)
'''
fig3 = pyplot.figure(figsize=(6,5))
f3name = "radhist1_newdist.pdf"
ax3 = fig3.add_subplot(111)
#ax1 = sns.distplot(masslist1)
#X2 = np.sort(masslist1)
#F2 = np.array(range(len(masslist1)))/float(len(masslist1))
# Cumulative distributions:
#ax3.plot(X2, F2)
ax3.hist(radlist1, bins=25)
ax3.set_xlabel("WD Radius (solar radii)")
ax3.set_ylabel("N")
#ax3.set_xlabel("WD Mass (solar masses)")
#ax3.set_ylabel("N")
fig3.savefig(f3name, dpi=300)

#logt_all, logg_all, M_all, Merr_all, Age_all, Age_err_all = np.loadtxt('../Natalie/Tabla_ELM.sort.dat', unpack=True)

fig4 = pyplot.figure(figsize=(6,5))
f4name = "radmass1_newdist.pdf"
ax4 = fig4.add_subplot(111)
#ax1 = sns.distplot(masslist1)
#X2 = np.sort(masslist1)
#F2 = np.array(range(len(masslist1)))/float(len(masslist1))
# Cumulative distributions:
#ax3.plot(X2, F2)
#ax4.scatter(Mo_He, Ro_He, c=10.0**t_He, s=30, cmap="PuRd", edgecolor='none', vmin=14000, vmax=16000)
ax4.scatter(t_He, logg_He, c=Ro_He, s=30, cmap="PuRd", edgecolor='none', vmin=0.01, vmax=0.03)
#ax4.scatter(masslist1, radlist1, c=t1, s=5, cmap="PuRd", edgecolor='none', vmin=14000, vmax=16000)
#ax4.scatter(t1, g1, c=masslist1, s=5, cmap="PuRd", edgecolor='none', vmin=0.3, vmax=0.5)
ax4.scatter(t1, g1, c=radlist1, s=5, cmap="PuRd", edgecolor='none', vmin=0.01, vmax=0.03)
ax4.set_xlabel("WD Temp")
ax4.set_ylabel("logg")
ax4.set_xlim([4.1,4.3])
ax4.set_ylim([7.0,7.8])
#ax3.set_xlabel("WD Mass (solar masses)")
#ax3.set_ylabel("N")
fig4.savefig(f4name, dpi=300)

temp, logg, Mo, Ro = np.loadtxt('../Natalie/wdtable.txt', unpack=True)

fig5 = pyplot.figure(figsize=(6,5))
f5name = "radmass2_newdist.pdf"
ax5 = fig5.add_subplot(111)
#ax1 = sns.distplot(masslist1)
#X2 = np.sort(masslist1)
#F2 = np.array(range(len(masslist1)))/float(len(masslist1))
# Cumulative distributions:
#ax3.plot(X2, F2)
#ax4.scatter(Mo_He, Ro_He, c=10.0**t_He, s=30, cmap="PuRd", edgecolor='none', vmin=14000, vmax=16000)
ax5.scatter(Mo, Ro, c=logg, s=30, cmap="PuRd", edgecolor='none', vmin=6.5, vmax=7.5)
#ax4.scatter(masslist1, radlist1, c=t1, s=5, cmap="PuRd", edgecolor='none', vmin=14000, vmax=16000)
ax5.scatter(masslist2, radlist2, c=g2, s=5, cmap="PuRd", edgecolor='none', vmin=6.5, vmax=7.5)
ax5.set_xlabel("WD Mass")
ax5.set_ylabel("WD Radius")
ax5.set_ylim([0.01,0.025])
ax5.set_xlim([0.4,0.8])
#ax3.set_xlabel("WD Mass (solar masses)")
#ax3.set_ylabel("N")
fig5.savefig(f5name, dpi=300)
'''
