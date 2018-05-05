from astropy import units
from matplotlib import pyplot
import numpy as np
import numpy.ma as ma
from numpy import sum
#import seaborn
#import corner
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator 
from scipy.interpolate import RectBivariateSpline
#from matplotlib.mlab import griddata
from scipy.interpolate import griddata
from scipy import interpolate
import sys


Teff, logg, M, Age = np.loadtxt("Bergeron_WDgrid.txt", unpack=True)

tlist = Teff[0:60]
glist = logg[0::60]

tableM = np.zeros([len(glist), len(tlist)])
tableage = np.zeros([len(glist), len(tlist)])

for i in xrange(0, len(glist)):
	for j in xrange(0, len(tlist)):
		tableM[i][j] = M[i*(len(tlist))+j]
		tableage[i][j] = Age[i*(len(tlist))+j]

#interpMo = RegularGridInterpolator((glist,tlist), tableM, bounds_error=False, fill_value=None)
interpage = RegularGridInterpolator((glist,tlist), tableage, bounds_error=False, fill_value=None)

t_He, logg_He, M_He, Ro_He, age_He = np.loadtxt('../Natalie/mygrid.txt', unpack=True)

#modage_He = griddata((t_He, logg_He), age_He, (t1, g1), method='linear', fill_value=1e20)
#modmass_He = griddata((t_He, logg_He), M_He, (modT_He, modg_He), method='linear', fill_value=1e20)

g1, t1, g2, t2, dist = np.loadtxt('posterior_wd2.txt', unpack=True)

modage_He = griddata((t_He, logg_He), age_He, (np.log10(t1), g1), method='cubic', fill_value=1e20)


#zi = griddata((t_He, logg_He), age_He, (t1, g1), method='linear')
ziage = np.array(zip(modage_He))
heage = ziage.flatten()
gridage = heage[np.where(heage < 1e20)]

agelist2 = np.zeros(len(g2))

for i in xrange(0, len(g2)):
	wdvals2 = np.array([g2[i],t2[i]])
	ageval2 = interpage(wdvals2)[0]
	agelist2[i] = ageval2 / 10.0**6.

print "age1, .16, .5, .84:"
print(np.nanpercentile(gridage, 50, axis=0) - np.nanpercentile(gridage, 16, axis=0))
print(np.nanpercentile(gridage, 50, axis=0))
print(np.nanpercentile(gridage, 84, axis=0) - np.nanpercentile(gridage, 50, axis=0))

print "age2, .16, .5, .84:"
print(np.nanpercentile(agelist2, 50, axis=0) - np.nanpercentile(agelist2, 16, axis=0))
print(np.nanpercentile(agelist2, 50, axis=0))
print(np.nanpercentile(agelist2, 84, axis=0) - np.nanpercentile(agelist2, 50, axis=0))

pyplot.rc('font', size=22)          # controls default text sizes
pyplot.rc('axes', titlesize=22)     # fontsize of the axes title
pyplot.rc('axes', labelsize=26)    # fontsize of the x and y labels
pyplot.rc('xtick', labelsize=22)    # fontsize of the tick labels
pyplot.rc('ytick', labelsize=22)    # fontsize of the tick labels
pyplot.rc('legend', fontsize=14)    # legend fontsize
pyplot.rc('figure', titlesize=30)  # fontsize of the figure title

fig1 = pyplot.figure(figsize=(6,5))
f1name = "agehist1.pdf"
ax1 = fig1.add_subplot(111)
#ax1 = sns.distplot(masslist1)
#ax1.hist(masslist1, bins=25)
ax1.hist(gridage[~np.isnan(gridage)], bins=50)
ax1.axvline(x=(np.nanpercentile(gridage, 16, axis=0)),ls='--',color='#3a516c')
ax1.axvline(x=(np.nanpercentile(gridage, 50, axis=0)),ls='--',color='#3a516c')
ax1.axvline(x=(np.nanpercentile(gridage, 84, axis=0)),ls='--',color='#3a516c')
#ax1.axvline(x=0.42, ymin=0, ymax=1, color='black')
#ax1.axvline(x=(0.42+0.01), ymin=0, ymax=1, color='black', ls='--')
#ax1.axvline(x=(0.42-0.03), ymin=0, ymax=1, color='black', ls='--')
ax1.set_xlabel("WD Age (Myr)")
ax1.set_ylabel("N")
fig1.savefig(f1name, dpi=300)


fig2 = pyplot.figure(figsize=(6,5))
f2name = "agehist2.pdf"
ax2 = fig2.add_subplot(111)
#ax2 = sns.distplot(masslist2)
ax2.hist(agelist2, bins=50)
ax2.axvline(x=(np.nanpercentile(agelist2, 16, axis=0)),ls='--',color='#3a516c')
ax2.axvline(x=(np.nanpercentile(agelist2, 50, axis=0)),ls='--',color='#3a516c')
ax2.axvline(x=(np.nanpercentile(agelist2, 84, axis=0)),ls='--',color='#3a516c')
ax2.set_xlabel("WD Age (Myr)")
ax2.set_ylabel("N")
fig2.savefig(f2name, dpi=300)

cm = pyplot.cm.get_cmap('viridis')

# Get the histogramp
Y1,X1 = np.histogram(gridage, 50, normed=False)
x_span1 = X1.max()-X1.min()
C1 = [cm(((x-X1.min())/x_span1)) for x in X1]


print(X1[np.argmax(Y1)])

fig3 = pyplot.figure(figsize=(8,6))
f3name = "agehist1new.pdf"
ax3 = fig3.add_subplot(111)
ax3.bar(X1[:-1],Y1,color=C1,width=X1[1]-X1[0])
ax3.set_xlabel("WD Age (Myr)")
ax3.set_ylabel("N")
pyplot.tight_layout()
fig3.savefig(f3name, dpi=900)

Y2,X2 = np.histogram(agelist2, 50, normed=False)
x_span2 = X2.max()-X2.min()
C2 = [cm(((x-X2.min())/x_span2)) for x in X2]

fig4 = pyplot.figure(figsize=(8,6))
f4name = "agehist2new.pdf"
ax4 = fig4.add_subplot(111)
ax4.bar(X2[:-1],Y2,color=C2,width=X2[1]-X2[0])
ax4.axvline(x=(np.nanpercentile(agelist2, 16, axis=0)),ls='--',color='#a1a4a8')
ax4.axvline(x=(np.nanpercentile(agelist2, 50, axis=0)),ls='--',color='#a1a4a8')
ax4.axvline(x=(np.nanpercentile(agelist2, 84, axis=0)),ls='--',color='#a1a4a8')
ax4.set_xlabel("WD Age (Myr)")
ax4.set_ylabel("N")
pyplot.tight_layout()
fig4.savefig(f4name, dpi=900)
