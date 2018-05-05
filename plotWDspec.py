from astropy.io import ascii
from astropy import units
import numpy
import pylab as plt
import numpy as np
import numpy.ma as ma
from numpy import sum
from matplotlib import pyplot
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

w1, f1, stdev1, err1 = np.loadtxt('../Natalie/lcb201010_x1dsum.txt.bin30.unred.newerrors', unpack=True)
add_disp = 9.1e-18
#adding intrinsic dispersion
toterr1 = np.sqrt(err1**2.0 + add_disp**2.0)
w1_1 = ma.masked_less(w1, 1141)
w1_2 = ma.masked_inside(w1_1, 1178., 1250.)
w1_3 = ma.masked_inside(w1_2, 1292., 1318.)
w1_4 = ma.masked_inside(w1_3, 1348., 1367.)
w1_5 = ma.masked_inside(w1_4, 1332., 1340.)
dataw1 = ma.masked_greater(w1_5, 1725.)
dataw1c = 1.0*dataw1.compressed()

#w2, f2, err2 = np.loadtxt('lcb202010_x1dsum.txt.bin30.unred', unpack=True)
w2, f2, stdev2, err2 = np.loadtxt('../Natalie/lcb202010_x1dsum.txt.bin30.unred.newerrors', unpack=True)
#adding intrinsic dispersion 
toterr2 = np.sqrt(err2**2.0 + add_disp**2.0)
w2_1 = ma.masked_less(w2, 1142)
w2_2 = ma.masked_inside(w2_1, 1182., 1250.)
w2_3 = ma.masked_inside(w2_2, 1290., 1318.)
w2_4 = ma.masked_inside(w2_3, 1348., 1368.)
w2_5 = ma.masked_inside(w2_4, 1330., 1340.)
dataw2 = ma.masked_greater(w2_5, 1533.)
dataw2c = 1.0*dataw2.compressed()

fig1 = pyplot.figure(figsize=(10,5))
fig2 = pyplot.figure(figsize=(10,5))


f1name = "WD1spectrum_hst_nomod.pdf"
f2name = "WD2spectrum_hst_nomod.pdf"

linecolor = "#9ca8b5"
alphaval = 0.2
width = 2.0
xlimits1 = (1100,1850)
xlimits2 = (1100,1850)
#xlimits2 = (1125,1750)
ylimits = (0,2.5e-16)
modwidth = 1.25

ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)

ax1.errorbar(w1, f1, yerr=toterr1, fmt='.', color='#75a3d8', linewidth=0.5, markersize=4, capsize=0)
ax2.errorbar(w2, f2, yerr=toterr2, fmt='.', color='#75a3d8', linewidth=0.5, markersize=4, capsize=0)

ax1.axvspan(1100., 1141., color='#a8a8a8', alpha=0.5)
ax1.axvspan(1178., 1250., color='#a8a8a8', alpha=0.5)
ax1.axvspan(1292., 1318., color='#a8a8a8', alpha=0.5)
ax1.axvspan(1348., 1367., color='#a8a8a8', alpha=0.5)
ax1.axvspan(1332., 1340., color='#a8a8a8', alpha=0.5)
ax1.axvspan(1725., 1900., color='#a8a8a8', alpha=0.5)


ax2.axvspan(1100., 1142., color='#a8a8a8', alpha=0.5)
ax2.axvspan(1182., 1250., color='#a8a8a8', alpha=0.5)
ax2.axvspan(1290., 1318., color='#a8a8a8', alpha=0.5)
ax2.axvspan(1348., 1368., color='#a8a8a8', alpha=0.5)
ax2.axvspan(1330., 1340., color='#a8a8a8', alpha=0.5)
ax2.axvspan(1533., 1900., color='#a8a8a8', alpha=0.5)


ax1.set_xlim(xlimits1)
ax2.set_xlim(xlimits2)
ax1.set_ylim(ylimits)
ax2.set_ylim(ylimits)

ax1.set_xlabel(r'Wavelength ($\AA$)')
ax2.set_xlabel(r'Wavelength ($\AA$)')

ax1.set_ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)')
ax2.set_ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)')

fig1.savefig(f1name, dpi=1000)
fig2.savefig(f2name, dpi=1000)

