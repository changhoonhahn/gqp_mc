import os,sys
import numpy as np
# -- provabgs --
from provabgs import models as Models

dat_dir = '/Users/chahah/data/arcoiris/provabgs_cnf/'

ftheta = os.path.join(dat_dir, 'train.set0.thetas.npy')
thetas = np.load(ftheta)

fseds = os.path.join(dat_dir, 'train.set0.seds.npy')
seds = np.load(fseds)

m_nmf = Models.NMF(burst=True, emulator=False)
m_nmf._ssp_initiate()
print('----------------------------') 
print(m_nmf._ssp.libraries)
print() 

other_waves, other_seds = [], []
for i in range(thetas.shape[0]):
    theta = thetas[i,:-1] # SPS parameters
    zred = thetas[i,-1] # redshift

    other_w, other_sed = m_nmf.sed(theta, zred)

    other_waves.append(other_w) 
    other_seds.append(other_sed)

other_waves = np.array(other_waves)
other_seds = np.array(other_seds)

np.save(os.path.join(dat_dir, 'train.set0.%s_%s.waves.npy' %
    (m_nmf._ssp.spec_library.decode("utf-8"), m_nmf._ssp.isoc_library.decode("utf-8"))), other_waves)
np.save(os.path.join(dat_dir, 'train.set0.%s_%s.seds.npy' %
    (m_nmf._ssp.spec_library.decode("utf-8"), m_nmf._ssp.isoc_library.decode("utf-8"))), other_seds)
