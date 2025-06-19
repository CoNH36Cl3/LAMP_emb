import numpy as np
from pyscf import gto, scf
from embed_sim_dmrg import ssdmet
from embed_sim_dmrg.dmrg_plugin import dmrgscf_mixer , dmrgsoc_2step

title = 'CoSH4'

def get_mol(dihedral):
     mol = gto.M(atom = '''
                Co             
                S                  1            2.30186590
                S                  1            2.30186590    2            109.47122060
                S                  1            2.30186590    3            109.47122065    2            -120.00000001                  0
                S                  1            2.30186590    4            109.47122060    3            120.00000001                   0
                H                  2            1.30714645    1            109.47121982    4            '''+str(-60-dihedral)+'''      0
                H                  4            1.30714645    1            109.47121982    3            '''+str(60+dihedral)+'''       0
                H                  5            1.30714645    1            109.47121982    4            '''+str(-180+dihedral)+'''     0
                H                  3            1.30714645    1            109.47121982    4            '''+str(60-dihedral)+'''       0
     ''',
     basis={'default':'def2tzvp','s':'6-31G*','H':'6-31G*'}, symmetry=0 ,spin = 3,charge = -2,verbose= 4)

     return mol

mol = get_mol(0)

mf = scf.rohf.ROHF(mol).x2c().density_fit()

chk_fname = title + '_rohf.chk'

mf.chkfile = chk_fname
mf.init_guess = 'chk'
mf.level_shift = .1
mf.max_cycle = 1000
mf.max_memory = 100000
mf.kernel()

mydmet = ssdmet.SSDMET(mf, title=title, imp_idx='Co.*').density_fit()
# if impurity is not assigned, the orbitals on the first atom is chosen as impurity
mydmet.build()

ncas, nelec, es_mo = mydmet.avas('Co 3d', minao='def2tzvp', threshold=0.5)

settings={"quicktest":False,"ptmpsdim":500,"threads":32,"memory":64,"maxM":1500,"tol":1E-10,"scratch":"/tmp","extrakeyword":None,"maxiter":None,"two_dot_to_one_dot":None,"scheduleSweeps":None,"scheduleMaxMs":None,"scheduleTols":None,"scheduleNoises":None}

es_cas = dmrgscf_mixer.dmrgscf_mixer(mf,mydmet.es_mf, ncas, nelec,settings=settings)
import h5py

es_cas.kernel(es_mo)

mydmet2 = ssdmet.SSDMET(mf, title=title, imp_idx='Co.*')
mydmet2.build()
#es_mf need to be defined on a ssdmet without density fit for eri generation 

soc_settings={"memory":64,"threads":32,"scratch":"/tmp","reordering":True,"thres_of_itgs":1e-10,"bond_dim_init":250,"n_roots_for_soc":16,"bond_dims_schedule": [500, 500, 500, 500, 500, 1000, 1000, 1000, 1000, 1000, 1500, 1500, 1500, 1500, 1500],"noise_schedule": [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 0],"thrd_schedule": [1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-8],"dav_max_iter":400,"n_sweeps":20}
soc2=dmrgsoc_2step.DMRGSOC_2STEP(title,mf,es_cas,dmrgsoc_settings=soc_settings,dmet_settings=True,es_orb=mydmet.es_orb,es_mf=mydmet2.es_mf)
soc2.kernel()

es_ecorr = dmrgscf_mixer.dmrgscf_nevpt2(mf,es_cas,settings)
soc22=dmrgsoc_2step.DMRGSOC_2STEP(title,mf,es_cas,dmrgsoc_settings=soc_settings,dmet_settings=True,es_orb=mydmet.es_orb,es_mf=mydmet2.es_mf,ecorr=es_ecorr)
soc22.kernel()
