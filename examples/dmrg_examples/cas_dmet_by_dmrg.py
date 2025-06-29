import numpy as np
from pyscf import scf
from embed_sim import ssdmet, myavas, sacasscf_mixer, siso,rdiis
from embed_sim.dmrg_plugin import dmrgscf_mixer,dmrgsoc_2step

title = 'CoSPh4'
    
from pyscf import gto

mol = gto.M(atom='''
        Co   1.506499   10.764206    0.414024
        S    0.900702   12.969251    0.863740
        S   -0.630821    9.850987    0.263627
        S    2.753892   10.415545   -1.502746
        S    3.027943    9.878195    1.923416
        C    2.377311   13.975173    0.913216
        C    2.276518   15.365718    0.836418
        H    1.444112   15.762234    0.713835
        C    3.399137   16.154649    0.937832
        H    3.312247   17.078944    0.888601
        C    4.645140   15.606362    1.112844
        H    5.397607   16.149181    1.179058
        C    4.758096   14.244532    1.186689
        H    5.597454   13.867157    1.321825
        C    3.649380   13.407744    1.065829
        H    3.753648   12.483449    1.085522
        C   -0.483108    8.032887    0.187074
        C   -1.603989    7.293178    0.071383
        H   -2.422493    7.725245   -0.019692
        C   -1.574447    5.919042    0.086153
        H   -2.366884    5.440487    0.000000
        C   -0.364938    5.235392    0.231381
        H   -0.335395    4.306995    0.285534
        C    0.776797    5.988774    0.292919
        H    1.598776    5.558075    0.347072
        C    0.738565    7.384787    0.273226
        H    1.527526    7.875648    0.319995
        C    2.008897   11.246043   -2.857063
        C    0.792437   11.947467   -2.723896
        H    0.359725   11.969345   -1.900278
        C    0.238079   12.605139   -3.809910
        H   -0.559572   13.071388   -3.704557
        C    0.851522   12.577793   -5.050260
        H    0.465730   13.018063   -5.772218
        C    2.036702   11.896878   -5.208288
        H    2.450298   11.872266   -6.040521
        C    2.615389   11.247410   -4.127935
        H    3.423466   10.801671   -4.246087
        C    2.286945    9.211500    3.375947
        C    3.044626    9.114422    4.556237
        H    3.906575    9.464451    4.578390
        C    2.530237    8.503239    5.693449
        H    3.049839    8.432139    6.461437
        C    1.239051    8.001440    5.679173
        H    0.886278    7.585780    6.431899
        C    0.479633    8.125864    4.519314
        H   -0.391005    7.797712    4.514391
        C    0.962741    8.709702    3.404255
        H    0.424023    8.782168    2.648574
    ''',
    basis={'default':'def2tzvp','C':'6-31G*','H':'6-31G*'}, symmetry=0 ,spin = 3,charge = -2,verbose= 4)


mf = scf.rohf.ROHF(mol).x2c()
chk_fname = title + '_rohf.chk'

mf.chkfile = chk_fname
mf.init_guess = 'atom'
mf.level_shift = 2
mf.max_cycle = 1000
mf.diis = rdiis.RDIIS(rdiis_prop='dS',imp_idx=mol.search_ao_label(['Co.*d']),power=0.2)
mf.max_memory = 100000
mf.conv_tol=1e-8
mf.kernel()


settings={"quicktest":False,"ptmpsdim":None,"threads":32,"memory":64,"maxM":1500,"tol":1E-10,"scratch":"/tmp","extrakeyword":None,"maxiter":None,"two_dot_to_one_dot":None,"scheduleSweeps":None,"scheduleMaxMs":None,"scheduleTols":None,"scheduleNoises":None}

ncas, nelec, mo = myavas.avas(mf, 'Co 3d', canonicalize=False)
mycas = dmrgscf_mixer.dmrgscf_mixer(mf,mf, ncas, nelec, statelis=[0, 40, 0, 10],settings=settings)
cas_chk = title + '_cas_5_7.chk'
mycas.chkfile = cas_chk

from pyscf import lib
try:
    mo = lib.chkfile.load(cas_chk, 'mcscf/mo_coeff')
except IOError:
    pass
mycas.natorb = True
mycas.kernel(mo)

cas_dmet = ssdmet.SSDMET(mycas, title=title+'_cas', imp_idx='Co *')
cas_dmet.build()
cas_dmet_2 = ssdmet.SSDMET(mycas, title=title+'_cas', imp_idx='Co *')
cas_dmet_2.build()
ncas, nelec, es_mo = cas_dmet.avas(['Co 3d', 'Co 4d'], minao='def2tzvp', threshold=0.5)

es_cas = dmrgscf_mixer.dmrgscf_mixer(mf,cas_dmet.es_mf, ncas, nelec, statelis=[0, 40, 0, 10],settings=settings)
es_cas.kernel(es_mo)

print(cas_dmet.es_mf.mo_coeff.shape,es_cas.mo_coeff.shape,mf.mo_coeff.shape,cas_dmet_2.es_mf._eri.shape)

soc_settings={"memory":64,"threads":32,"scratch":"/tmp","reordering":True,"thres_of_itgs":1e-10,"bond_dim_init":250,"n_roots_for_soc":16,"bond_dims_schedule": [500, 500, 500, 500, 500, 1000, 1000, 1000, 1000, 1000, 1500, 1500, 1500, 1500, 1500],"noise_schedule": [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 0],"thrd_schedule": [1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-8],"dav_max_iter":400,"n_sweeps":20}
soc2=dmrgsoc_2step.DMRGSOC_2STEP(title,mf,es_cas,dmrgsoc_settings=soc_settings,dmet_settings=True,es_orb=cas_dmet.es_orb,es_mf=cas_dmet_2.es_mf)
print(soc2.mc.ncore,soc2.mc.ncas)
soc2.kernel()

es_ecorr = dmrgscf_mixer.dmrgscf_nevpt2(mf,es_cas,settings)
soc22=dmrgsoc_2step.DMRGSOC_2STEP(title,mf,es_cas,dmrgsoc_settings=soc_settings,dmet_settings=True,es_orb=cas_dmet.es_orb,es_mf=cas_dmet_2.es_mf,ecorr=es_ecorr)
soc22.kernel()
