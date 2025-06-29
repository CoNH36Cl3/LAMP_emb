"""
two step soc approach for dmrg wfn , with Hsomf 
two step is supported with NEVPT2 ,same as siso 
copied and adapted from liblan interface for dmrg
other usages are same as dmrgsoc_1step
"""
from pyblock2.driver.core import DMRGDriver, SymmetryTypes,SOCDMRGDriver
from pyblock2._pyscf.ao2mo import soc_integrals as itgsoc
from pyblock2._pyscf.ao2mo import integrals as itg
from pyscf import scf,gto
from embed_sim.dmrg_plugin.dmrgscf_mixer import read_statelis
import numpy as np
import time
default_socsettings={"memory":4,"threads":8,"scratch":"/tmp","reordering":True,"thres_of_itgs":1e-10,"bond_dim_init":250,"n_roots_for_soc":16,"bond_dims_schedule": [500, 500, 500, 500, 500, 1000, 1000, 1000, 1000, 1000, 1500, 1500, 1500, 1500, 1500],"noise_schedule": [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 0],"thrd_schedule": [1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-8],"dav_max_iter":400,"n_sweeps":20}
class DMRGSOC_2STEP():
    def __init__(self,title,mf_without_dmet,mc,statelis=None,dmrgsoc_settings=default_socsettings,dmet_settings=True,es_orb=None,es_mf=None,ecorr=None):
        self.title=title
        self.mol=mf_without_dmet.mol
        if es_mf is not None:
            self.es_mf=es_mf
        else:
            self.es_mf=mf_without_dmet
        self.mc=mc
        self.dmrgsoc_settings=default_socsettings
        self.mf_without_dmet=mf_without_dmet
        self.statelis = read_statelis(mc)
        self.dmet_settings=dmet_settings
        self.es_orb=es_orb
        e_corr_full=[]
        if ecorr is not None:
		    #a statelis of [0,10,0,20] ecorr should be a format like[[0]*10,[0]*20]
            k0=0
            for i in self.statelis:
                if i>0:
                    e_corr=np.array(ecorr[k0:k0+i])
                    k0+=i
                    e_corr_full.append(e_corr)
        else:
            k0=0
            for i in self.statelis:
                if i>0:
                    e_corr=np.zeros([i])
                    k0+=i
                    e_corr_full.append(e_corr) 
        self.ecorr=e_corr_full
        print("ecorr=",e_corr_full)
    def form_somf_itgs(self):
        #form somf integrals in orbital basis
        ncore,ncas=self.mc.ncore,self.mc.ncas
        self.es_mf.mo_coeff=self.mc.mo_coeff
        dm_ao= np.einsum('yij->ij', self.mf_without_dmet.make_rdm1(), optimize=True)
        hso_ao = itgsoc.get_somf_hsoao(self.mf_without_dmet, dmao=dm_ao, amfi=True, x2c1e=False, x2c2e=False)
        if self.dmet_settings:
            hso_dmet=np.einsum('rij,ip,jq->rpq',hso_ao,(self.es_orb.conj()),self.es_orb)
            hso_dmet_cas = np.einsum('rij,ip,jq->rpq', hso_dmet,
            self.es_mf.mo_coeff[:, ncore:ncore + ncas],
            self.es_mf.mo_coeff[:, ncore:ncore + ncas], optimize=True)
        else:
            hso_dmet_cas = np.einsum('rij,ip,jq->rpq', hso_ao,
            self.es_mf.mo_coeff[:, ncore:ncore + ncas],
            self.es_mf.mo_coeff[:, ncore:ncore + ncas], optimize=True)
        return hso_dmet_cas
    def restart_casci(self):
        self.es_mf.mo_coeff=self.mc.mo_coeff
        settings=self.dmrgsoc_settings
        ncore,ncas=self.mc.ncore,self.mc.ncas
        ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(self.es_mf, ncore, ncas, pg_symm=False)
        driver = SOCDMRGDriver(
        scratch=(settings["scratch"] ), symm_type=SymmetryTypes.SU2, n_threads=settings["threads"] ,
        stack_mem=(settings["memory"]*1E9) , mpi=True)
        if settings["reordering"]:
            idx = driver.orbital_reordering(np.abs(h1e), np.abs(g2e))
            h1e = h1e[idx][:, idx]
            g2e = g2e[idx][:, idx][:, :, idx][:, :, :, idx]
        print("idx=")
        print(idx)
        all_eners = []
        all_mpss = []
        twoss = []
        for i in range(len(self.statelis)):
            if(self.statelis[i]==0):
                continue
            driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=i, orb_sym=orb_sym, singlet_embedding=False)
            if settings["thres_of_itgs"] is not None:
                threshold=settings["thres_of_itgs"]
                h1e[np.abs(h1e) < threshold] = 0
                g2e[np.abs(g2e) < threshold] = 0
            bond_dim_init,n_roots_for_soc=settings["bond_dim_init"],settings["n_roots_for_soc"]
            mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=0)
            ket = driver.get_random_mps(tag="KET", bond_dim=bond_dim_init, nroots=self.statelis[i])
            bond_dims_schedule,noise_schedule,thrd_schedule=settings["bond_dims_schedule"],settings["noise_schedule"],settings["thrd_schedule"]
            energies = driver.dmrg(
            mpo,ket,n_sweeps=settings["n_sweeps"],bond_dims=bond_dims_schedule,noises=noise_schedule,
            thrds=thrd_schedule,iprint=0,dav_max_iter=settings["dav_max_iter"],cutoff=1E-24)
            if settings["reordering"]:
                driver.reorder_idx = idx
                # reorder_idx generates a inverse reorder after calculation ,so somf intgs dont need to reorder
            skets = []
            for j in range(len(energies)):
                sket = driver.split_mps(ket, j, "S%d-KET%d" % (i, j))
                if driver.mpi.rank == driver.mpi.root:
                    print('split mps = ', j)
                skets.append(sket)
            all_eners.append(energies)
            twoss.append(i)
            all_mpss.append(skets)
        pdms_dict = {}
        ip = 0
        for si in range(len(twoss)):
            for i, iket in enumerate(all_mpss[si]):
                jp = 0
                for sj in range(len(twoss)):
                    for j, jket in enumerate(all_mpss[sj]):
                        if i + ip < j + jp or abs(twoss[si] - twoss[sj]) > 2:
                            continue
                        if driver.mpi.rank == driver.mpi.root:
                            print('compute pdm = ', i + ip, j + jp)
                        dm = driver.get_trans_1pdm(iket, jket, soc=True, iprint=0)
                        print("dm=")
                        print(dm)
                        pdms_dict[(i + ip, j + jp)] = dm
                    jp += len(all_eners[sj])
            ip += len(all_eners[si])
        return driver,all_eners,twoss,pdms_dict
    def kernel(self):
        time_start=time.time()
        self.es_mf.mo_coeff=self.mc.mo_coeff
        ncore,ncas=self.mc.ncore,self.mc.ncas
        driver,all_eners,twoss,pdms_dict=self.restart_casci()
        hso_dmet_cas=self.form_somf_itgs()
        full_energy=all_eners
        for j in range(len(all_eners)):
            for k in range(len(all_eners[j])):
                full_energy[j][k]+=self.ecorr[j][k]
        print(full_energy)
        energies = driver.soc_two_step(full_energy, twoss, pdms_dict, hso_dmet_cas, iprint=0)
        time_end=time.time()
        print("time for soc="+str(time_end-time_start))
        energies_base=energies[0]
        delta_in_cm=[]
        au2cm = 219474.63111558527
        for energy in energies:
            delta_in_cm.append(au2cm*(energy-energies_base))
        print("delta E in cm-1")
        print(delta_in_cm)
        print("absolute E in a.u.")
        print(energies)
        return 0
