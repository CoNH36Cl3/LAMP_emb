"""
one step soc approach for dmrg wfn , with Hsomf 
one step is not supported with NEVPT2 in principle 
copied and adapted from liblan interface for dmrg


WARNING 
This is not supported with ssdmet.density_fit(),you can use density_fit for casscf mo_coeff and nevpt2, 
then rebuild a mydmet without density_fit and use mydmet_nodensityfit.somf

example:
mf.kernel()
mydmet = ssdmet.SSDMET(mf, title=title, imp_idx='Co.*').density_fit()
mydmet.build()
es_cas = dmrgscf_mixer.dmrgscf_mixer(mf,mydmet.es_mf, ncas, nelec,settings=settings)
es_cas.kernel(mo_init)
mydmet2 = ssdmet.SSDMET(mf, title=title, imp_idx='Co.*')
mydmet2.build()
soc1=dmrgsoc_1step.DMRGSOC_1STEP(title,mf,es_cas,dmrgsoc_settings=soc_settings,dmet_settings=True,es_orb=mydmet.es_orb,es_mf=mydmet2.es_mf)
soc1.kernel()

"""
from embed_sim.dmrg_plugin.dmrgscf_mixer import read_statelis
from pyblock2.driver.core import DMRGDriver, SymmetryTypes,SOCDMRGDriver
from pyblock2._pyscf.ao2mo import soc_integrals as itgsoc
from pyblock2._pyscf.ao2mo import integrals as itg
from pyscf import scf,gto
import numpy as np
import time
default_socsettings={"memory":4,"threads":8,"scratch":"/tmp","reordering":True,"thres_of_itgs":1e-10,"bond_dim_init":250,"n_roots_for_soc":16,"bond_dims_schedule": [500, 500, 500, 500, 500, 1000, 1000, 1000, 1000, 1000, 1500, 1500, 1500, 1500, 1500],"noise_schedule": [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 0],"thrd_schedule": [1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-8],"dav_max_iter":400,"n_sweeps":20}
class DMRGSOC_1STEP():
    def __init__(self,title,mf_without_dmet,mc,statelis=None,dmrgsoc_settings=default_socsettings,dmet_settings=True,es_orb=None,es_mf=None):
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
    def extract_h(self):
        #extract 1e or 2e itgs and active space info from mc , itgs are representated in casscf orbs instead of atomic basis and projected into active space.
        #casci_base=self.mc._scf
        self.es_mf.mo_coeff=self.mc.mo_coeff
        ncore,ncas=self.mc.ncore,self.mc.ncas
        ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(self.es_mf, ncore, ncas, pg_symm=False)
        return ncas, n_elec, spin, ecore, h1e, g2e, orb_sym
    def form_somf_itgs(self):
        #extract somf itgs in atomic basis and transform it into casscf orb basis 
        dm_ao = np.einsum('yij->ij', self.mf_without_dmet.make_rdm1(), optimize=True)
        hso_ao = itgsoc.get_somf_hsoao(self.mf_without_dmet, dmao=dm_ao, amfi=True, x2c1e=False, x2c2e=False)
        #extract somf itgs in atomic basis
        if self.dmet_settings:
            ncore,ncas=self.mc.ncore,self.mc.ncas
            hso_dmet=np.einsum('rij,ip,jq->rpq',hso_ao,(self.es_orb.conj()),self.es_orb)
            hso_dmet_cas = np.einsum('rij,ip,jq->rpq', hso_dmet,
            self.es_mf.mo_coeff[:, ncore:ncore + ncas],
            self.es_mf.mo_coeff[:, ncore:ncore + ncas], optimize=True)
            # transform somf itg in dmet basis and then active orbs ; somf only has 1e parts 
        else:
            ncore,ncas=self.mc.ncore,self.mc.ncas
            hso_dmet_cas = np.einsum('rij,ip,jq->rpq', hso_ao,
            self.es_mf.mo_coeff[:, ncore:ncore + ncas],
            self.es_mf.mo_coeff[:, ncore:ncore + ncas], optimize=True)
            # transform somf itg in active orbs directly
        return hso_dmet_cas
    def full_h_setup(self):
        #set up full Hsomf and split alpha/beta-> (alpha,beta),since sign of somf operator are differnent in spin,this means new dmrg space will be doubled
        #split h1e without soc
        ncas, n_elec, spin, ecore, h1e, g2e, orb_sym=self.extract_h()
        hso_dmet_cas=self.form_somf_itgs()
        space_size=ncas*2
        gh1e_dmet_cas = np.zeros((space_size, space_size), dtype=complex)
        for i in range(space_size):
            for j in range(i % 2, space_size, 2):
                gh1e_dmet_cas[i, j] = h1e[i // 2, j // 2]	
        #add hsomf
        for i in range(space_size):
            for j in range(space_size):
                if i % 2 == 0 and j % 2 == 0:  # aa
                    gh1e_dmet_cas[i, j] += hso_dmet_cas[2, i // 2, j // 2] * 0.5
                elif i % 2 == 1 and j % 2 == 1:  # bb
                    gh1e_dmet_cas[i, j] -= hso_dmet_cas[2, i // 2, j // 2] * 0.5
                elif i % 2 == 0 and j % 2 == 1:  # ab
                    gh1e_dmet_cas[i, j] += (
                    hso_dmet_cas[0, i // 2, j // 2] - hso_dmet_cas[1, i // 2, j // 2] * 1j
                    ) * 0.5
                elif i % 2 == 1 and j % 2 == 0:  # ba
                    gh1e_dmet_cas[i, j] += (
                    hso_dmet_cas[0, i // 2, j // 2] + hso_dmet_cas[1, i // 2, j // 2] * 1j
                    ) * 0.5
        #split g2e 
        gg2e_dmet_cas=np.zeros((space_size, space_size, space_size, space_size), dtype=complex)
        for i in range(space_size):
            for j in range(i % 2, space_size, 2):
                for k in range(space_size):
                    for l in range(k % 2, space_size, 2):
                        gg2e_dmet_cas[i, j, k, l] = g2e[i // 2, j // 2, k // 2, l // 2]
        return  gh1e_dmet_cas,gg2e_dmet_cas
    def kernel(self):
		#set up 1-step dmrg calculation and output
        ncas, n_elec, spin, ecore, h1e, g2e, orb_sym=self.extract_h()
        settings=self.dmrgsoc_settings
        driver= DMRGDriver(scratch=(settings["scratch"]), symm_type=SymmetryTypes.SGFCPX, stack_mem=(settings["memory"]*1E9), n_threads=settings["threads"])
        gh1e_dmet_cas,gg2e_dmet_cas=self.full_h_setup()
        if settings["reordering"]==True:
            idx = driver.orbital_reordering(np.abs(gh1e_dmet_cas), np.abs(gg2e_dmet_cas))
            print('reordering = ', idx)
            gh1e_dmet_cas = gh1e_dmet_cas[idx][:, idx]
            gg2e_dmet_cas = gg2e_dmet_cas[idx][:, idx][:, :, idx][:, :, :, idx]
        orb_sym = [orb_sym[x // 2] for x in range(ncas * 2)]
        driver.initialize_system(n_sites=(ncas*2), n_elec=n_elec, spin=0, orb_sym=orb_sym)
        if settings["thres_of_itgs"] is not None:
            threshold=settings["thres_of_itgs"]
            gh1e_dmet_cas[np.abs(gh1e_dmet_cas) < threshold] = 0
            gg2e_dmet_cas[np.abs(gg2e_dmet_cas) < threshold] = 0
        print(ncas, n_elec, spin, ecore, gh1e_dmet_cas.shape,gg2e_dmet_cas.shape)
        mpo = driver.get_qc_mpo(gh1e_dmet_cas, gg2e_dmet_cas, ecore=ecore)
        bond_dim_init,n_roots_for_soc=settings["bond_dim_init"],settings["n_roots_for_soc"]
        ket = driver.get_random_mps(tag="KET", bond_dim=bond_dim_init,nroots=n_roots_for_soc)
        bond_dims_schedule,noise_schedule,thrd_schedule=settings["bond_dims_schedule"],settings["noise_schedule"],settings["thrd_schedule"]
        time_start=time.time()
        energies = driver.dmrg(mpo, ket, n_sweeps=settings["n_sweeps"], bond_dims=bond_dims_schedule, noises=noise_schedule,
        thrds=thrd_schedule, iprint=1, dav_max_iter=settings["dav_max_iter"], cutoff=1E-24)
        time_end=time.time()
        au2cm = 219474.63111558527
        au2ev = 27.21139
        e0 = energies[0]
        ener_cm = []
        for ix, ex in enumerate(energies):
            ener_cm.append((ex - e0) * au2cm)
            print("%5d %20.10f Ha %15.6f eV %10.4f cm-1" % (ix, ex, (ex - e0) * au2ev, (ex - e0) * au2cm))
        print("time_used_for_soc=")
        print(time_end-time_start)
        return energies,time_end-time_start
