"""
dmrg-casscf and nevpt2 interface 
may be buggy for new functions 
copied and adapted from liblan interface for dmrg
"""
from pyscf import scf, mcscf, gto, dmrgscf, lib,fci,mrpt
from pyscf.mcscf import avas
from pyscf.lib import logger
import numpy as np
import os
import h5py 
default_settings={"quicktest":False,"ptmpsdim":500,"threads":8,"memory":4,"maxM":1500,"tol":1E-10,"scratch":"/tmp","extrakeyword":["real_density_matrix","davidson_soft_max_iter 1600", "noreorder", "cutoff 1E-24"],"maxiter":None,"two_dot_to_one_dot":None,"scheduleSweeps":None,"scheduleMaxMs":None,"scheduleTols":None,"scheduleNoises":None}
dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = ''
lib.param.TMPDIR = os.path.abspath(lib.param.TMPDIR)
def dmrgscf_mixer(mf_without_dmet,mf, ncas, nelec, statelis=None, weights = None, fix_spin_shift=0.5,settings=default_settings):
    solver = mcscf.CASSCF(mf,ncas,nelec)
    """
    solver.canonicalization = True
    solver.natorb=True
    """
    if statelis is None:
        logger.info(solver,'statelis is None')
        from embed_sim_dmrg import spin_utils
        statelis = spin_utils.gen_statelis(ncas, nelec)
        logger.info(solver,'generate statelis %s', statelis)
    solvers = []
    logger.info(solver,'Attempting SA-CASSCF with')
    for i in range(len(statelis)):
        if statelis[i]:
            newsolvers=dmrgscf.DMRGCI(mf_without_dmet.mol,maxM=settings["maxM"],tol=settings["tol"])
            newsolvers.spin=i
            newsolvers.threads=settings["threads"]
            newsolvers.memory=settings["memory"]
            newsolvers.nroots=statelis[i]
            newsolvers.runtimeDir = settings["scratch"] + "/%d" % i
            newsolvers.scratchDirectory = settings["scratch"] + "/%d" % i
            if settings["extrakeyword"] is not None:
                newsolvers.block_extra_keyword =settings["extrakeyword"]
            if settings["maxiter"] is not None:
                newsolvers.maxiter=settings["maxiter"]
            if settings["two_dot_to_one_dot"] is not None:
                newsolvers["two_dot_to_one_dot"]=settings["two_dot_to_one_dot"]
            if settings["scheduleSweeps"] is not None:
                newsolvers.maxiter=settings["scheduleSweeps"]
            if settings["scheduleMaxMs"] is not None:
                newsolvers.maxiter=settings["scheduleMaxMs"]
            if settings["scheduleTols"] is not None:
                newsolvers.maxiter=settings["scheduleTols"]
            if settings["scheduleNoises"] is not None:
                newsolvers.maxiter=settings["scheduleNoises"]
            solvers.append(newsolvers)
    statetot = np.sum(statelis)
    if weights is None:
        weights = np.ones(statetot)/statetot
    mcscf.state_average_mix_(solver, solvers, weights)
    if settings["quicktest"]:
        solver.max_cycle=1
    """
    if settings["reload"]:
        solver=sacasscf_load_chk(solver,"dmrgscf.h5")
    else:
        solver.kernel(mo)
    if settings["saving"]:
        sacasscf_dump_chk(solver,"dmrgscf.h5")
    """
    return solver    
    
"""
def sacasscf_dump_chk(solver,sav):
    with h5py.File(sav, 'w') as fh5:
        fh5['mo_coeff'] = solver.mo_coeff
def sacasscf_load_chk(solver,sav):
    with h5py.File(sav, 'r') as fh5:
        solver.mo_coeff=fh5['mo_coeff'][:]
        solver.kernel()
    return solver
"""

def read_statelis(mc):
    spins = []
    nroots = []
    for solver in mc.fcisolver.fcisolvers:
        spins.append(solver.spin)
        nroots.append(solver.nroots)
    max_spin = np.max(np.array(spins))
    statelis = np.zeros(max_spin + 1, dtype=int)
    statelis[spins] = nroots
    return statelis
    
def dmrgscf_nevpt2(mf,mc,settings=default_settings):
    return dmrgscf_nevpt2_casci_ver(mf,mc,settings)

from pyscf.fci.addons import _unpack_nelec

def dmrgscf_nevpt2_undo_ver(mc,settings=default_settings):
    from pyscf.mcscf.addons import StateAverageFCISolver
    from pyscf.mcscf.df import _DFCAS
    if isinstance(mc.fcisolver, dmrgscf.DMRGCI):
        spins = []
        nroots = []
        for solver in mc.fcisolver.fcisolvers:
            spins.append(solver.spin)
            nroots.append(solver.nroots)
        e_corrs = []
        print('undo state_average')
        sa_fcisolver = mc.fcisolver
        mc.fcisolver = mc.fcisolver.undo_state_average()
        for i, spin in enumerate(spins):
            mc.nelecas = _unpack_nelec(mc.nelecas, spin)
            mc.fcisolver.spin = spin
            nroot = nroots[i]
            for iroot in range(0, nroot):
                if isinstance(mc, _DFCAS):
                    from embed_sim.df import DFNEVPT
                    if settings["ptmpsdim"] is not None:
                        nevpt2 = DFNEVPT(mc, root=iroot+np.sum(nroots[:i],dtype=int), spin=spin).compress_approx(maxM= settings["ptmpsdim"])
                    else:
                        nevpt2 = DFNEVPT(mc, root=iroot+np.sum(nroots[:i],dtype=int), spin=spin)
                else:
                    print('spin', spin, 'iroot', iroot)
                    if settings["ptmpsdim"] is not None:
                        nevpt2 = mrpt.NEVPT(mc, root=iroot+np.sum(nroots[:i],dtype=int)).compress_approx(maxM= settings["ptmpsdim"]).set(canonicalized=True)
                    else:
                        nevpt2 = mrpt.NEVPT(mc, root=iroot+np.sum(nroots[:i],dtype=int)).set(canonicalized=True)
                nevpt2.verbose = logger.INFO-1 # when verbose=logger.INFO, meta-lowdin localization is called and cause error in DMET-NEVPT2
                e_corr = nevpt2.kernel()
                e_corrs.append(e_corr)
        print('redo state_average')
        mc.fcisolver = sa_fcisolver
    else:
        raise TypeError(mc.fcisolver, 'Not DMRG Solver')
    return np.array(e_corrs)
    
def dmrgscf_nevpt2_casci_ver(mf_without_dmet,mc,settings=default_settings):
    print('sacasscf_nevpt2_casci_ver')
    from pyscf.mcscf.addons import StateAverageFCISolver
    from pyscf.mcscf.df import _DFCAS
    if isinstance(mc.fcisolver, dmrgscf.DMRGCI):
        spins = []
        nroots = []
        for solver in mc.fcisolver.fcisolvers:
            spins.append(solver.spin)
            nroots.append(solver.nroots)
        e_corrs = []
        for i, spin in enumerate(spins):
            print('CASCI')
            mc_ci = mcscf.CASCI(mc._scf, mc.ncas, mc.nelecas)
            mc_ci.nelecas = _unpack_nelec(mc.nelecas, spin)
            mc_ci.fcisolver=dmrgscf.DMRGCI(mf_without_dmet.mol,maxM=settings["maxM"],tol=settings["tol"])
            mc_ci.fcisolver.spin = spin
            mc_ci.fcisolver.nroots = nroots[i] # this is important for convergence of CASCI
            mc_ci.fcisolver.threads=settings["threads"]
            mc_ci.fcisolver.memory=settings["memory"]
            mc_ci.fcisolver.runtimeDir = settings["scratch"] + "/%d" % spin
            mc_ci.fcisolver.scratchDirectory = settings["scratch"] + "/%d" % spin
            if settings["extrakeyword"] is not None:
                mc_ci.fcisolver.block_extra_keyword =settings["extrakeyword"]
            if settings["maxiter"] is not None:
                mc_ci.fcisolver.maxiter=settings["maxiter"]
            if settings["two_dot_to_one_dot"] is not None:
                mc_ci.fcisolver.two_dot_to_one_dot=settings["two_dot_to_one_dot"]
            if settings["scheduleSweeps"] is not None:
                mc_ci.fcisolver.scheduleSweeps=settings["scheduleSweeps"]
            if settings["scheduleMaxMs"] is not None:
                mc_ci.fcisolver.scheduleMaxMs=settings["scheduleMaxMs"]
            if settings["scheduleTols"] is not None:
                mc_ci.fcisolver.scheduleTols=settings["scheduleTols"]
            if settings["scheduleNoises"] is not None:
                mc_ci.fcisolver.fcisolver.scheduleNoises=settings["scheduleNoises"]
            mc_ci.fcisolver.canonicalization = False
            mc_ci.fcisolver.natorb = False
            mc_ci.kernel(mc.mo_coeff)
            nroot = nroots[i]
            for iroot in range(0, nroot):
                if isinstance(mc, _DFCAS):
                    from embed_sim.df import DFNEVPT
                    if settings["ptmpsdim"] is not None:
                        nevpt2 = DFNEVPT(mc_ci, root=iroot, spin=spin).compress_approx(maxM= settings["ptmpsdim"]).set(canonicalized=True)
                    else:
                        nevpt2 = DFNEVPT(mc_ci, root=iroot, spin=spin).set(canonicalized=True)
                else:
                    print('spin', spin, 'iroot', iroot)
                    if settings["ptmpsdim"] is not None:
                        nevpt2 = mrpt.NEVPT(mc_ci, root=iroot).compress_approx(maxM= settings["ptmpsdim"]).set(canonicalized=True)
                    else:
                        nevpt2 = mrpt.NEVPT(mc_ci, root=iroot).set(canonicalized=True)
                nevpt2.verbose = logger.INFO-1 # when verbose=logger.INFO, meta-lowdin localization is called and cause error in DMET-NEVPT2
                # nevpt2.verbose = 0 # when verbose=logger.INFO, meta-lowdin localization is 
                e_corr = nevpt2.kernel()
                e_corrs.append(e_corr)
    else:
        raise TypeError(mc.fcisolver, 'Not DMRG Solver')
    return np.array(e_corrs)


def analysis(mc):
    if isinstance(mc.fcisolver, dmrgscf.DMRGCI):
        spins = []
        nroots = []
        for solver in mc.fcisolver.fcisolvers:
            spins.append(solver.spin)
            nroots.append(solver.nroots)
        e_corrs = []
        for i, spin in enumerate(spins):
            mc_ci = mcscf.CASCI(mc._scf, mc.ncas, mc.nelecas)
            mc_ci.nelecas = _unpack_nelec(mc.nelecas, spin)
            mc_ci.fcisolver.spin = spin
            mc_ci.fix_spin_(shift=0.5, ss=(spin/2)*(spin/2+1))
            mc_ci.fcisolver.nroots = nroots[i] # this is important for convergence of CASCI
            mc_ci.kernel(mc.mo_coeff)
            nroot = nroots[i]
            for iroot in range(0, nroot):
                print('analyze spin', spin, 'iroot', iroot)
                #mc_ci.analyze(ci=mc_ci.ci[iroot])
                print("ci coeff is not avilable for DMRG wfn")
    else:
        raise TypeError(mc.fcisolver, 'Not DMRG Solver')
    return np.array(e_corrs)
