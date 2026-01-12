# pyRUQT
Modular Python-based Code for Multiconfigurational Non-Equilibrium Green's Function Methodologies

Most recent changes:
(12/23/2025): Improved PBC calculator for NEGF-DFT calculations. New run_pyRUQT.py example script which can be used to access all pyRUQT calculations. All other examples are outdated and moved to Old_Examples. We recommend using the run_pyRUQT.py for all pyRUQT calculations. Check out our YouTube channel for tutorials on how to use pyRUQT for charge transport calculations: https://www.youtube.com/@hoyresearchgroup

(12/4/2025): Added support for changing DF auxbasis, charge, and spin for pyscf calculations.

(6/29/2025): Added support for excited state calculations using MC-PDFT.

(5/21/2025): Updated PySCF calculator to allow for far more SCF/MC-PDFT calculation options to be set (particularly SCF and active space optimization options) and added support for restarts and orbital visualization (creates a molden file). Modified all supercell calculations to use the "num_elec_atoms" keyword to set the number of atoms in the electrode (the program now calculates the # of orbitals based on your chosen exmol_prog). Added new es_calc module for running separate electronic structure calculations with PySCF (MOLCAS not supported) which generates both chk files and MolEl.dat files for restarts and separate transport calculations respectively.

This is the Python-based successor to the Rowan University Transport (RUQT) code. It is designed to provide a modular framework for calculating charge 
transport using non-equilibrium Green's functions built from multiconfigurational electronic structure methods. It can use both an optmized version of the 
orginial RUQT code (RUQT-Fortran) or the Atomic Simulation Engine (ASE) for transport calculations and is currently capable of performing NEGF-MCPDFT, NEGF-
DFT (PySCF), and mixed method NEGF calculations (ex. MC-PDFT for extended molecule region and DFT for electrodes). Support for NEGF-RDM to come in future 
(NEGF-RDM will require the Maple Quantum Chemistry Toolbox).

Each currently supported NEGF engine types (ASE and RUQT-Fortran) offer a different approach to treating electrode-extended molecule interactions and coupling:

1. ASE Transport Engine (sie_negf class): Semi-infinite leads determined with an efficient decimation technique to determine Fermi level, device/electrode 
interactions, and coupling (see Paper #2). Separate Hamiltonian and Overlap matrices for the extended molecule and repeating electrode blocks are used to 
construct the Green's functions unless using the supercell option.

2. RUQT-Fortran Transport Engine (wbl_negf class): Metal wide band limit approximation with user provided Fermi level and coupling constants (Papers 1 & 3). Only 1 
Hamiltonian and Overlap matrix is used to contruct the Green's Functions which are divided by program into the electrode and extended molecule regions 
based on number of electrode atoms specified by user.

This software runs the standard Landuaer current, conductance, and zero-bias transmission calculations found in RUQT-Fortran/ASE and adds additional calculation types and features not found in either program:

New Calculation Types:
  1. Differential Conductance (using both RUQT-Fortran and ASE engines)
  2. Supercell calculations with ASE (no separate electrode required)
    
New features:
  1.  Automatically run MC-PDFT (Pyscf and Molcas) and DFT(PySCF) calculations (with or without periodic boundary conditions) from pyRUQT for transport calculations
  2.  Full alignment of diagonal elements of electrode/extended molecule Hamiltonians for ASE calculations
  3.  Options to include additional electrode-molecule coupling for ASE NEGF caculations
  4.  Automatic plotting of transport results in PNG format
  5.  Non-transport calculations using PySCF through es_calc class. Useful for optimizing the junction's electronic structure before running transport calculations (and you can use the chkfile/mo files from es_calc as guess files for   sie_negf/wbl_negf calcualtions)

Required:

Python3 (pref 3.9+), Numpy, Scipy, and Matplotlib

MKL (RUQT-Fortran)

NEGF and ES Calculator Options and Required Supporting Libraries. Only 1 of the following is required but all three are recommended.

    For sie_negf class: Atomic Simulation Environment from https://wiki.fysik.dtu.dk/ase/

    For wbl_negf class: Compiled RUQT executable. Compile the RUQT.x executable in the RUQT subdirectory.

    For es_calc class: PySCF from https://pyscf.org/
       
Electronic Structure Calculator Options. Only 1 of the following are required but both recommended:

    OpenMolcas(sandx_fock branch) installation (best run as separate calculation but can be run by pyRUQT) from https://gitlab.com/Molcas/OpenMolcas/-/tree/sandx_fock

    PySCF (enables non-Molcas NEGF-DFT/MC-PDFT transport calculations by pyRUQT) from https://pyscf.org/
       
Quick Installation (for now, Python package install coming in future)

    Put the pyruqt.py and ruqt.py files in your python module folder.

    Install ASE, OpenMolcas(sandx_fock branch), and PySCF for all users.

    Use the run_pyRUQT.py script in the examples to get started running calculations. Check out https://www.youtube.com/@hoyresearchgroup for tutorial videos

If you use this code in your research please cite:

1. Andrew M. Sand, Justin T. Malme, and Erik P. Hoy, “A multiconfigurational pair-density functional theory approach to molecular junctions”, J. Chem. Phys., 155(11), 114115 (2021). https://doi.org/10.1063/5.0063293 

If you use the ASE transport engine also cite:

2. Ask Hjorth Larsen et al. J. Phys.: Condens. Matter 29, 273002 (2017). https://doi.org/10.1088/1361-648X/aa680e

If you use the RUQT-Fortran transport engine also cite:

3. Erik P. Hoy, David A. Mazziotti, and Tamar Seideman, “Development and application of a 2-electron reduced density matrix approach to electron transport 
via molecular junctions”, J. Chem. Phys. 147, 184110 (2017). https://doi.org/10.1063/1.4986804
