# pyRUQT
Modular Python-based Code for Multiconfigurational Non-Equilibrium Green's Function Methodologies


***Update (8/31/2022): First public version of pyRUQT (v0.1) is now available. More updates and improvements to follow later this year.***

This is the Python-based successor to the Rowan University Transport (RUQT) code. It is designed to provide a modular framework for calculating charge 
transport using non-equilibrium Green's functions built from multiconfigurational electronic structure methods. It can use both an optmized version of the 
orginial RUQT code (RUQT-Fortran) or the Atomic Simulation Engine (ASE) for transport calculations and is currently capable of performing NEGF-MCPDFT, NEGF-
DFT (PySCF), and mixed method NEGF calculations (ex. MC-PDFT for extended molecule region and DFT for electrodes). Support for NEGF-RDM to come in future 
(NEGF-RDM will require the Maple Quantum Chemistry Toolbox).

Each currently supported NEGF engine types (ASE and RUQT-Fortran) offer a different approach to treating electrode-extended molecule interactions and coupling:

1. ASE Transport Engine (sie_negf class): Semi-infinite leads determined with an efficient decimation technique to determine Fermi level, device/electrode 
interactions, and coupling (see Paper #2). Separate Hamiltonian and Overlap matrices for the extended molecule and repeating electrode blocks are used to 
construct the Green's functions.

2. RUQT-Fortran Transport Engine (wbl_negf class): Metal wide band limit approximation with user provided Fermi level and coupling constants (Papers 1 & 3). Only 1 
Hamiltonian and Overlap matrix is used to contruct the Green's Functions which are divided by program into the electrode and extended molecule regions 
based on number of electrode atoms specified by user.

This software runs the standard Landuaer current, conductance, and zero-bias transmission calculations found in RUQT-Fortran/ASE and adds additional calculation types and features not found in either program:

New Calculation Types:
  1. Differential Conductance (using both RUQT-Fortran and ASE engines)
    
New features:
  1.  Automatically run simple Molcas MC-PDFT and pySCF DFT calculations from pyRUQT for transport calculations
  2.  Full alignment of diagonal elements of electrode/extended molecule Hamiltonians for ASE calculations
  3.  Options to include additional electrode-molecule coupling for ASE NEGF caculations
  4.  Automatic plotting of transport results in PNG format

Required:

Python3, Numpy, Scipy, and Matplotlib

MKL (RUQT-Fortran)


NEGF Transport Calculator Options. Only 1 of the following are required but both are recommended to enable both NEGF calculators.

    For sie_negf class: Atomic Simulation Environment from https://wiki.fysik.dtu.dk/ase/

    For wbl_negf class: Compiled RUQT executable. Compile the RUQT.x executable in the RUQT subdirectory.

Electronic Structure Calculator Options. Only 1 of the following are required but both recommended:

    OpenMolcas(sandx_fock branch) installation (best run as separate calculations but can be run by pyRUQT) from https://gitlab.com/Molcas/OpenMolcas/-/tree/sandx_fock

    PySCF (enables non-Molcas NEGF-DFT calculations and mixed DFT/PDFT transport calculations by pyRUQT) from https://pyscf.org/

Quick Installation (for now, Python package install coming in future)

    Put the pyruqt.py and ruqt.py files in your python module folder.

    Install ASE, OpenMolcas(sandx_fock branch), and optionally PySCF for all users.

    Check the examples folder for scripts to get started running calculations.

If you use this code in your research please cite:

1. Andrew M. Sand, Justin T. Malme, and Erik P. Hoy, “A multiconfigurational pair-density functional theory approach to molecular junctions”, J. Chem. Phys., 155(11), 114115 (2021). https://doi.org/10.1063/5.0063293 

If you use the ASE transport engine also cite:

2. Ask Hjorth Larsen et al. J. Phys.: Condens. Matter 29, 273002 (2017). https://doi.org/10.1088/1361-648X/aa680e

If you use the RUQT-Fortran transport engine also cite:

3. Erik P. Hoy, David A. Mazziotti, and Tamar Seideman, “Development and application of a 2-electron reduced density matrix approach to electron transport 
via molecular junctions”, J. Chem. Phys. 147, 184110 (2017). https://doi.org/10.1063/1.4986804
