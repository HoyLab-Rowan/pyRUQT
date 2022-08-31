# pyRUQT
Modular Python-based Code for Multiconfigurational Non-Equilibrium Green's Function Methodologies

***Note: This is a placeholder for the release version that currently in progress. Full release of code coming later this month (August 2022)***

This is the Python-based successor to the Rowan University Transport (RUQT) code. It is designed to provide a modular framework for calculating charge 
transport using non-equilibrium Green's functions built from multiconfigurational electronic structure methods.It can use both an optmized version of the 
orginial RUQT code (RUQT-Fortran) or the Atomic Simulation Engine (ASE) for transport calculations and is currently capable of performing NEGF-MCPDFT, NEGF-
DFT (PySCF), and mixed method NEGF calculations (ex. MC-PDFT for extended molecule region and DFT for electrodes). Support for NEGF-RDM to come in furture 
(NEGF-RDM will require Maple with QC-Toolbox).

Each currently supported engine type (ASE and RUQT-Fortran) offer a different approach to treating electrode-extended molecule interactions and coupling:

1. ASE Transport Engine (default): Semi-infinite leads determined with an efficient decimation technique to determine Fermi level, device/electrode 
interactions, and coupling (see Paper #2). Separate Hamiltonian and Overlap matrices for the extended molecule and repeating electrode blocks are used to 
construct the Green's functions.

2. RUQT-Fortran Transport Engine: Metal wide band limit approximation with user provided Fermi level and coupling constants (Papers 1 & 3). Only 1 
Hamiltonian and Overlap matrix is used to contruct the Green's Functions which are divided by program into the electrode and extended molecule regions 
based on number of electrode atoms specificed by user.

This software runs the standard Landuaer current, conductance, and zero-bias transmission calculations found in RUQT-Fortran/ASE and adds additional 
calculation types and features not found in either:

New Calculation Types:
  1. Differential Conductance
    
New features:
  1.  Automatically run simple Molcas MC-PDFT calculations from pyRUQT
  2.  Full alignment of diagonal elements of electrode/extended molecule Hamiltonians for ASE calculations
  3.  Options to include additional electrode-molecule coupling for ASE NEGF caculations
  4.  Automatic plotting of transport results in PNG format

Required:

Python3, Numpy, Scipy, and Matplotlib (ASE only)

MKL (RUQT-Fortran)


NEGF Transport Calculator Options. Only 1 of the following are required but both recommended.

    (Default): Atomic Simulation Environment from https://wiki.fysik.dtu.dk/ase/

    (Optional): Compiled RUQT executable. Compile the RUQT.x executable in the RUQT subdirectory.

Electronic Structure Calculator Options. Only 1 of the following are required but both recommended:

    OpenMolcas(sandx_fock branch) installation (can be run separately, required for NEGF-MCPDFT) from https://gitlab.com/Molcas/OpenMolcas/-/tree/sandx_fock

    PySCF (enables non-Molcas NEGF-DFT calculations and mixed DFT/PDFT transport calculations by pyRUQT) from https://pyscf.org/

Quick Installation

    Put the pyruqt.py and ruqt.py files in your python module folder.

    Install ASE, OpenMolcas(sandx_fock branch), and optionally PySCF for all users.

    Use the pyRUQT_example.py script to get started running calculations.

If you use this code in your research please cite:

1. Andrew M. Sand, Justin T. Malme, and Erik P. Hoy, “A multiconfigurational pair-density functional theory approach to molecular junctions”, J. Chem. Phys., 155(11), 114115 (2021). https://doi.org/10.1063/5.0063293 

If you use the ASE transport engine also cite:

2. Ask Hjorth Larsen et al. J. Phys.: Condens. Matter 29, 273002 (2017). https://dog.org/10.1088/1361-648X/aa680e

If you use the RUQT-Fortran transport engine also cite:

3. Erik P. Hoy, David A. Mazziotti, and Tamar Seideman, “Development and application of a 2-electron reduced density matrix approach to electron transport 
via molecular junctions”, J. Chem. Phys. 147, 184110 (2017). https://doi.org/10.1063/1.4986804
