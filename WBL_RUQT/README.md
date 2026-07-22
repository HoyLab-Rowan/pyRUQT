# RUQT

***This version of RUQT builds the the WBL-NEGF libraries for the pip distribution of pyRUQT. It is no longer a standalone code, is maintained seperately from the original code, and likely will not work with old RUQT inputs or outside of pyRUQT without modificaiton. You can use mesonpy to rebuild these libraries on your own computer if needed after pip installation.

***Rowan University Quantum Transport (RUQT) NEGF Libraries*** 

RUQT is a wide band limit non-equilibrium Green's function (NEGF) based software package focused on studying electron correlation effects in charge transport problems. It is focused on integrating N- and 2-electron methods into the NEGF transport formalism and is the home of the NEGF-RDM and NEGF-PDFT transport methods. As of July 2021, it is capable of performing Landauer NEGF calculations using electronic structure data acquired from separate Hartree-Fock, Density Functional Theory, Multiconfiguration Pair Density Functional Theory, and Parametric 2-Electron Reduced Density Matrix Theory calculations. The code itself is written in Fortran90+ and has been converted into a Fortran library for pyRUQT's pip version. For older standalone versions of the code, see the separate repository on Github.

RUQT currently can read data from the following electronic structure software packages (more methods/package support to come in future):

HF and DFT: Q-Chem, PYSCF, GAMESS, Maple Quantum Chemistry Toolbox, and Molcas 
p2-RDM: Maple Quantum Chemistry and GAMESS (Not publically available)

The RUQT NEGF libraries are capable of performing non-self-consistent current and transmission calculations with metal wide band limit electrodes (fixed Fermi level). 

***Installation***

Required Libraries: Intel MKL (free or paid version)

Installed as part of pyRUQT through pip

***How to Use***

To run calculations, use pyRUQT with pySCF or Molcas for HF, DFT, or MC-PDFT calculations NEGF-RDM is not supported by pyRUQT right now. Use the standalone RUQT and corresponding Maple scripts instead.

NEGF-MCPDFT(Molcas)

In order to run NEGF-MCPDFT calculations using Molcas specifically, you will need to install the sandx_fock OpenMolcas branch available here (https://gitlab.com/Molcas/OpenMolcas/-/tree/sandx_fock) which generates the MolEl.dat files used by RUQT. These contain the MC-PDFT effective Hamilionian and overlap matrices from an OpenMolcas MC-PDFT calculation (see example directory for necessary inputs). These matrices are only available from the sandx_fock branch of OpenMolcas which can be installed according to the regular Openmolcas installation instructions. 

***If you use this code, cite:***

For NEGF-RDM:

Erik P. Hoy, David A. Mazziotti, and Tamar Seideman, “Development and application of a 2-electron reduced density matrix approach to electron transport via molecular 
junctions” J. Chem. Phys. 147, 184110 (2017).

For NEGF-MCPDFT:

Andrew M. Sand, Justin T. Malme, and Erik P. Hoy, “A multiconfigurational pair-density functional theory approach to molecular junctions”, J. Chem. Phys, 155(11), 114115 (2021). https://doi.org/10.1063/5.0063293 
