List of all updates to pyRUQT)
(4/7/2025): Added ability to run MC-PDFT calculations using PySCF. Created new pyscf_settings keyword (see pyscf_mcpdft_example.py script in examples folder for details on how to use). Currently requires installing pyscf-forge in addition to PySCF: https://github.com/pyscf/pyscf-forge

(7/1/2024): Added ability to run PySCF calculations with periodic boundary conditions through pyRUQT and added the ability to use PySCF (without PBC) as a calculator for the wbl_negf module, a new KEYWORD file with detailed explanations of all sie_negf keywords, this changelog file, and minor bugfixes/updates to wbl_negf.

(4/30/2024): Added tools folder with additional calculators for pyRUQT

(8/25/2023): Support added for supercell calculations which removes the need for a separate electrode calculation when using sie_negf class. Activate using elec_prog="supercell" and include the number of orbitals in each electrode in the supercell via the elec_size=[X1,X2] keyword where X1 and X2 is the number of orbitals in left and right electrodes respectively.

(5/8/2023): General bug fixs/improvements. New Current Calculator and support for non-identical electrodes

