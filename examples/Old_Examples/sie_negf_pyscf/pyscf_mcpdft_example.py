import ruqt
import pyruqt

trans_calc=pyruqt.sie_negf(output="sie_mcpdft_lichain_casci",exmol_dir="./",exmol_geo="li_wire.xyz",exmol_prog="pyscf",elec_prog="supercell",basis_set="sto-3g",ecp=None,pyscf_settings=["mcpdft","casci",[4,4],"rks","pbe",5],elec_size=[10,10],conv_tol=1E-7,min_trans_energy=-5,max_trans_energy=5)
trans_calc.current()

#To run this file put the following into the command line replacing "script_name" with the name of this file: nohup python script_name.py &> script_name.log"
#Pyscf output will be written to the .log file. Pyruqt output will be writen to a .out file
#If the .err file is empty, your run likely worked. Check the .out/.log files to confirm
#To run MC-PDFT in pyscf, use the regular calculation variables shown above plus the new "pyscf_settings" variable. This is a list of variables that do the following:

#List spot (1): Select "dft" or "mcpdft" to change calculation type,

#List spot (2): If (1) is "mcpdft" select "casscf" or "casci", does nothing if (1) is "dft"

#List spot (3): CAS active space in brackets, [number of orbitals,number of electrons], does nothing if (1) is "dft"

#List spot (4): Type of SCF solver to use, choose rhf for Restricted Hartree-Fock and rks for Restricted DFT

#List Post (5): DFT functional and translated funcational to use, replaces separate "dft_functional" variable

#List Post (6): Amount of Pyscf output printed to log file, Default is 5, Higher=More output, Generally Leave Alone
