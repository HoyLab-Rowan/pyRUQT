##Parameters for semi-infinite electrode calculation using ASE##
Format= Keyword : {Default Value,Type} : Explaination of Keyword : Options/Context

***Keywords That Generaly Need Be Defined Each Calculation***
output     : {pyruqt_results,str} : Sets the filename to be used for all calculation outputfiles                                      : Define uniquely for each new calculation to avoid overwriting data
exmol_dir : {None,str}           : Defines location of extended molecular region geometry xyz file for pySCF or MolEl.dat for MOLCAS : Always needed
elec_dir  : {None,str}           : Defines location of the left electrode xyz file or MolEl.dat                                      : Not needed if elec_prog type is "supercell", Replaced by "elec_size"
elec2_dir : {None,str}           : Defines location of right electrode xyz/MolEl.dat if not same as left electrode                   : Can be omitted if electrodes are superimposable mirror images of each other

***Calculation Dependent or Optional Keywords***

**Keywords that define the electronic structure programs to be used**
  exmol_prog : {"molcas",str} : Defines what electronic structure program is used for the extended molecular region : Options are currently "molcas" or "pyscf"
  elec_prog  : {"molcas",str} : Same as exmol_prog but for the left and right electrodes                            : Additional option of "supercell" which activates the supercell method (no separate electrode calc)

**pyRUQT transport keywords**
 *Transport Keywords
  min_trans_energy : {-2.0,float}    : Lowest transmission energy to be calculated  in eV                        :
  max_trans_energy : {2.0,float}     : Highest transmission energy to be calculated in eV                        :
  delta_energy     : {0.01,float}    : Energy difference between transmission points to be calculated in eV      :
  min_bias         : {-2.0,float}    : Lowest voltage bias to be calculated in Volts                             : Not needed for transmission calculations
  max_bias         : {2.0,float}     : Highest bias voltage to be calculated in Volts                            : Not needed for transmission calculations
  delta_bias       : {0.1,float}     : Difference between bias voltage points in Volts                           : Not needed for transmission calculations
  spin_pol         : {False,logical} : Sets the current to 2*value when using data from 1-electron spin orbitals : Use if input data is from single electron orbitals vs double electron (standard for MOLCAS)
  dos_calc         : {False,bool}    : Generates the density of states using ASE                                 : Currently non-functional         
  fd_change        : {0.001,float}   : Finite difference to be used when calculating the difference conductance  :
  ase_current      : {False,bool}    : Use ASE's inbuilt current calculator instead of pyRUQT                    : Recommended to keep False as pyRUQT routine uses an improved numerical integration technique over ASE

 *Alignment/Electrode Keywords
  full_align    : {True,bool}      : Uses improved wieghed average alignment routine in pyRUQT over ASE's default alignment (align_elec)                      : Recommended to always use unless using ASE's align_elec instead
  align_elec    : {0,int}          : Default ASE alignment routine to align electrode/extended molecular regions based a selected element                     : Not recommended, replaced by "full_align"
  coupling_calc : {None,str}       : Improve coupling interactions defining coupling elements in off-diagonal paritioned Green's functions set to zero in ASE : Use "Fock_EX" value to activate
  coupled       : {"molecule",str} : Defines how much of the extended molecular region should receive the additional Fock_EX coupling with electrode          : Options are "molecule" or "extended_molecule"
  n_elec_units  : {2,int}          : Number of repeating electrode units in the electrode regions                                                             : Only default of 2 currently supported

 *Supercell Calculations
  elec_size : {None,[int,int]} : Basis functions to be placed in the electrode regions for a supercell calculation in list format, [left,right] : Required if elec_prog="supercell"


**Molcas specific keywords,Use if either exmol_prog or elec_prog equal "molcas"**
  state_num   : {1,int} : Select CI state to be used for transport calculations         : Values other than 1 need a corresponding Effective Hamiltonian matrix to be in MolEl.dat
  state_num_e : {1,int} : Select excited CI state to be used for transport calculations : Unused for now


**PySCF specific keywords, Use if either exmol_prog or elec_prog equal "pyscf"**
  exmol_geo      : {None,str}      : Define the full xyz file name for extended molecular region                                               : Be sure to include any extensions in name (ex. exmol_geo="geo_file.xyz")
  elec_geo       : {None,str}      : Define the full xyz file name for left electrode                                                          : Can be omitted if elec_prog="supercell" 
  elec2_geo      : {None,str}      : Define the full xyz file name for right electrode (if not same as left electrode)                         : Can be omitted if electrodes are superimposable mirror images of each other
  dft_functional : {"pbe",str}     : Defines the DFT functional to be used by PySCF                                                            : See pySCF manual for a full list of options
  basis_set      : {None,str}      : Defines basis set to be used by PySCF                                                                     : See PySCF manual for full list of keywords
  ecp            : {None,str}      : If your basis set has an ECP (or auxbasis for PBC PySCF) it must be defined separately using this keyword : Can be left empty for non-ECP Gaussian basis like 6-31G
  pyscf_pbc      : {False,logical} : Uses periodic basis set for pySCF calculations instead of Gaussian orbitals                               : Requires "lattice_v" to be defined with a list of the 3 lattice vectors, [a1,b2,c3]
  conv_tol       : {1E-7,float}    : Defines the SCF convergence of PySCF in Hartees                                                           :
  max_iter       : {100,int}       : Max number of PySCF iterations allowed before SCF termination                                             :

 *PySCF Keywords when using Periodic Boundary Conditions (Along X-Axis)
  pyscf_pbc     : {False,logical} : Use periodic basis set for pySCF calculations instead of Gaussian orbitals : Requires "lattice_v" to be defined with a list of the 3 lattice vectors: [a1,a2,a3]
  lattice_v     : {None,float}    : List of the 3 lattice vectors in 1-D list format of [a1,a2,a3]             : Required to be Defined for PBC pySCF
  meshum        : {10,float}      : # of mesh points used by pySCF per lattice vector                          : Highly recommended to adjust from default
  verbosity     : {4,int}         : Degree of pySCF output                                                     : Recommended to use 2 for no output and 4 for useable output (must pipe script output to file via command line)
