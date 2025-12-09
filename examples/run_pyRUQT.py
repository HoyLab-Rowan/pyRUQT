import pyruqt
import ruqt
from pyscf import gto

#These are the calculation settings you should need for any calculation
calc_type="ase"
method="mcpdft"
xyz_dir="./"
xyzfile="h6.xyz"
dft_functional="pbe"
basis_set='sto-3g'
aux_basis='df_default'
ecp=None
spin=0
charge=0
outfile="h6_test"+"_"+calc_type

#sie_negf Specific Keywords
exmol_prog="pyscf"
num_elec_atoms=2
min_trans_energy=-2.0
max_trans_energy=2.0
delta_energy=0.01
min_bias=-1.0
max_bias=1.0
delta_bias=0.1
temp=1E-5
full_align=True
align_with_atom=0
dos_calc=True

#PySCF MCPDFT Settings (Only used if method="mcpdft")
mcscf_type="casscf"
scf_solver="rks"
active_space=[4,4]
active_orb=[]
auto_as=False
verbosity=5

#Commands to generate orbital visualization
display_orbitals=False

#PySCF SCF/MCSCF convergence settings
conv_tol=1E-7
max_iter=200
scf_algo="diis"
diis_start_cycle=1
damping=0
level_shift=0
scf_guess="minao"
frac_occ=False

#These are the calls to the pyscf calculation subroutine
if calc_type=="qc":
 es_cal=pyruqt.es_calc(output=outfile,es_dir=xyz_dir,es_geo=xyzfile,dft_functional=dft_functional,basis_set=basis_set,ecp=ecp,scf_algo=scf_algo,es_method=method,conv_tol=conv_tol,mcscf_type=mcscf_type,scf_solver=scf_solver,active_space=active_space,active_orb=active_orb,auto_as=auto_as,verbosity=verbosity,display_orbitals=display_orbitals,max_iter=max_iter,diis_start_cycle=diis_start_cycle,damping=damping,level_shift=level_shift,scf_guess=scf_guess,frac_occ=frac_occ,aux_basis=aux_basis,spin=spin,charge=charge)
 es_cal.single_point()
elif calc_type=="ase":
 es_cal=pyruqt.sie_negf(exmol_prog=exmol_prog,elec_prog="supercell",output=outfile,exmol_dir=xyz_dir,exmol_geo=xyzfile,dft_functional=dft_functional,basis_set=basis_set,ecp=ecp,scf_algo=scf_algo,num_elec_atoms=num_elec_atoms,es_method=method,conv_tol=conv_tol,mcscf_type=mcscf_type,scf_solver=scf_solver,active_space=active_space,active_orb=active_orb,auto_as=auto_as,verbosity=verbosity,display_orbitals=display_orbitals,max_iter=max_iter,diis_start_cycle=diis_start_cycle,damping=damping,level_shift=level_shift,scf_guess=scf_guess,frac_occ=frac_occ,min_trans_energy=min_trans_energy,max_trans_energy=max_trans_energy,delta_energy=delta_energy,min_bias=min_bias,max_bias=max_bias,delta_bias=delta_bias,temp=temp,full_align=full_align,dos_calc=dos_calc,align_elec=align_with_atom,aux_basis=aux_basis,spin=spin,charge=charge)
 es_cal.current()
elif calc_type=="wbl":
 es_cal_wbl=pyruqt.wbl_negf(exmol_prog=exmol_prog,output=outfile,exmol_dir=xyz_dir,exmol_geo=xyzfile,dft_functional=dft_functional,basis_set=basis_set,ecp=ecp,scf_algo=scf_algo,num_elec_atoms=num_elec_atoms,es_method=method,conv_tol=conv_tol,mcscf_type=mcscf_type,scf_solver=scf_solver,active_space=active_space,active_orb=active_orb,auto_as=auto_as,verbosity=verbosity,display_orbitals=display_orbitals,max_iter=max_iter,diis_start_cycle=diis_start_cycle,damping=damping,level_shift=level_shift,scf_guess=scf_guess,frac_occ=frac_occ,min_trans_energy=min_trans_energy,max_trans_energy=max_trans_energy,delta_energy=delta_energy,min_bias=min_bias,max_bias=max_bias,delta_bias=delta_bias,temp=temp,aux_basis=aux_basis,spin=spin,charge=charge)
 es_cal_wbl.current()
