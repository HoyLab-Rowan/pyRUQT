from ase import transport,Atoms,units
from pyscf import gto
import matplotlib.pyplot as plt
import numpy as np
import ruqt
from numpy import linalg
import sys

class sie_negf:
 def __init__(self, **kwargs):

  self.input_parameters = {'output'    : "pyruqt_results",
                          'exmol_prog' : "pyscf",
                          'exmol_dir'  : "./",
                          'elec_prog'  : "supercell",
                          'elec_dir'   : None,
                          'elec2_dir'  : None,
                          'run_molcas'  : False,
                          'min_trans_energy' : -2.0,
                          'max_trans_energy' : 2.0,
                          'delta_energy' : 0.001,
                          'min_bias'     : -2,
                          'max_bias'     : 2,
                          'delta_bias'   : 0.1,
                          'temp'         : 1E-5,
                          'full_align'   : True,
                          'basis_set'  : None,
                          'ecp'        : None,
                          'n_elec_units'   : 2,
                          'exmol_geo'  : None,
                          'elec_geo'   : None,
                          'elec2_geo'  : None,
                          'state_num'  : 1, 
                          'trans_state' : 1,
                          'coupling_calc' : "none",
                          'coupled'       : "molecule",
                          'spin_pol'      : False,
                          'align_elec'    : 0,
                          'dos_calc'      : False,
                          'fd_change'     : 0.001,
                          'ase_current'   : False,
                          'num_elec_atoms': 0,
                          'conv_tol'      : 1E-7,
                          'max_iter'      : 100,
                          'pyscf_pbc'     : False,
                          'lattice_v'     : None,
                          'meshnum'       : None,
                          'verbosity'     : 5,
                          'cell_dim'      : 2,
                          'pbc_spin'      : None,
                          'aux_basis'     : "df_default",
                          'pdos_states'   : [],
                          'eigenchannels' : 0,
                          'es_method'  : "mcpdft",
                          'dft_functional' : "pbe",
                          'mcscf_type'    : "casci",
                          'scf_solver'    : "rks",
                          'active_space'  : [2,2],
                          'active_orb'    : [],
                          'auto_as'       : False,
                          'display_orbitals' : True,
                          'read_mc_mo'    : False,
                          'molel_read_dir' : "",
                          'diis_start_cycle' : 1,
                          'scf_algo'      : "cdiis",
                          'damping'       : 0,
                          'level_shift'   : 0,
                          'scf_guess'     : 'minao',
                          'frac_occ'      : 'false',
                          'molcas_supercell'      : False,
                          'charge'        : 0,
                          'spin'          : 0,
                          'smearing' : None,
                          'smearing_width' : 0.05,
                          'remove_linear_dep' : True}
  self.param_update(**kwargs)
  
 def param_update(self,**kwargs):
  inp=self.input_parameters
  for key in kwargs:
   if key in inp:
    inp[key] = kwargs[key]
   elif key not in inp:
    raise KeyError('%r not a vaild keyword. Please check your input parameters.' % key)

 def calc_setup(self):
  inp=self.input_parameters
  outputfile=open(inp['output']+".out",'w')
  sys.stderr=open(inp['output']+".err",'w')

  pyscf_settings=[inp['es_method'],inp['mcscf_type'],inp['active_space'],inp['scf_solver'],inp['dft_functional'],inp['verbosity'],inp['active_orb'],inp['auto_as'],inp['display_orbitals'],inp['output'],inp['state_num'],inp['trans_state'],inp['aux_basis']]
  pyscf_conv_settings=[inp['max_iter'],inp['conv_tol'],inp['diis_start_cycle'],inp['damping'],inp['level_shift'],inp['scf_algo'],inp['scf_guess'],inp['read_mc_mo'],inp['molel_read_dir'],inp['frac_occ'],inp['charge'],inp['spin'],inp['smearing'],inp['smearing_width'],inp['remove_linear_dep']]
 
  print("Performing non-self-consistent NEGF transport calculations using semi-infinite tight-binding electrodes",file=outputfile)
  print("Using Atomic Simulation Environment to calculate electrode interactions and transport",file=outputfile)
  print("Running Calculation using the following paramaters:",file=outputfile)
  print(self.input_parameters,file=outputfile)
  energies=np.arange(inp['min_trans_energy'],inp['max_trans_energy']+inp['delta_energy'],inp['delta_energy'])
  bias=np.arange(inp['min_bias'],inp['max_bias']+inp['delta_bias'],inp['delta_bias'])
  bias=np.where(abs(bias)<0.01, 0.0001, bias)

  if inp['exmol_prog']=="molcas":
   print("Using Molcas calculation at "+inp['exmol_dir']+" for extended molecular region",file=outputfile)
   print("Using the effective Hamiltonian for electronic state "+str(inp['state_num'])+" of extended mol. region",file=outputfile)
  elif inp['exmol_prog']=="pyscf":
   print("Calculating extended molecular region using Pyscf with "+pyscf_settings[4],file=outputfile)
   if pyscf_settings[0]=="mcpdft":
    print("Using Pyscf for an MC-PDFT calculation",file=outputfile)
  if inp['elec_prog']=="molcas":
   print("Using Molcas calculation at "+inp['elec_dir']+" for left electrode",file=outputfile)
   if inp['elec2_dir']!=None:
    print("Using non-identical electrodes. Right electrode geometry taken from: "+inp['elec2_dir'],file=outputfile)
   else:
    print("Assuming symmetric electrodes",file=outputfile)
   print("Using the effective Hamiltonian for electronic state "+str(inp['state_num'])+" of extended mol. region",file=outputfile)

  if inp['exmol_prog']=="molcas":
   h,s,norb,numelec,actorb,actelec,states=ruqt.esc_molcas2(inp['exmol_dir'],"MolEl.dat",inp['state_num'],outputfile)
   #h,s=ruqt.esc_molcas(exmol_file,exmol_dir,exmol_molcasd,state_num,outputfile)
  elif inp['exmol_prog']=="pyscf":
   if inp['pyscf_pbc']==True:
    h,s,norb,numelec=ruqt.esc_pyscf_pbc(inp['exmol_dir']+inp['exmol_geo'],pyscf_settings[4],inp['basis_set'],inp['ecp'],inp['lattice_v'],inp['meshnum'],inp['cell_dim'],pyscf_settings,pyscf_conv_settings)
   else:
    h,s,norb,numelec,elec_orb=ruqt.esc_pyscf2(inp['exmol_dir']+inp['exmol_geo'],pyscf_settings[4],inp['basis_set'],inp['ecp'],inp['num_elec_atoms'],pyscf_settings,pyscf_conv_settings)

  if inp['elec_prog']=="molcas":
   h1,s1,norb_le,numelec_le,actorb_le,actelec_le,states_le=ruqt.esc_molcas2(inp['elec_dir'],"MolEl.dat",inp['state_num'],outputfile)
   if inp['elec2_dir']!=None:
    h2,s2,norb_re,numelec_re,actorb_re,actelec_re,states_re=ruqt.esc_molcas2(inp['elec_dir'],"MolEl.dat",inp['state_num'],outputfile)
   else:
    h2=None
    s2=None

  elif inp['elec_prog']=="pyscf":
   if inp['pyscf_pbc']==True:
    h1,s1,norb_le,numelec_le=ruqt.esc_pyscf_pbc(inp['elec_dir']+inp['elec_geo'],pyscf_settings[4],inp['basis_set'],inp['ecp'],inp['lattice_v'],inp['meshnum'],inp['cell_dim'],pyscf_settings,pyscf_conv_settings)
   else:
    h1,s1,norb_le,numelec_le,elec_orb_le=ruqt.esc_pyscf2(inp['elec_dir']+inp['elec_geo'],pyscf_settings[4],inp['basis_set'],inp['ecp'],inp['num_elec_atoms'],pyscf_settings,pyscf_conv_settings)
   if inp['elec2_geo']!=None:
    if inp['pyscf_pbc']==True:
     h2,s2,norb_re,numelec_re=ruqt.esc_pyscf_pbc(inp['elec_dir']+inp['elec_geo'],pyscf_settings[4],inp['basis_set'],inp['ecp'],inp['lattice_v'],inp['meshnum'],inp['cell_dim'],pyscf_settings,pyscf_conv_settings)
    else:
     h2,s2,norb_re,numelec_re,elec_orb_re=ruqt.esc_pyscf2(inp['elec_dir']+inp['elec_geo'],pyscf_settings[4],inp['basis_set'],inp['ecp'],inp['num_elec_atoms'],pyscf_settings,pyscf_conv_settings)
   else:
    h2=None
    s2=None

  elif inp['elec_prog']=="supercell":
   l_elec=0
   r_elec=0
   if inp['exmol_prog']=="molcas":
    if inp['molcas_supercell']==True:
     size_ex,elec_orb=ruqt.read_syminfo(inp['exmol_dir'],0,inp['num_elec_atoms'],inp['output'])
    else:
     geo2=gto.M(atom=inp['exmol_dir']+inp['exmol_geo'],basis=inp['basis_set'],ecp=inp['ecp']) 
     ao_data=gto.mole.ao_labels(geo2,fmt=False)
     atom_num=0
     ao_index=0
     elec_orb=0
     ao_data_len=len(ao_data)
     while atom_num < inp['num_elec_atoms']:
      atom_num=ao_data[ao_index][0]
      ao_index+=1
      if ao_index == ao_data_len:
       print("The # of electrode atoms is incorrect")
       break
     elec_orb=int(ao_index)-1

    l_elec=elec_orb
    r_elec=elec_orb
   else:
    l_elec=elec_orb
    r_elec=elec_orb
   print("Orbitals in Left and Right Electrode: "+str(elec_orb),file=outputfile)
   h1=h[:l_elec,:l_elec]
   h2=h[-r_elec:,-r_elec:]
   s1=s[:l_elec,:l_elec]
   s2=s[-r_elec:,-r_elec:]

  if inp['coupling_calc']=="Fock_EX":
   hc1,sc1,hc2,sc2=ruqt.calc_coupling(h,s,h1,h2,s1,s2,inp['coupled'],inp['n_elec_units'])
  else:
   hc1=None
   sc1=None
   hc2=None
   sc2=None

  return(energies,bias,outputfile,h,h1,h2,s,s1,s2,hc1,sc1,hc2,sc2)

 def transmission(self):
  energies,bias,outputfile,h,h1,h2,s,s1,s2,hc1,sc1,hc2,sc2=sie_negf.calc_setup(self)
  inp=self.input_parameters
  print("Calculating "+str(len(energies))+" transmission energies from: "+str(max(energies))+" eV to "+str(min(energies))+" eV",file=outputfile)
  print("Final transmission values will be printed to "+inp['output']+".trans"+" in relative transmission vs eV",file=outputfile)

 
  print("Performing NEGF Transport Calculations using the Atomic Simulation Environment",file=outputfile)
  if inp['full_align']==True:
   h=ruqt.full_align(h,h1,s,inp['n_elec_units'])
  if inp['align_elec']>=1:
   inp['align_elec']-=1
   print("Aligning the "+str(inp['align_elec'])+" element of both electrode and extended molecule",file=outputfile)
   calc = transport.TransportCalculator(h=h, h1=h1,h2=h2, s=s, s1=s1,s2=s2,hc1=hc1,sc1=sc1,hc2=hc2,sc2=sc2, energies=energies,dos=inp['dos_calc'],logfile=inp['output']+".trans",align_bf=inp['align_elec'],pdos=inp['pdos_states'],eigenchannels=inp['eigenchannels'])
  elif inp['align_elec']<1:
   calc=transport.TransportCalculator(h=h, h1=h1,h2=h2, s=s, s1=s1,s2=s2,hc1=hc1,sc1=sc1,hc2=hc2,sc2=sc2, energies=energies,dos=inp['dos_calc'],logfile=inp['output']+".trans",pdos=inp['pdos_states'],eigenchannels=inp['eigenchannels'])
  T = calc.get_transmission()
  t_plot=plt.plot(energies, T)
  plt.xlabel('E-E(Fermi) (eV)')
  plt.ylabel('Transmission (rel)')
  plt.savefig(inp['output']+"_trans.png")
  plt.clf()

#Automatic printing and plotting routines for Total and Projected Density of States
  if inp['dos_calc']==True or inp['pdos_states']!=[]:
   dosfile=open(inp['output']+".dos",'w')
  if inp['dos_calc']==True:
   print("Calculating Total DOS",file=outputfile)
   d_plot=plt.plot(energies,calc.dos_e)
   plt.xlabel('E-E(Fermi) (eV)')
   plt.ylabel('Density of States')
   plt.savefig(inp['output']+"_dos.png")
   plt.clf()
   print("Total DOS",len(calc.dos_e),file=dosfile)
   tempv=0
   while tempv <= len(energies)-1:
    print(energies[tempv],calc.dos_e[tempv],file=dosfile)
    tempv+=1
    if tempv > len(energies):
     break

  if inp['pdos_states']!=[]:
   pdos_len=len(inp['pdos_states'])
   print("Plotting PDOS States:"+str(pdos_len),file=outputfile)
   vec=0
   pdos_ne_comb=np.zeros(len(energies))
   while vec <= pdos_len-1:
    #print(str(calc.pdos_ne.shape),file=outputfile)
    d_plot=plt.plot(energies,calc.pdos_ne[vec,:])
    plt.xlabel('E-E(Fermi) (eV)')
    plt.ylabel('Projected Density of States')
    plt.savefig(inp['output']+"_pdos"+str(vec+1)+".png")
    plt.clf()
    print("PDOS State",inp['pdos_states'][vec],len(energies),file=dosfile)
    tempv=0
    while tempv <= len(energies)-1:
     print(energies[tempv],calc.pdos_ne[vec,tempv],file=dosfile)
     pdos_ne_comb[tempv]+=calc.pdos_ne[vec,tempv]
     tempv+=1
     if tempv > len(energies):
      break
    vec+=1
    if pdos_len < 0 or vec > pdos_len:
     break
   print("PDOS Combined",file=dosfile)
   for i in range(0,len(energies)-1):
    print(energies[i],pdos_ne_comb[i],file=dosfile)
   d_plot=plt.plot(energies,pdos_ne_comb)
   plt.xlabel('E-E(Fermi) (eV)')
   plt.ylabel('Projected Density of States')
   plt.savefig(inp['output']+"_pdos_combined"+".png")
   plt.clf()

  #if inp['eigenchannels'] > 0:
  # d_plot=plt.plot(energies,calc.dos_e)
  # plt.xlabel('E-E(Fermi) (eV)')
  # plt.ylabel('Eigenchannels')
  # plt.savefig(inp['output']+"_eigen.png")
  # plt.clf()


  return T,calc,energies,bias,outputfile

 def current(self):
  #energies,bias,outputfile,h,h1,h2,s,s1,s2,hc1,sc1,hc2,sc2=sie_negf.calc_setup(self)
  inp=self.input_parameters

  T,calc,energies,bias,outputfile=sie_negf.transmission(self)
  print("Performing a Landauer current calculation for the following bias voltage range(V): "+str(bias),file=outputfile)
  print("Calculating "+str(len(energies))+" transmission energies from: "+str(max(energies))+" eV to "+str(min(energies))+" eV",file=outputfile)
  print("Final transmission values will be printed to "+inp['output']+".trans"+" in relative transmission vs eV",file=outputfile)
  print("Final current values will be printed to "+inp['output']+".iv"+" in volts vs ampheres",file=outputfile)
  print("Final conductance values will be printed to "+inp['output']+".con"+" in volts vs G_0",file=outputfile)

  if inp['ase_current']==True:
   I = calc.get_current(bias,T=inp['temp'],E=energies,T_e=T,spinpol=inp['spin_pol'])
  else:
   I = ruqt.get_current(bias,T=inp['temp'],E=energies,T_e=T,spinpol=inp['spin_pol'],delta_e=inp['delta_energy'])
  i_plot=plt.plot(bias,2.*units._e**2/units._hplanck*I)
  plt.xlabel('Voltage (V)')
  plt.ylabel('Current (A)')
  plt.savefig(inp['output']+"_current.png")

  b_range=len(bias)
  cond=np.zeros(b_range)
  for x in range(len(bias)):
   cond[x]=I[x]/bias[x]
  I=2.*units._e**2/units._hplanck*I
  np.savetxt(inp['output']+".iv",np.c_[bias,I],fmt="%s")

  plt.clf()
  c_plot=plt.plot(bias,cond)
  plt.xlabel('Voltage (V)')
  plt.ylabel('Conductance (G_0)')
  plt.savefig(inp['output']+"_conductance.png")
  np.savetxt(inp['output']+".con",np.c_[bias,cond],fmt="%s")

 def diff_conductance(self):
  #energies,bias,outputfile,h,h1,h2,s,s1,s2,hc1,sc1,hc2,sc2=sie_negf.calc_setup(self)
  inp=self.input_parameters
  T,calc,energies,bias,outputfile=sie_negf.transmission(self)
  print("Calculating differential conductance using numerical derivatives",file=outputfile)
  print("Calculting each value using the +/-"+str(inp['fd_change'])+" voltage points around it.",file=outputfile)
  print("Performing the diff. conductance calculation for the following bias voltage range(V): "+str(bias),file=outputfile)
  print("Calculating "+str(len(energies))+" transmission energies from: "+str(max(energies))+" eV to "+str(min(energies))+" eV around electrode Fermi level.",file=outputfile)
  print("Final transmission values will be printed to "+inp['output']+".trans"+" in relative transmission vs eV",file=outputfile)
  print("Final diff. conductance values will be printed to "+inp['output']+".dcon"+" in volts vs G_0",file=outputfile)
 
  #T,calc=sie_negf.transmission(self)

  DE=ruqt.get_diffcond(calc,bias,inp['temp'],energies,T,inp['fd_change'],inp['ase_current'],inp['delta_energy'])

  c_plot=plt.plot(bias,DE)
  plt.xlabel('Voltage (V)')
  plt.ylabel('Diff. Conductance (G_0)')
  plt.savefig(inp['output']+"_diffcon.png")
  np.savetxt(inp['output']+".dcon",np.c_[bias,DE],fmt="%s")


class wbl_negf:
 def __init__(self, **kwargs):
  
  self.input_parameters = {'output'     : "pyruqtresults",
                          'exmol_dir'  : "./",
                          'num_elec_atoms' : 0,
                          'exmol_prog' : "pyscf",
                          'run_molcas'  : False,
                          'min_trans_energy' : -2.0,
                          'max_trans_energy' : 2.0,
                          'delta_energy' : 0.01,  
                          'min_bias'     : -2,
                          'max_bias'     : 2,
                          'delta_bias'   : 0.1,
                          'temp'         : 1E-5,
                          'basis_set'  : None,
                          'ecp'        : None,
                          'exmol_geo'  : None,
                          'state_num'  : 1,
                          'trans_state' :1,
                          'FermiE'     :-5.30,
                          'FermiD'     : 0.07,
                          'qc_method'  : "dft",
                          'rdm_type'   : 1,
                          'fort_data'  : None,
                          'fort_trans' : False,
                          'fd_change'  : 0.001,
                          'ase_current' : False,
                          'conv_tol'      : 1E-7,
                          'max_iter'      : 200,
                          'pyscf_pbc'     : False,
                          'lattice_v'     : None,
                          'meshnum'       : None,
                          'cell_dim'      : 2,
                          'pbc_spin'      : None,
                          'aux_basis'     : "df_default",
                          'pdos_states'   : [],
                          'eigenchannels' : 0,
                          'verbosity'     : 5,
                          'es_method'  : "mcpdft",
                          'dft_functional' : "pbe",
                          'mcscf_type'    : "casci",
                          'scf_solver'    : "rks",
                          'active_space'  : [2,2],
                          'active_orb'    : [],
                          'auto_as'       : False,
                          'display_orbitals' : True,
                          'read_mc_mo'    : False,
                          'molel_read_dir' : "",
                          'scf_attempts'  : 1,
                          'diis_start_cycle' : 1,
                          'use_chkfile'   : False,
                          'scf_algo'      : "adiis",
                          'damping'       : 0,
                          'level_shift'   : 0,
                          'scf_guess'     : 'minao',
                          'frac_occ'      : 'false',
                          'molcas_supercell'      : False,
                          'aux_basis'     : "df_default",
                          'charge'        : 0,
                          'spin'          : 0,
                          'smearing' : None,
                          'smearing_width' : 0.05,
                          'remove_linear_dep' : True}
  self.param_update(**kwargs)

 def param_update(self,**kwargs):
  inp=self.input_parameters
  for key in kwargs:
   if key in inp:
    inp[key] = kwargs[key]
   elif key not in inp:
    raise KeyError('%r not a vaild keyword. Please check your input parameters.' % key)

 def calc_setup(self):
  inp=self.input_parameters
  outputfile=open(inp['output']+".out",'w')
  sys.stderr=open(inp['output']+".err",'w')
  

  pyscf_settings=[inp['es_method'],inp['mcscf_type'],inp['active_space'],inp['scf_solver'],inp['dft_functional'],inp['verbosity'],inp['active_orb'],inp['auto_as'],inp['display_orbitals'],inp['output'],inp['state_num'],inp['trans_state'],inp['aux_basis']]
  pyscf_conv_settings=[inp['max_iter'],inp['conv_tol'],inp['diis_start_cycle'],inp['damping'],inp['level_shift'],inp['scf_algo'],inp['scf_guess'],inp['read_mc_mo'],inp['molel_read_dir'],inp['frac_occ'],inp['charge'],inp['spin'],inp['smearing'],inp['smearing_width'],inp['remove_linear_dep']]

  print("Performing non-self-consistent NEGF calculations using metal wide band limit approximation for electrodes",file=outputfile)
  print("Using RUQT-Fortran to calculate transport",file=outputfile)
  print("Running Calculation using the following paramaters:",file=outputfile)
  print(self.input_parameters,file=outputfile)
  energies=np.arange(inp['min_trans_energy'],inp['max_trans_energy']+inp['delta_energy'],inp['delta_energy'])
  bias=np.arange(inp['min_bias'],inp['max_bias']+inp['delta_bias'],inp['delta_bias'])
  bias=np.where(abs(bias)<0.01, 0.01, bias)

  if inp['exmol_prog']=="molcas":
   print("Using Molcas calculation at "+inp['exmol_dir']+" for extended molecular region",file=outputfile)
   print("Using the effective Hamiltonian for electronic state "+str(inp['state_num'])+" of extended mol. region",file=outputfile)
  elif inp['exmol_prog']=="pyscf":
   print("Calculating extended molecular region using Pyscf with "+pyscf_settings[4],file=outputfile)

  if inp['exmol_prog']=="molcas":
   h,s,norb,numelec,actorb,actelec,states=ruqt.esc_molcas2(inp['exmol_dir'],"MolEl.dat",inp['state_num'],outputfile)
   if inp['molcas_supercell']==True:
    elec_orb=0
   else:
    geo2=gto.M(atom=inp['exmol_dir']+inp['exmol_geo'],basis=inp['basis_set'],ecp=inp['ecp'])
    ao_data=gto.mole.ao_labels(geo2,fmt=False)
    atom_num=0
    ao_index=0
    elec_orb=0
    ao_data_len=len(ao_data)
    while atom_num < inp['num_elec_atoms']:
     atom_num=ao_data[ao_index][0]
     ao_index+=1
     if ao_index == ao_data_len:
      print("The # of electrode atoms is incorrect")
      break
    elec_orb=ao_index-1

   #h,s=ruqt.esc_molcas(exmol_file,exmol_dir,exmol_molcasd,state_num,outputfile)
  elif inp['exmol_prog']=="pyscf":
   if inp['pyscf_pbc']==True:
    h,s,norb,numelec=ruqt.esc_pyscf_pbc(inp['exmol_dir']+inp['exmol_geo'],pyscf_settings[4],inp['basis_set'],inp['ecp'],inp['lattice_v'],inp['meshnum'],inp['cell_dim'],pyscf_settings,pyscf_conv_settings)
   else:
    h,s,norb,numelec,elec_orb=ruqt.esc_pyscf2(inp['exmol_dir']+inp['exmol_geo'],pyscf_settings[4],inp['basis_set'],inp['ecp'],inp['num_elec_atoms'],pyscf_settings,pyscf_conv_settings)

  return(energies,bias,outputfile,h,s,norb,numelec,elec_orb)

 def transmission(self):
  inp=self.input_parameters
  energies,bias,outputfile,h,s,norb,numelec,elec_orb=wbl_negf.calc_setup(self)
  print("Calculating "+str(len(energies))+" transmission energies from: "+str(max(energies))+" eV to "+str(min(energies))+" eV",file=outputfile)
  print("Final transmission values will be printed to "+inp['output']+".trans"+" in relative transmission vs eV",file=outputfile)
  ruqt.fort_inputwrite("T",inp['FermiE'],inp['FermiD'],inp['temp'],inp['max_bias'],inp['min_bias'],inp['delta_bias'],inp['min_trans_energy'],inp['max_trans_energy'],inp['delta_energy'],inp['qc_method'],inp['rdm_type'],inp['exmol_dir'],inp['fort_data'],inp['exmol_prog'],inp['num_elec_atoms'],outputfile,inp['state_num'],norb,numelec,elec_orb,inp['molcas_supercell'])
  T,I=ruqt.fort_calc("RUQT.x","fort_ruqt",energies,bias,"T",outputfile)
  t_plot=plt.plot(energies, T)
  plt.xlabel('E-E(Fermi) (eV)')
  plt.ylabel('Transmission (rel)')
  plt.savefig(inp['output']+"_trans.png")
  plt.clf()

 def current(self):
  energies,bias,outputfile,h,s,norb,numelec,elec_orb=wbl_negf.calc_setup(self)
  inp=self.input_parameters
  print("Performing a Landauer current calculation for the following bias voltage range(V): "+str(bias),file=outputfile)
  print("Calculating "+str(len(energies))+" transmission energies from: "+str(max(energies))+" eV to "+str(min(energies))+" eV",file=outputfile)
  print("Final transmission values will be printed to "+inp['output']+".trans"+" in relative transmission vs eV",file=outputfile)
  print("Final current values will be printed to "+inp['output']+".iv"+" in volts vs ampheres",file=outputfile)
  print("Final conductance values will be printed to "+inp['output']+".con"+" in volts vs G_0",file=outputfile)

  if inp['fort_trans']==False:
   ruqt.fort_inputwrite("C",inp['FermiE'],inp['FermiD'],inp['temp'],inp['max_bias'],inp['min_bias'],inp['delta_bias'],inp['min_trans_energy'],inp['max_trans_energy'],inp['delta_energy'],inp['qc_method'],inp['rdm_type'],inp['exmol_dir'],inp['fort_data'],inp['exmol_prog'],inp['num_elec_atoms'],outputfile,inp['state_num'],norb,numelec,elec_orb,inp['molcas_supercell'])
   T,I=ruqt.fort_calc("RUQT.x","fort_ruqt",energies,bias,"C",outputfile)

  elif inp['fort_trans']==True:
   print("Calculating current with ASE transport",file=outputfile)
   ruqt.fort_inputwrite("T",inp['FermiE'],inp['FermiD'],inp['temp'],inp['max_bias'],inp['min_bias'],inp['delta_bias'],inp['min_trans_energy'],inp['max_trans_energy'],inp['delta_energy'],inp['qc_method'],inp['rdm_type'],inp['exmol_dir'],inp['fort_data'],inp['exmol_prog'],inp['num_elec_atoms'],outputfile,inp['state_num'],norb,numelec,elec_orb,inp['molcas_supercell'])
   T,I=ruqt.fort_calc("RUQT.x","fort_ruqt",energies,bias,"T",outputfile)
   
   h=np.zeros((2,2))
   h1=np.zeros((2,2))
   calc = transport.TransportCalculator(h=h, h1=h1, energies=energies,logfile="temp")
   if inp['ase_current']==True:
    I = calc.get_current(bias,T=inp['temp'],E=energies,T_e=T,spinpol=inp['spin_pol'])
   else:
    I = ruqt.get_current(bias,T=inp['temp'],E=energies,T_e=T,spinpol=inp['spin_pol'],delta_e=inp['delta_energy'])
   I=2.*units._e**2/units._hplanck*I
 
  t_plot=plt.plot(energies, T)
  plt.xlabel('E-E(Fermi) (eV)')
  plt.ylabel('Transmission (rel)')
  plt.savefig(inp['output']+"_trans.png")
  plt.clf()

  i_plot=plt.plot(bias,I)
  plt.xlabel('Voltage (V)')
  plt.ylabel('Current (A)')
  plt.savefig(inp['output']+"_current.png")
  b_range=len(bias)
  cond=np.zeros(b_range)
  for x in range(len(bias)):
   cond[x]=I[x]/(bias[x]*2.*units._e**2/units._hplanck)

  np.savetxt(inp['output']+".iv",np.c_[bias,I],fmt="%s")
  plt.clf()
  c_plot=plt.plot(bias,cond)
  plt.xlabel('Voltage (V)')
  plt.ylabel('Conductance (G_0)')
  plt.savefig(inp['output']+"_conductance.png")
  np.savetxt(inp['output']+".con",np.c_[bias,cond],fmt="%s")


 def diff_conductance(self):
  energies,bias,outputfile,h,s,norb,numelec,elec_orb=wbl_negf.calc_setup(self)
  inp=self.input_parameters
  print("Calculating differential conductance using numerical derivatives",file=outputfile)
  print("Calculting each value using the +/-"+str(inp['fd_change'])+" voltage points around it.",file=outputfile)
  print("Performing the diff. conductance calculation for the following bias voltage range(V): "+str(bias),file=outputfile)
  print("Calculating "+str(len(energies))+" transmission energies from: "+str(max(energies))+" eV to "+str(min(energies))+" eV around electrode Fermi level.",file=outputfile)
  print("Final transmission values will be printed to "+inp['output']+".trans"+" in relative transmission vs eV",file=outputfile)
  print("Final diff. conductance values will be printed to "+inp['output']+".dcon"+" in volts vs G_0",file=outputfile) 

  print("Not available in RUQT-Fortran. Using RUQT-Fortan transmission with pyRUQT DiffCond calculator.",file=outputfile)
  ruqt.fort_inputwrite("T",inp['FermiE'],inp['FermiD'],inp['temp'],inp['max_bias'],inp['min_bias'],inp['delta_bias'],inp['min_trans_energy'],inp['max_trans_energy'],inp['delta_energy'],inp['qc_method'],inp['rdm_type'],inp['exmol_dir'],inp['fort_data'],inp['exmol_prog'],inp['num_elec_atoms'],outputfile,inp['state_num'],norb,numelec,elec_orb,inp['molcas_supercell'])
  T,I=ruqt.fort_calc("RUQT.x","fort_ruqt",energies,bias,"T",outputfile)
  t_plot=plt.plot(energies, T)
  plt.xlabel('E-E(Fermi) (eV)')
  plt.ylabel('Transmission (rel)')
  plt.savefig(inp['output']+"_trans.png")
  plt.clf()

  h=np.zeros((2,2))
  h1=np.zeros((2,2))
  calc = transport.TransportCalculator(h=h, h1=h1, energies=energies,logfile="temp")

  DE=ruqt.get_diffcond(calc,bias,inp['temp'],energies,T,inp['fd_change'],inp['ase_current'],inp['delta_energy'])
  c_plot=plt.plot(bias,DE)
  plt.xlabel('Voltage (V)')
  plt.ylabel('Diff. Conductance (G_0)')
  plt.savefig(inp['output']+"_diffcon.png")
  np.savetxt(inp['output']+".dcon",np.c_[bias,DE],fmt="%s")

class es_calc:
 def __init__(self, **kwargs):

  self.input_parameters = {'output'    : "es_results",
                          'es_prog' : "pyscf",
                          'es_dir'  : "./",
                          'basis_set'  : None,
                          'ecp'        : None,
                          'es_geo'  : None,
                          'state_num'  : 1,
                          'conv_tol'      : 1E-7,
                          'max_iter'      : 100,
                          'pyscf_pbc'     : False,
                          'lattice_v'     : None,
                          'meshnum'       : None,
                          'verbosity'     : 5,
                          'cell_dim'      : 2,
                          'pbc_spin'      : None,
                          'aux_basis'     : "df_default",
                          'pdos_states'   : [],
                          'eigenchannels' : 0,
                          'es_method'  : "mcpdft",
                          'dft_functional' : "pbe",
                          'mcscf_type'    : "casci",
                          'scf_solver'    : "rks",
                          'active_space'  : [2,2],
                          'active_orb'    : [],
                          'auto_as'       : False,
                          'display_orbitals' : True,
                          'read_mc_mo'    : False,
                          'molel_read_dir' : "",
                          'scf_attempts'  : 1,
                          'diis_start_cycle' : 1,
                          'scf_algo'      : "cdiis",
                          'damping'       : 0,
                          'level_shift'   : 0,
                          'scf_guess'     : 'minao',
                          'frac_occ'      : 'false',
                          'charge'        : 0,
                          'spin'          : 0,
                          'smearing' : None,
                          'smearing_width' : 0.05,
                          'remove_linear_dep' : True}
  self.param_update(**kwargs)

 def param_update(self,**kwargs):
  inp=self.input_parameters
  for key in kwargs:
   if key in inp:
    inp[key] = kwargs[key]
   elif key not in inp:
    raise KeyError('%r not a vaild keyword. Please check your input parameters.' % key)

 def single_point(self):
  inp=self.input_parameters
  outputfile=open(inp['output']+".out",'w')
  sys.stderr=open(inp['output']+".err",'w')

  pyscf_settings=[inp['es_method'],inp['mcscf_type'],inp['active_space'],inp['scf_solver'],inp['dft_functional'],inp['verbosity'],inp['active_orb'],inp['auto_as'],inp['display_orbitals'],inp['output'],inp['state_num'],0,inp['aux_basis']]
  pyscf_conv_settings=[inp['max_iter'],inp['conv_tol'],inp['diis_start_cycle'],inp['damping'],inp['level_shift'],inp['scf_algo'],inp['scf_guess'],inp['read_mc_mo'],inp['molel_read_dir'],inp['frac_occ'],inp['charge'],inp['spin'],inp['smearing'],inp['smearing_width'],inp['remove_linear_dep']]

  print("Performing Standalone PySCF or MOLCAS calculation.",file=outputfile)
  print("Running Calculation using the following paramaters:",file=outputfile)
  print(self.input_parameters,file=outputfile)

  if inp['es_prog']=="molcas":
   print("Running Molcas calculation using geometry in "+inp['es_dir'],file=outputfile)
   print("Calculating electronic state "+str(inp['state_num']),file=outputfile)
  elif inp['es_prog']=="pyscf":
   print("Using Pyscf for electronic structure calculation",file=outputfile)
   if pyscf_settings[0]=="mcpdft":
    print("Using Pyscf for an MC-PDFT calculation",file=outputfile)

  if inp['es_prog']=="molcas":
   print("Currently not supported. Calculator still in development.",file=outputfile)
   #h,s,norb,numelec,actorb,actelec,states=ruqt.esc_molcas2(inp['es_prog'],"MolEl.dat",inp['state_num'],outputfile)
  elif inp['es_prog']=="pyscf":
   if inp['pyscf_pbc']==True:
    h,s,norb,numelec=ruqt.esc_pyscf_pbc(inp['es_dir']+inp['es_geo'],pyscf_settings[4],inp['basis_set'],inp['ecp'],inp['lattice_v'],inp['meshnum'],inp['cell_dim'],pyscf_settings,pyscf_conv_settings)
   else:
    h,s,norb,numelec,elec_orb=ruqt.esc_pyscf2(inp['es_dir']+inp['es_geo'],pyscf_settings[4],inp['basis_set'],inp['ecp'],0,pyscf_settings,pyscf_conv_settings)

  print("Single Point Calculation Complete. Check the .log and .err files for output data/errors.",file=outputfile)
