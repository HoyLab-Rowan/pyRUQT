from ase import transport,Atoms,units
import matplotlib.pyplot as plt
import numpy as np
import ruqt
from numpy import linalg

class sie_negf:
 def __init__(self, **kwargs):

  """
  Parameters for semi-infinite electrode calculation using ASE

  Keywords Required Each Calculation:
  ouput : str
  exmol_dir : {None,str}
  elec_dir : {None,str}

  Calculation Dependent or Optional Keywords:
  exmol_prog : {"molcas",str}, optional
  elec_prog : {"molcas",str},optional
  run_molcas : {False,bool}, optional
  min_trans_energy : {-2.0,float},optional
  max_trans_energy : {2.0,float},optional
  delta_energy : {0.01,float},optional
  min_bias : {-2.0,float},optional
  max_bias : {2.0,float},optional
  delta_bias : {0.1,float},optioinal
  full_align : {True,bool},optional
  dft_functional : {"pbe",str},optional
  basis_set : {"lanl2dz",str},optional
  ecp : {"lanl2dz",None,str},optional
  n_elec_units : {2,int},optional
  exmol_geo : {None,str},optional
  elec_geo : {None,str},optional
  state_num : {1,int},optional
  state_num_e : {1,int},optional
  coupling_calc : {"none",str},optional
  coupled : {"coupled",str},optional
  spin_pol : {False,logical},optional
  align_elec : {0,int},optional
  dos_calc : {False,bool},optional
  fd_change : {0.001,float},optional
  """
  self.input_parameters = {'output'    : "pyruqt_results",
                          'exmol_prog' : "molcas",
                          'exmol_dir'  : None,
                          'elec_prog'  : "molcas",
                          'elec_dir'   : None,
                          'run_molcas'  : False,
                          'min_trans_energy' : -2.0,
                          'max_trans_energy' : 2.0,
                          'delta_energy' : 0.001,
                          'min_bias'     : -2,
                          'max_bias'     : 2,
                          'delta_bias'   : 0.1,
                          'full_align'   : True,
                          'dft_functional' : "pbe",
                          'basis_set'  : "lanl2dz",
                          'ecp'        : "lanl2dz",
                          'n_elec_units'   : 2,
                          'exmol_geo'  : None,
                          'elec_geo'   : None,
                          'state_num'  : 1, 
                          'state_num_e' : 1,
                          'coupling_calc' : "none",
                          'coupled'       : "molecule",
                          'spin_pol'      : False,
                          'align_elec'    : 0,
                          'dos_calc'      : False,
                          'fd_change'     : 0.001} 

  print("Running Calculation using the following paramaters:",file=self.input_parameters['output']+'.out')
  self.param_update(**kwargs)
  print(self.input_parameters,file=self.input_parameters['output']+'.out')

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
  print("Performing non-self-consistent NEGF transport calculations using semi-infinite tight-binding electrodes")
  print("Using Atomic Simulation Environment to calculate electrode interactions and transport",file=outputfile)
  energies=np.arange(inp['min_trans_energy'],inp['max_trans_energy']+inp['delta_energy'],inp['delta_energy'])
  bias=np.arange(inp['min_bias'],inp['max_bias']+inp['delta_bias'],inp['delta_bias'])
  bias=np.where(abs(bias)<0.01, 0.01, bias)

  if inp['exmol_prog']=="molcas":
   print("Using Molcas calculation at "+inp['exmol_dir']+" for extended molecular region",file=outputfile)
   print("Using the effective Hamiltonian for electronic state "+str(inp['state_num'])+" of extended mol. region",file=outputfile)
  elif inp['exmol_prog']=="pyscf":
   print("Calculating extended molecular region using Pyscf with "+inp['dft_functional']+" in "+inp['basis_set']+" basis set",file=outputfile)

  if inp['elec_prog']=="molcas":
   print("Using Molcas calculation at "+inp['elec_dir']+" for left electrode",file=outputfile)
   print("Using the effective Hamiltonian for electronic state "+str(inp['state_num'])+" of extended mol. region",file=outputfile)
   print("Assuming symmetric electrodes",file=outputfile)

  if inp['exmol_prog']=="molcas":
   h,s,norb,numelec,actorb,actelec,states=ruqt.esc_molcas2(inp['exmol_dir'],"MolEl.dat",inp['state_num'],outputfile)
   #h,s=ruqt.esc_molcas(exmol_file,exmol_dir,exmol_molcasd,state_num,outputfile)
  elif inp['exmol_prog']=="pyscf":
   h,s=ruqt.esc_pyscf(inp['exmol_dir']+inp['exmol_geo'],inp['dft_functional'],inp['basis_set'],inp['ecp'])

  if inp['elec_prog']=="molcas":
   h1,s1,norb_le,numelec_le,actorb_le,actelec_le,states_le=ruqt.esc_molcas2(inp['elec_dir'],"MolEl.dat",inp['state_num'],outputfile)
  elif self.elec_prog=="pyscf":
   h1,s1=ruqt.esc_pyscf(inp['elec_dir']+inp['elec_geo'],inp['dft_functional'],inp['basis_set'],inp['ecp'])
  if inp['coupling_calc']=="Fock_EX":
   hc1,sc1,hc2,sc2=ruqt.calc_coupling(h,s,h1,s1,inp['coupled'],inp['n_elec_units'])
  else:
   hc1=None
   sc1=None
   hc2=None
   sc2=None

  return(energies,bias,outputfile,h,h1,s,s1,hc1,sc1,hc2,sc2)

 def transmission(self):
  energies,bias,outputfile,h,h1,s,s1,hc1,sc1,hc2,sc2=sie_negf.calc_setup(self)
  inp=self.input_parameters
  print("Calculating "+str(len(energies))+" transmission energies from: "+str(max(energies))+" eV to "+str(min(energies))+" eV",file=outputfile)
  print("Final transmission values will be printed to "+inp['output']+".trans"+" in relative transmission vs eV",file=outputfile)

 
  print("Performing NEGF Transport Calculations using the Atomic Simulation Environment",file=outputfile)
  if inp['full_align']==True:
   h=ruqt.full_align(h,h1,s,inp['n_elec_units'])
  if inp['align_elec']>=1:
   inp['align_elec']-=1
   print("Aligning the "+str(inp['align_elec'])+" element of both electrode and extended molecule",file=outputfile)
   calc = transport.TransportCalculator(h=h, h1=h1, s=s, s1=s1,hc1=hc1,sc1=sc1,hc2=hc2,sc2=sc2, energies=energies,dos=inp['dos_calc'],logfile=inp['output']+".trans",align_bf=inp['align_elec'])
  elif inp['align_elec']<1:
   calc=transport.TransportCalculator(h=h, h1=h1, s=s, s1=s1,hc1=hc1,sc1=sc1,hc2=hc2,sc2=sc2, energies=energies,dos=inp['dos_calc'],logfile=inp['output']+".trans")
  T = calc.get_transmission()
  t_plot=plt.plot(energies, T)
  plt.xlabel('E-E(Fermi) (eV)')
  plt.ylabel('Transmission (rel)')
  plt.savefig(inp['output']+"_trans.png")
  plt.clf()
  return T 

 def current(self):
  energies,bias,outputfile,h,h1,s,s1,hc1,sc1,hc2,sc2=sie_negf.calc_setup(self)
  inp=self.input_parameters
  print("Performing a Landauer current calculation for the following bias voltage range(V): "+str(bias),file=outputfile)
  print("Calculating "+str(len(energies))+" transmission energies from: "+str(max(energies))+" eV to "+str(min(energies))+" eV",file=outputfile)
  print("Final transmission values will be printed to "+output+".trans"+" in relative transmission vs eV",file=outputfile)
  print("Final current values will be printed to "+output+".iv"+" in volts vs ampheres",file=outputfile)
  print("Final conductance values will be printed to "+output+".con"+" in volts vs G_0",file=outputfile)

  T=sie_negf.transmission(self)
  I = calc.get_current(bias,T=inp['temp'],E=energies,T_e=T,spinpol=inp['spin_pol'])
  i_plot=plt.plot(bias,2.*units._e**2/units._hplanck*I)
  plt.xlabel('Voltage (V)')
  plt.ylabel('Current (A)')
  plt.savefig(output+"_current.png")

  b_range=len(bias)
  cond=np.zeros(b_range)
  for x in range(len(bias)):
   cond[x]=I[x]/bias[x]
  I=2.*units._e**2/units._hplanck*I
  np.savetxt(self.output+".iv",np.c_[bias,I],fmt="%s")

  plt.clf()
  c_plot=plt.plot(bias,cond)
  plt.xlabel('Voltage (V)')
  plt.ylabel('Conductance (G_0)')
  plt.savefig(self.output+"_conductance.png")
  np.savetxt(self.output+".con",np.c_[bias,cond],fmt="%s")

 def diff_conductance(self):
  energies,bias,outputfile,h,h1,s,s1,hc1,sc1,hc2,sc2=sie_negf.calc_setup(self)
  inp=self.input_parameters
  print("Calculating differential conductance using numerical derivatives",file=outputfile)
  print("Calculting each value using the +/-"+str(inp['fd_change'])+" voltage points around it.",file=outputfile)
  print("Performing the diff. conductance calculation for the following bias voltage range(V): "+str(bias),file=outputfile)
  print("Calculating "+str(len(energies))+" transmission energies from: "+str(max(energies))+" eV to "+str(min(energies))+" eV around electrode Fermi level.",file=outputfile)
  print("Final transmission values will be printed to "+inp['output']+".trans"+" in relative transmission vs eV",file=outputfile)
  print("Final diff. conductance values will be printed to "+inp['output']+".dcon"+" in volts vs G_0",file=outputfile)
 
  T=sie_negf.transmission(self)

  DE=ruqt.get_diffcond(calc,bias,inp['temp'],energies,T,inp['fd_change'])

  c_plot=plt.plot(bias,DE)
  plt.xlabel('Voltage (V)')
  plt.ylabel('Diff. Conductance (G_0)')
  plt.savefig(inp['output']+"_diffcon.png")
  np.savetxt(inp['output']+".dcon",np.c_[bias,DE],fmt="%s")


class wbl_negf:
 def __init__(self, **kwargs):
  
  self.input_parameters = {'output'     : "pyruqt_results",
                          'exmol_dir'  : None,
                          'num_elec_atoms' : None,
                          'exmol_prog' : "molcas",
                          'run_molcas'  : False,
                          'min_trans_energy' : -2.0,
                          'max_trans_energy' : 2.0,
                          'delta_energy' : 0.01,
                          'min_bias'     : -2,
                          'max_bias'     : 2,
                          'delta_bias'   : 0.1,
                          'dft_functional' : "pbe",
                          'basis_set'  : "lanl2dz",
                          'ecp'        : "lanl2dz",
                          'exmol_geo'  : None,
                          'state_num'  : 1,
                          'FermiE'     :-5.30,
                          'FermiD'     : 0.07,
                          'qc_method'  : "dft",
                          'rdm_type'   : 1,
                          'fort_data'  : None,
                          'fort_trans' : False,
                          'fd_change'  : 0.001}
  print("Running Calculation using the following paramaters:",file=self.input_parameters['output']+'.out')
  self.param_update(**kwargs)
  print(self.input_parameters,file=self.input_parameters['output']+'.out')

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
  print("Performing non-self-consistent NEGF calculations using metal wide band limit approximation for electrodes",file=outputfile)
  print("Using RUQT-Fortran to calculate transport",file=outputfile)
  energies=np.arange(min_trans_energy,max_trans_energy+delta_energy,delta_energy)
  bias=np.arange(min_bias,max_bias+delta_bias,delta_bias)
  bias=np.where(abs(bias)<0.01, 0.01, bias)

  if inp['exmol_prog']=="molcas":
   print("Using Molcas calculation at "+inp['exmol_dir']+" for extended molecular region",file=outputfile)
   print("Using the effective Hamiltonian for electronic state "+str(inp['state_num'])+" of extended mol. region",file=outputfile)
  elif inp['exmol_prog']=="pyscf":
   print("Calculating extended molecular region using Pyscf with "+inp['dft_functional']+" in "+inp['basis_set']+" basis set",file=outputfile)

  if inp['exmol_prog']=="molcas":
   h,s,norb,numelec,actorb,actelec,states=ruqt.esc_molcas2(inp['exmol_dir'],"MolEl.dat",inp['state_num'],outputfile)
   #h,s=ruqt.esc_molcas(exmol_file,exmol_dir,exmol_molcasd,state_num,outputfile)
  elif inp['exmol_prog']=="pyscf":
   h,s=ruqt.esc_pyscf(inp['exmol_dir']+inp['exmol_geo'],inp['dft_functional'],inp['basis_set'],inp['ecp'])

  return(energies,bias,outputfile,h,s,norb,numelec)

 def transmission(self):
  inp=self.input_parameters
  energies,bias,outputfile,h,s,norb,numelec=wbl_negf.calc_setup(self)
  print("Calculating "+str(len(energies))+" transmission energies from: "+str(max(energies))+" eV to "+str(min(energies))+" eV",file=outputfile)
  print("Final transmission values will be printed to "+inp['output']+".trans"+" in relative transmission vs eV",file=outputfile)
  ruqt.fort_inputwrite("T",inp['FermiE'],inp['FermiD'],inp['temp'],inp['max_bias'],inp['min_bias'],inp['delta_bias'],inp['min_trans_energy'],inp['max_trans_energy'],inp['delta_energy'],inp['qc_method'],inp['rdm_type'],inp['exmol_dir'],inp['fort_data'],inp['exmol_prog'],inp['num_elec_atoms'],outputfile,inp['state_num'],norb,numelec)
  T,I=ruqt.fort_calc("RUQT.x","fort_ruqt",energies,bias,"T",outputfile)
  t_plot=plt.plot(energies, T)
  plt.xlabel('E-E(Fermi) (eV)')
  plt.ylabel('Transmission (rel)')
  plt.savefig(inp['output']+"_trans.png")
  plt.clf()

 def current(self):
  energies,bias,outputfile,h,s,norb,numelec=wbl_negf.calc_setup(self)
  inp=self.input_parameters
  print("Performing a Landauer current calculation for the following bias voltage range(V): "+str(bias),file=outputfile)
  print("Calculating "+str(len(energies))+" transmission energies from: "+str(max(energies))+" eV to "+str(min(energies))+" eV",file=outputfile)
  print("Final transmission values will be printed to "+output+".trans"+" in relative transmission vs eV",file=outputfile)
  print("Final current values will be printed to "+output+".iv"+" in volts vs ampheres",file=outputfile)
  print("Final conductance values will be printed to "+output+".con"+" in volts vs G_0",file=outputfile)

  if inp['fort_trans']==False:
   ruqt.fort_inputwrite("C",inp['FermiE'],inp['FermiD'],inp['temp'],inp['max_bias'],inp['min_bias'],inp['delta_bias'],inp['min_trans_energy'],inp['max_trans_energy'],inp['delta_energy'],inp['qc_method'],inp['rdm_type'],inp['exmol_dir'],inp['fort_data'],inp['exmol_prog'],inp['num_elec_atoms'],outputfile,inp['state_num'],norb,numelec)
   T,I=ruqt.fort_calc("RUQT.x","fort_ruqt",energies,bias,calc_type,outputfile)

  elif inp['fort_trans']==True:
   print("Calculating current with ASE transport",file=outputfile)
   ruqt.fort_inputwrite("T",inp['FermiE'],inp['FermiD'],inp['temp'],inp['max_bias'],inp['min_bias'],inp['delta_bias'],inp['min_trans_energy'],inp['max_trans_energy'],inp['delta_energy'],inp['qc_method'],inp['rdm_type'],inp['exmol_dir'],inp['fort_data'],inp['exmol_prog'],inp['num_elec_atoms'],outputfile,inp['state_num'],norb,numelec)
   T,I=ruqt.fort_calc("RUQT.x","fort_ruqt",energies,bias,"T",outputfile)
   
   h=np.zeros((2,2))
   h1=np.zeros((2,2))
   calc = transport.TransportCalculator(h=h, h1=h1, energies=energies,logfile="temp")
   I = calc.get_current(bias,T=temp,E=energies,T_e=T,spinpol=spin_pol)
   I=2.*units._e**2/units._hplanck*I
 
  t_plot=plt.plot(energies, T)
  plt.xlabel('E-E(Fermi) (eV)')
  plt.ylabel('Transmission (rel)')
  plt.savefig(inp['output']+"_trans.png")
  plt.clf()

  i_plot=plt.plot(bias,I)
  plt.xlabel('Voltage (V)')
  plt.ylabel('Current (A)')
  plt.savefig(output+"_current.png")
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
  energies,bias,outputfile,h,s,norb,numelec=wbl_negf.calc_setup(self)
  inp=self.input_parameters
  print("Calculating differential conductance using numerical derivatives",file=outputfile)
  print("Calculting each value using the +/-"+str(inp['fd_change'])+" voltage points around it.",file=outputfile)
  print("Performing the diff. conductance calculation for the following bias voltage range(V): "+str(bias),file=outputfile)
  print("Calculating "+str(len(energies))+" transmission energies from: "+str(max(energies))+" eV to "+str(min(energies))+" eV around electrode Fermi level.",file=outputfile)
  print("Final transmission values will be printed to "+inp['output']+".trans"+" in relative transmission vs eV",file=outputfile)
  print("Final diff. conductance values will be printed to "+inp['output']+".dcon"+" in volts vs G_0",file=outputfile) 

  print("Not available in RUQT-Fortran. Using RUQT-Fortan transmission with pyRUQT DiffCond calculator.",file=outputfile)
  ruqt.fort_inputwrite("T",inp['FermiE'],inp['FermiD'],inp['temp'],inp['max_bias'],inp['min_bias'],inp['delta_bias'],inp['min_trans_energy'],inp['max_trans_energy'],inp['delta_energy'],inp['qc_method'],inp['rdm_type'],inp['exmol_dir'],inp['fort_data'],inp['exmol_prog'],inp['num_elec_atoms'],outputfile,inp['state_num'])
  T,I=ruqt.fort_calc("RUQT.x","fort_ruqt",energies,bias,"T",outputfile)
  t_plot=plt.plot(energies, T)
  plt.xlabel('E-E(Fermi) (eV)')
  plt.ylabel('Transmission (rel)')
  plt.savefig(inp['output']+"_trans.png")
  plt.clf()

  h=np.zeros((2,2))
  h1=np.zeros((2,2))
  calc = transport.TransportCalculator(h=h, h1=h1, energies=energies,logfile="temp")

  DE=ruqt.get_diffcond(calc,bias,inp['temp'],energies,T,inp['fd_change'])
  c_plot=plt.plot(bias,DE)
  plt.xlabel('Voltage (V)')
  plt.ylabel('Diff. Conductance (G_0)')
  plt.savefig(inp['output']+"_diffcon.png")
  np.savetxt(inp['output']+".dcon",np.c_[bias,DE],fmt="%s")
