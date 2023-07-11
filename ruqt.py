import numpy as np
import scipy
from pyscf import gto,dft,scf
from ase import transport,Atoms,units
import matplotlib.pyplot as plt
import string,subprocess

#This is the main module file for the pyRUQT program which contains all functions needed to run pyRUQT.

#These functions are new RUQT routines for getting properties not calculated by ASE

#Calculates differental conductance
def get_diffcond(calc,bias,temp,energies,T,fd_change,ase_current,delta_e):
 p_bias=bias+fd_change
 n_bias=bias-fd_change
 if ase_current==True:
  I_p=calc.get_current(p_bias,T=temp,E=energies,T_e=T)
  I_n=calc.get_current(n_bias,T=temp,E=energies,T_e=T)
 else:
  I_p=ruqt.get_current(p_bias,T=temp,E=energies,T_e=T,delta_e=delta_e)
  I_n=ruqt.get_current(n_bias,T=temp,E=energies,T_e=T,delta_e=delta_e)
 b_range=len(bias)
 DE=np.zeros(b_range)
 for x in range(b_range):
  DE[x]=(I_p[x]-I_n[x])/(p_bias[x]-n_bias[x])
 return DE

#Aligns all diagonal elments of the repeating electrode blocks in H_elec and H_exmol
def full_align(h_mm,h1_ii,s_mm,elec_units):
 import numpy

 n=int(len(h1_ii)/elec_units-1)
 diff=0.0
 for i in range(0,n-1):
  diff += ((h_mm[i,i] - h1_ii[i, i])/s_mm[i, i])
 diff2=diff/(n+1)
 h_mm -= diff2 * s_mm
 return h_mm

#Functions are used by pyRUQT to gather data for NEGF calculations

#These are the routines for older versions of OpenMolcas (pre-July 2022)
def molcas_matread(data_dir,datafile,norb,data_mat):
 with open(data_dir+datafile,'r') as matrixfile:
  for line in matrixfile:
   line_data=line.split()
   data_mat[int(line_data[0])-1,int(line_data[1])-1]=float(line_data[2])
 matrixfile.close()

def molcas_matread_sym(data_dir,datafile,norb,data_mat):
 with open(data_dir+datafile,'r') as matrixfile:
  for line in matrixfile:
   line_data=line.split()
   data_mat[int(line_data[0])-1,int(line_data[1])-1]=float(line_data[2])
   data_mat[int(line_data[1])-1,int(line_data[0])-1]=float(line_data[2])
 matrixfile.close()

def basisfxn_read(data_dir,datafile,outputfile):
 filesearch=open(data_dir+datafile,'r')
 line=""
 while "Basis functions" not in line:
  line=filesearch.readline()
  if "Basis functions" in line:
   norb = int(line.split()[-1])
   print("Basis Functions in "+datafile+": "+str(norb),file=outputfile)
  elif len(line) ==0:
   print("Fatal Error: Can not find basis function count in "+data_dir+datafile)
   break
 filesearch.close
 return norb

def orb_read_molcas(data_dir,exmol_molcasd,datafile,num_elec,outputfile):
 import os
 filesearch=open(data_dir+datafile,'r')
 line=""
 while "Basis functions" not in line:
  line=filesearch.readline()
  if "Basis functions" in line:
   norb = int(line.split()[-1])
   print("Basis Functions in "+datafile+": "+str(norb),file=outputfile)
  elif not line:
   print("Fatal Error: Can not find basis function count in "+data_dir+datafile)
   break

 while " Aufbau " and " Occupied orbitals " not in line:
  line=filesearch.readline()
  if " Aufbau " or " Occupied orbitals " in line:
   numocc = int(line.split()[-1])
   print("Occupied orbitals in "+datafile+": "+str(numocc),file=outputfile)
  elif not line:
   print("Fatal Error: Can not find Aufbau count in "+data_dir+datafile)
   print("Make sure to include and &SCF molecule in your MOLCAS calculation")
   break
 numvirt=norb-numocc

 filesearch.close
 for file in os.listdir(data_dir+exmol_molcasd):
  if file.endswith(".SymInfo"):
   orb_file=os.path.join(data_dir+exmol_molcasd, file)

 filesearch2=open(orb_file)

 line=filesearch2.readline()
 line_data=line.split()

 while "2" not in line_data[1]:
  line=filesearch2.readline()#  line_data=line.split()
  line_data=line.split()
  if not line:
   print("Fatal Error: Can not find electrode orbital number")
   break
 size_elec=num_elec*(int(line_data[0]))
 size_ex=norb-2*size_elec

 filesearch.close
 return norb,numocc,numvirt,size_ex,size_elec

def read_syminfo(data_dir,norb,num_elec,outputfile):
 import os
 for file in os.listdir(data_dir):
  if file.endswith(".SymInfo"):
   orb_file=os.path.join(data_dir, file)

 filesearch2=open(orb_file)

 line=filesearch2.readline()
 line_data=line.split()

 while "2" not in line_data[1]:
  line=filesearch2.readline()#  line_data=line.split()
  line_data=line.split()
  if not line:
   print("Fatal Error: Can not find electrode orbital number",file=outputfile)
   break
 size_elec=num_elec*(int(line_data[0]))
 size_ex=norb-2*size_elec

 filesearch2.close
 return size_ex,size_elec


def esc_molcas(calc_file,calc_dir,data_dir,state_num,outputfile):
 #import numpy as np
 norb=basisfxn_read(calc_dir,calc_file,outputfile)
 h=np.zeros((norb,norb))
 s=np.zeros((norb,norb))
 molcas_matread(calc_dir,data_dir+"FOCK_AO_"+str(state_num),norb,h)
 h=h*27.2114
 molcas_matread_sym(calc_dir,data_dir+"Overlap",norb,s)
 return h,s

#These are the newer Molcas data reading routines
#Reads MolEl.dat files from post-July 2022 Molcas (sandx_fock branch)
def molel_matread(matrixfile,norb,data_mat,mat_type):
 if mat_type=='S':
  line=matrixfile.readline()
  while "Molecular orbital coefficients" not in line:
   line_data=line.split()
   data_mat[int(line_data[0])-1,int(line_data[1])-1]=float(line_data[2])
   data_mat[int(line_data[1])-1,int(line_data[0])-1]=float(line_data[2])
   line=matrixfile.readline()

 elif mat_type=='H':
   line=matrixfile.readline()
   while "State" and "Orbital Energies" not in line:
    line_data=line.split()
    data_mat[int(line_data[0])-1,int(line_data[1])-1]=float(line_data[2])
    line=matrixfile.readline()

def esc_molcas2(data_dir,data_file,state_num,outputfile):
 filesearch=open(data_dir+data_file,'r')
 for i in range (0,2):
  line=filesearch.readline()
 line_data=line.split()
 states,norb,numelec,actorb,actelec=list(map(int,line_data))
 line=filesearch.readline()

 h=np.zeros((norb,norb))
 s=np.zeros((norb,norb))
 print("Reading data for state "+str(state_num)+" out of "+str(states)+" elec. states.",file=outputfile)
 molel_matread(filesearch,norb,s,"S")

 while "Effective Hamiltonian" not in line:
  line=filesearch.readline()

 line=filesearch.readline()
 line_data=line.split()
 while int(line_data[1])!=state_num and "State" not in line:
  line=filesearch.readline()
  line_data=line.split()

 if "Orbital Energies" in line:
  print("Can not find your effective Hamiltonian. Check your MolEl.dat file formatting",file=outputfile)
 elif int(line_data[1])==state_num:
  molel_matread(filesearch,norb,h,"H")
 h=h*27.211396641308
 return h,s,norb,numelec,actorb,actelec,states

#calculates electric structure info (Hamiltonian, Overlap) with PySCF
def esc_pyscf(geofile,dft_functional,basis_set,ecp):
 #from pyscf import gto,dft,scf 
 geo=gto.M(atom=geofile,basis=basis_set,ecp=ecp)
 rks_elec=dft.RKS(geo).set(max_cycle=100)
 rks_elec.xc=dft_functional
 rks_elec.kernel() 
 if rks_elec.converged==False:
  scf.addons.dynamic_level_shift_(rks_elec,factor=0.5)
  rks_elec.damp=0.5
  rks_elec.diis_start_cycle=2
  rks_elec.kernel()  
 h = rks_elec.get_fock()
 h=h*27.2114
 s = rks_elec.get_ovlp()
 return h,s 

#These routines calculate the electrode-molecule coupling when coupling_calc is set to Fock_EX
def calc_coupling(h,s,h1,h2,s1,s2,coupled,elec_units):
 import numpy as np
 #l_h_dim=np.ndarray.shape(h)
 #l_h1_dim=np.ndarray.size(h1)
 l_h=len(h)
 l_h1_0=len(h1)
 l_h1=l_h1_0//elec_units
 l_mol=l_h-2*l_h1
 hc1=np.zeros((l_h1,l_h),dtype=np.complex)
 sc1=np.zeros((l_h1,l_h),dtype=np.complex)
 if h2==None:
  hc2=np.zeros(shape=(l_h1,l_h),dtype=np.complex)
  sc2=np.zeros(shape=(l_h1,l_h),dtype=np.complex)
 else:
  l_h2_0=len(h2)
  l_h2=l_h2_0//elec_units
  hc2=np.zeros(shape=(l_h2,l_h),dtype=np.complex)
  sc2=np.zeros(shape=(l_h2,l_h),dtype=np.complex)

 if coupled=="molecule":
  hc1[:l_h1,:l_h1]=h1[:l_h1,l_h1:2*l_h1]
  sc1[:l_h1,:l_h1]=s1[:l_h1,l_h1:2*l_h1]
  hc1[:l_h1,l_h1:(l_h-l_h1)]=h[:l_h1,l_h1:(l_h-l_h1)]
  sc1[:l_h1,l_h1:(l_h-l_h1)]=s[:l_h1,l_h1:(l_h-l_h1)]
  if h2==None:
   hc2[-l_h1:,-l_h1:]=h1[l_h1:2*l_h1,:l_h1]
   sc2[-l_h1:,-l_h1:]=s1[l_h1:2*l_h1,:l_h1]
   hc2[-l_h1:,-(l_mol+l_h1):-l_h1]=np.transpose(h[l_h1:(l_h-l_h1),:l_h1])
   sc2[-l_h1:,-(l_mol+l_h1):-l_h1]=np.transpose(s[l_h1:(l_h-l_h1),:l_h1])
  else:
   hc2[-l_h2:,-l_h2:]=h2[l_h2:2*l_h2,:l_h2]
   sc2[-l_h2:,-l_h2:]=s2[l_h2:2*l_h2,:l_h2]
   hc2[-l_h2:,-(l_mol+l_h2):-l_h2]=np.transpose(h[l_h2:(l_h-l_h2),:l_h2])
   sc2[-l_h2:,-(l_mol+l_h2):-l_h2]=np.transpose(s[l_h2:(l_h-l_h2),:l_h2])

 elif coupled=="extended_molecule":
  hc1[:l_h1,:l_h1]=h1[:l_h1,l_h1:2*l_h1]
  sc1[:l_h1,:l_h1]=s1[:l_h1,l_h1:2*l_h1]
  hc1[:l_h1,l_h1:l_h]=h[:l_h1,l_h1:l_h]
  sc1[:l_h1,l_h1:l_h]=s[:l_h1,l_h1:l_h]
  if h2==None:
   hc2[-l_h1:,-l_h1:]=h1[l_h1:2*l_h1,:l_h1]
   sc2[-l_h1:,-l_h1:]=s1[l_h1:2*l_h1,:l_h1]
   hc2[-l_h1:,-l_h:-l_h1]=np.transpose(h[l_h1:l_h,:l_h1])
   sc2[-l_h1:,-l_h:-l_h1]=np.transpose(s[l_h1:l_h,:l_h1])
  else:
   hc2[-l_h2:,-l_h2:]=h2[l_h2:2*l_h2,:l_h2]
   sc2[-l_h2:,-l_h2:]=s1[l_h2:2*l_h2,:l_h2]
   hc2[-l_h2:,-l_h:-l_h2]=np.transpose(h[l_h2:l_h,:l_h2])
   sc2[-l_h2:,-l_h:-l_h2]=np.transpose(s[l_h2:l_h,:l_h2])

 return hc1,sc1,hc2,sc2

#The routines below are for creating inputs/calling/getting data from RUQT-Fortran transport calculations from pyRUQT

#These routines extract detailed paritioning data from RUQT-Fortan (currently unused).
def read_ruqtfortran_partdat(ruqt_dir,ruqt_file,elec_units):
 import numpy as np
 with open(ruqt_dir+ruqt_file+".partdat",'r') as matrixfile:
  line=matrixfile.readline()
  line_data=line.split()
  size_l=int(line_data[0])
  size_r=int(line_data[2])
  size_c=int(line_data[1])

  h=np.zeros((size_c,size_c),dtype=np.complex)
  s=np.zeros((size_c,size_c),dtype=np.complex)
  h1=np.zeros(shape=(size_l,size_l),dtype=np.complex)
  s1=np.zeros(shape=(size_l,size_l),dtype=np.complex)
  h2=np.zeros(shape=(size_r,size_r),dtype=np.complex)
  s2=np.zeros(shape=(size_r,size_r),dtype=np.complex)

  size_lr=size_l+size_c
  r_size=size_r//elec_units

  #Read in Overlap Matrices
  line=matrixfile.readline()
  for i in range(1,size_l):
   for j in range(1,size_l):
    line=matrixfile.readline()
    line_data=line.split()
    h1[int(line_data[0])-1,int(line_data[1])-1]=float(line_data[2])

  line=matrixfile.readline()
  for i in range(1,size_c):
   for j in range(1,size_c):
    line=matrixfile.readline()
    line_data=line.split()
    h[int(line_data[0])-1-size_l,int(line_data[1])-1-size_l]=float(line_data[2])

  line=matrixfile.readline()
  for i in range(1,size_r):
   for j in range(1,size_r):
    line=matrixfile.readline()
    line_data=line.split()
    h2[int(line_data[0])-size_lr,int(line_data[1])-1-size_lr]=float(line_data[2])

  #Read in Hamiltonian Matrices
  line=matrixfile.readline()
  for i in range(1,size_l):
   for j in range(1,size_l):
    line=matrixfile.readline()
    line_data=line.split()
    s1[int(line_data[0])-1,int(line_data[1])-1]=float(line_data[2])

  line=matrixfile.readline()
  for i in range(1,size_c):
   for j in range(1,size_c):
    line=matrixfile.readline()
    line_data=line.split()
    s[int(line_data[0])-1-size_l,int(line_data[1])-1-size_l]=float(line_data[2])

  line=matrixfile.readline()
  for i in range(1,size_r):
   for j in range(1,size_r):
    line=matrixfile.readline()
    line_data=line.split()
    s2[int(line_data[0])-1-size_lr,int(line_data[1])-1-size_lr]=float(line_data[2])

 matrixfile.close()
 return h,h1,h2,s,s1,s2

def read_ruqtfortran_sigma(ruqt_dir,ruqt_file,elec_units):
   
 with open(ruqt_dir+ruqt_file+".partdat",'r') as matrixfile:
  line=matrixfile.readline()
  line_data=line.split()
  size_l=int(line_data[0])
  size_r=int(line_data[2])
  size_c=int(line_data[1])

  l_size=size_l//elec_units
  r_size=size_r//elec_units
  sigma1=np.zeros(shape=(l_size,size_c),dtype=np.complex)
  sigma2=np.zeros(shape=(r_size,size_c),dtype=np.complex) 

  line=matrixfile.readline()
  for i in range(1,size_c):
   for j in range(1,size_c):
    line=matrixfile.readline()
    if i <= l_size:
     line_data=line.split()
     sigma1[line_data[0]-1,line_data[1]-1]=line_data[2]

  line=matrixfile.readline()
  for i in range(1,size_c):
   for j in range(1,size_c):
    line=matrixfile.readline()
    if i <= r_size:
     line_data=line.split()
     sigma2[int(line_data[0])-1,line_data[1]-1]=line_data[2]

 matrixfile.close()
 return sigma1,sigma2


#The next two functions create RUQT-Fortran inputs and run RUQT.x calculations
def fort_inputwrite(cal_typ,FermiE,Fermi_Den,temp,max_bias,min_bias,delta_bias,min_trans_energy,max_trans_energy,delta_energy,qc_method,rdm_type,exmol_dir,exmol_file,exmol_prog,num_elec_atoms,outputfile,state_num,norb,numelec):
 import string

#This part converts the python variables to ones reconized by the Fortran code
 if cal_typ == "T":
  Calc_Type="transmission"
 elif cal_typ == "C":
  Calc_Type="current"
 else:
  print("RUQT Fortran calculator only supports current and transmission calculations",file=outputfile)
  quit()

 Electrode_Type = "Metal_WBL" #Only option currently for RUQT-Fortran

 if qc_method=="rdm":
  rdm_doubles="T"
 elif qc_method=="neo":
  qc_code="pyscf"
  rdm_doubles="F"
  qc_method="hf"
 else:
  rdm_doubles="F"
 
 KT=temp*8.617333262E-5

 if rdm_type==1:
  use_b0="F"
  b0_type="rdm"
 elif rdm_type==2:
  use_b0=="T"
  b0_type="cisd"
 else:
  print("RDM calculation selection not supported. Please check README for options.")

 if exmol_prog=="molcas":
  cp_fock='cp '+exmol_dir+"/MolEl.dat"+' MolEl.dat'

  cpdata_1=subprocess.Popen(cp_fock,shell=True)
  cpdata_1.wait()
  #h,s,norb,numelec,actorb,actelec,states=esc_molcas2(exmol_dir,"MolEl.dat",state_num,outputfile) 
  size_ex,size_elec=read_syminfo(exmol_dir,norb,num_elec_atoms,outputfile)  
  numocc=int(numelec/2)
  numvirt=norb-numocc
 elif exmol_prog=="maple" or exmol_prog=="pyscf":
  norb,numocc,numvirt,size_ex,size_elec=orb_read_scfdat(exmol_dir,exmol_file,num_elec,outputfile)

#This part writes the options to the input file for the RUQT Fortran code
 negf_inp = open("fort_ruqt",'w')
 negf_inp.write("{0}".format(Calc_Type) +  "\n")
 negf_inp.write("{0}".format(Electrode_Type) +  "\n")
 negf_inp.write("{0}".format(FermiE) + "\n")
 negf_inp.write("{0}".format(FermiE) + "\n")
 negf_inp.write("{0}".format(Fermi_Den) + "\n")
 negf_inp.write("{0}".format(Fermi_Den) + "\n")
 negf_inp.write("{0}".format(norb) + "\n")
 negf_inp.write("{0}".format(norb) + "\n")
 negf_inp.write("{0}".format(0) + "\n")
 negf_inp.write("{0}".format(0) + "\n")
 negf_inp.write("{0}".format(numocc) + "\n")
 negf_inp.write("{0}".format(numvirt) + "\n")
 negf_inp.write("{0}".format(size_ex) + "\n")
 negf_inp.write("{0}".format(size_elec) + "\n")
 negf_inp.write("{0}".format(size_elec) + "\n")
 negf_inp.write("{0}".format(min_trans_energy+FermiE) + "\n")
 negf_inp.write("{0}".format(max_trans_energy+FermiE) + "\n")
 negf_inp.write("{0}".format(delta_energy) + "\n")
 negf_inp.write("{0}".format(min_bias) + "\n")
 negf_inp.write("{0}".format(max_bias) + "\n")
 negf_inp.write("{0}".format(delta_bias) + "\n")
 negf_inp.write("{0}".format(KT) + "\n")
 negf_inp.write("{0}".format(exmol_prog) + "\n")
 negf_inp.write("{0}".format(rdm_doubles) + "\n")
 negf_inp.write("{0}".format(qc_method) + "\n")
 negf_inp.write("{0}".format(use_b0) + "\n")
 negf_inp.write("{0}".format(b0_type) + "\n")
 negf_inp.write("{0}".format("F") + "\n")
 negf_inp.write("{0}".format("1") + "\n") 
 negf_inp.write("{0}".format(state_num)+"\n")
 negf_inp.close()

def fort_calc(ruqt_exe,calcname,energies,bias,calc_type,outputfile):
 import subprocess,string
 import numpy as np
 #This routine calls the RUQT Fortran transport calculator
 #Make sure to have a working RUQT.x executable compiled from the Github source code (ruqt_engine branch)

 run_com=ruqt_exe+' '+calcname
 
 ruqt_fort=subprocess.Popen(run_com,shell=True,stdout=outputfile)
 ruqt_fort.wait()

 T=np.zeros(len(energies),dtype=float)
 I=np.zeros(len(bias),dtype=float)

 negffile=open(calcname+".negf_dat",'r')
 line=negffile.readline()
 num=int(line)
 
 for i in range(0,num):
  line=negffile.readline()
  line_data=line.split()
  T[i]=float(line_data[1])

 if calc_type=="C":
  line=negffile.readline()
  num=int(line)
  for i in range(0,num):
   line=negffile.readline()
   line_data=line.split()
   I[i]=float(line_data[1])

 negffile.close()
 return T,I

#These routines are improved and/or fixed versions of ASE functions for SIE calculations.
def get_current(bias, T=None, E=None, T_e=None, spinpol=False, delta_e=None):

 kB=8.617333262E-5

 if not isinstance(bias, (int, float)):
   bias = bias[np.newaxis]
   E = E[:, np.newaxis]
 #  T_e = T_e[:, np.newaxis]

 fl = f_fermidistribution(E - bias / 2., kB * T)
 fr = f_fermidistribution(E + bias / 2., kB * T)

 iv=np.zeros(bias.size,dtype=float)
 fl_dat=np.asarray(fl)
 fr_dat=np.asarray(fr)
 f_dat=fl_dat-fr_dat 

 if spinpol:
  spin=0.5
 else:
  spin=1.0

 for b in range(bias.size):
  for k in range(E.size):
   iv[b]+=spin * f_dat[k,b] * T_e[k]*delta_e

 return iv

def f_fermidistribution(energy, kt):
    # fermi level is fixed to zero
    # energy can be a single number or a list
 assert kt >= 0., 'Negative temperature encountered!'

 if kt == 0:
  if isinstance(energy, float):
   return int(energy / 2. <= 0)
  else:
   return (energy / 2. <= 0).astype(int)
 else:
   #return 1. / (1. + np.exp(energy/kt))
   return 1. / (1. + np.exp(energy)**(1/kt))


#Routine to make .scf_dat file used by older RUQT-Fortran versions
def make_scfdat(mo_coeff,mo_energies,overlap,fock_mat,calcname):
 import numpy
 
 norb=len(mo_energies)-1
 scffile=open(calcname+".scf_dat",'w')
 
 scffile.write("{0}".format("Molecular Orbital Coefficients"))
 for x in range(0,norb):
  for y in range(0,norb):
  #scffile.write("{0},".format(x) + "{0},".format(y) +  "{0}".format(Smat[x,y]) + "\n")
   scffile.write(x+y+"{0}".format(mo_coeff[x,y]) + "\n")
 
 scffile.write("{0}".format("Molecular Orbital Energies"))
 for x in range(0,norb):
  #scffile.write("{0},".format(x) + "{0},".format(y) +  "{0}".format(Smat[x,y]) + "\n")
  scffile.write(x+"{0}".format(mo_energies[x]) + "\n")

 scffile.write("{0}".format("Overlap Matrix"))
 for x in range(0,norb):
  for y in range(0,norb):
  #scffile.write("{0},".format(x) + "{0},".format(y) +  "{0}".format(Smat[x,y]) + "\n")
   scffile.write(x+y+"{0}".format(overlap[x,y]) + "\n")

 scffile.write("{0}".format("Fock Matrix"))
 for x in range(0,norb):
  for y in range(0,norb):
  #scffile.write("{0},".format(x) + "{0},".format(y) +  "{0}".format(Smat[x,y]) + "\n")
   scffile.write(x+y+"{0}".format(fock_mat[x,y]) + "\n")
