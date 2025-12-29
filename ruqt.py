import numpy as np
import scipy
from pyscf import gto,dft,scf,mcscf,mcpdft,lo,tools
from pyscf.mcscf import avas
from ase import transport,Atoms,units
import matplotlib.pyplot as plt
import string,subprocess
from functools import reduce

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
  I_p=get_current(p_bias,T=temp,E=energies,T_e=T,delta_e=delta_e)
  I_n=get_current(n_bias,T=temp,E=energies,T_e=T,delta_e=delta_e)
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

 while str(num_elec+1) not in line_data[1]:
  line=filesearch2.readline()#  line_data=line.split()
  line_data=line.split()
  if not line:
   print("Fatal Error: Can not find electrode orbital number")
   break
 size_elec=line_data[0]-1#num_elec*(int(line_data[0]))
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

 while str(num_elec+1) not in line_data[1]:
  line=filesearch2.readline()#  line_data=line.split()
  line_data=line.split()
  if not line:
   print("Fatal Error: Can not find electrode orbital number",file=outputfile)
   break
 size_elec=int(line_data[0])-1 #num_elec*(int(line_data[0]))
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
def esc_pyscf_pbc(geofile,dft_functional,basis_set,ecp,lattice_v,meshnum,cell_dim,kpts,pyscf_settings,pyscf_conv_settings):
 from pyscf.pbc import gto as pbcgto
 from pyscf.pbc import scf as pbcscf
 from pyscf.pbc import dft as pbcdft
 from pyscf.pbc import df as pdf

 if pyscf_settings[0]=="dft":
  outputfile=open(pyscf_settings[9]+".log",'w')
  if meshnum != None:
   mesh_vec=[meshnum,meshnum,meshnum]
  else:
   mesh_vec=None
  cell=pbcgto.M(atom=geofile,basis=basis_set,pseudo=ecp,a=[[lattice_v[0],0,0],[0,lattice_v[1],0],[0,0,lattice_v[2]]],mesh=mesh_vec,verbose=pyscf_settings[5],dimension=cell_dim,spin=pyscf_conv_settings[11])

  if kpts!=None:
   cell.make_kpts(kpts)
  pbc_elec=pbcdft.KRKS(cell).set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],exp_to_discard=0.1)
  #print(pbc_elec.kpts)

  if pyscf_settings[12]=="no_df" or pyscf_settings[12]==None:
   print("Not using density fitting.",file=outputfile)
  elif pyscf_settings[12]=="df_default":
   print("Using default density fitting auxiliary basis.",file=outputfile)
   pbc_elec.with_df.auxbasis="weigand"
   pbc_elec.with_df=pdf.GDF(cell)
  elif pyscf_settings[12]=="multigrid":
    print("Using multigrid density fitting.",file=outputfile)
    from pyscf.pbc.dft import multigrid
    pbc_elec.with_df = multigrid.MultiGridFFTDF(cell, kpts)

  else:
   print("Using given density fitting auxiliary basis.",file=outputfile)
   pbc_elec.with_df.auxbasis=pyscf_settings[8]
   pbc_elec.with_df=pdf.GDF(cell)

  pbc_elec.chkfile=pyscf_settings[9]+".chk"
  pbc_elec.output=pyscf_settings[9]+".log"
  pbc_elec.ecp=ecp
  pbc_elec.diis_start_cycle=pyscf_conv_settings[2]

  if pyscf_conv_settings[5]=="adiis":
   pbc_elec.DIIS=pbcscf.ADIIS
  elif pyscf_conv_settings[5]=="ediis":
   pbc_elec.DIIS=pbcscf.EDIIS
  elif pyscf_conv_settings[5]=="soscf":
   pbc_elec=pbc_elec.newton()
  if pyscf_conv_settings[12]!=None:
   pbc_elec=pbcscf.addons.smearing_(pbc_elec, sigma=pyscf_conv_settings[13], method=pyscf_conv_settings[12]).run()
  if pyscf_conv_settings[14]==True:
   pbc_elec= scf.addons.remove_linear_dep_(pbc_elec).run()

  pbc_elec.xc=dft_functional
  pbc_elec.incore_anyway=True
  pbc_elec.kernel()

 elif pyscf_settings[0]=="rhf":
  if kpts!=None:
   cell.make_kpts(kpts)
  pbc_elec=pbcscf.KRHF(cell).set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],exp_to_discard=0.1)

  if pyscf_settings[12]=="no_df" or pyscf_settings[12]==None:
    print("Not using density fitting.",file=outputfile)
  elif pyscf_settings[12]=="df_default":
   print("Using default density fitting auxiliary basis.",file=outputfile)
   pbc_elec.with_df.auxbasis="weigand"
   pbc_elec.with_df=pdf.GDF(cell)
  else:
    print("Using given density fitting auxiliary basis.",file=outputfile)
    pbc_elec.with_df.auxbasis=pyscf_settings[8]
    pbc_elec.with_df=pdf.GDF(cell)

  pbc_elec.diis_start_cycle=pyscf_conv_settings[2]
  if pyscf_conv_settings[5]=="adiis":
   pbc_elec.DIIS=pbcscf.ADIIS
  elif pyscf_conv_settings[5]=="ediis":
   pbc_elec.DIIS=pbcscf.EDIIS
  elif pyscf_conv_settings[5]=="soscf":
   pbc_elec=pbc_elec.newton()
  if pyscf_conv_settings[12]!=None:
   pbc_elec=pbcscf.addons.smearing_(pbc_elec, sigma=pyscf_conv_settings[13], method=pyscf_conv_settings[12]).run()

  
  pbc_elec.chkfile=pyscf_settings[9]+".chk"
  pbc_elec.output=pyscf_settings[9]+".log"
  pbc_elec.ecp=ecp
  pbc_elec.incore_anyway=True
  pbc_elec.kernel()
 else:
  print("Due to PySCF updates, only PBC-DFT is currently supported. Please use es_method=\"dft\".",file=outputfile)

 h_scf=pbc_elec.get_fock()
 h_scf=h_scf*27.2114
 s=pbc_elec.get_ovlp()
 norb=len(h_scf)
 numelec=int(np.sum(pbc_elec.mo_occ))

 print ('NORB:',norb,'NUMELEC:',numelec)
 print ('h_scf:',h_scf)
 return h_scf,s,norb,numelec

def esc_pyscf(geofile,dft_functional,basis_set,ecp,convtol,maxiter,pyscf_settings):
 #from pyscf import gto,dft,scf 

 geo=gto.M(atom=geofile,basis=basis_set,ecp=ecp,verbose=pyscf_settings[5])
 if pyscf_settings[0]=="dft":
  if pyscf_settings[3]=="rks":
   pyscf_elec=dft.RKS(geo).set(max_cycle=maxiter,conv_tol=convtol)
   pyscf_elec.xc=dft_functional
   pyscf_elec.kernel() 
   if pyscf_elec.converged==False:
    pyscf_elec=dft.RKS(geo).set(max_cycle=2*maxiter,conv_tol=convtol,level_shift=0.2)
    #scf.addons.dynamic_level_shift_(pyscf_elec,factor=0.5)
    pyscf_elec.damp=0.5
    pyscf_elec.diis_start_cycle=2
    pyscf_elec.kernel()  
   if pyscf_elec.converged==False:
    print("Your SCF calculation did not converge after 2 attempts. Check your settings.")
    exit()
   h = pyscf_elec.get_fock()
   h=h*27.2114
   s = pyscf_elec.get_ovlp()
   #norb=len(h)
   #numelec=int(np.sum(pyscf_elec.mo_occ))

 elif pyscf_settings[0]=="mcpdft":
  #Pyscf mcpdft routine modified from original version created by Dr. Andrew Sand (Butler University)
  [nActEl,nAct]=pyscf_settings[2]
  if pyscf_settings[3]=="rks":
   #SCF DFT run
   pyscf_elec = dft.RKS(geo).set(max_cycle=maxiter,conv_tol=convtol)
   pyscf_elec.xc = dft_functional
   pyscf_elec.kernel()
   if pyscf_elec.converged==False:
    pyscf_elec=dft.RKS(geo).set(max_cycle=2*maxiter,conv_tol=convtol,level_shift=0.2)
    pyscf_elec.damp=0.5
    pyscf_elec.diis_start_cycle=2
    pyscf_elec.kernel()
   if pyscf_elec.converged==False:
    print("Your SCF RKS calculation did not converge after 2 attempts. Check your settings.")
    exit()

  elif pyscf_settings[3]=="rhf":
   #If we would rather, we could run Hartree-Fock as the pre-MR step.
   pyscf_elec = scf.RHF(geo).set(max_cycle=maxiter,conv_tol=convtol)
   pyscf_elec.init_guess = 'huckel'
   #pyscf_elec.xc = dft_functional
   pyscf_elec.kernel()
   if pyscf_elec.converged==False:
    pyscf_elec=scf.RHF(geo).set(max_cycle=2*maxiter,conv_tol=convtol,level_shift=0.2)
    pyscf_elec.damp=0.5
    pyscf_elec.diis_start_cycle=2
    pyscf_elec.kernel()
   if pyscf_elec.converged==False:
    print("Your SCF RHF calculation did not converge after 2 attempts. Check your settings.")
    exit()

  else:
    print("SCF method not recognized. Use rks or rhf keywords.")
    exit()

  #Now, we perform the CASSCF or CASCI.  The PDFT functional is tPBE by default and is changed in pyscf_settings not using dft_functional.
  print("Using t "+pyscf_settings[4]+" for MCPDFT functional")
  if pyscf_settings[1]=="casscf":
   mc = mcpdft.CASSCF(pyscf_elec, 't'+pyscf_settings[4], nAct, nActEl).run()
  elif pyscf_settings[1]=="casci":
   mc = mcpdft.CASCI(pyscf_elec, 't'+pyscf_settings[4], nAct, nActEl).run()
  else:
   print("MCSCF method not recognized. Use casscf or casci keywords.")
   exit()

  #The remainder of this file builds the PDFT fock matrix in an ao basis
  #The next two lines build the wave function-like parts of the fock matrix.
  dm = mc.make_rdm1()
  F = mc.get_hcore () + mc._scf.get_j(mc,dm)
  #The next lines handle the potential contributions (PDFT part) to the fock matrix.
  mc_pot, mc_pot2 = mc.get_pdft_veff()
  h_new = F + mc_pot
  #Now get overlap matrix
  h=np.array(h_new)
  s = pyscf_elec.get_ovlp()

 norb=len(h)
 numelec=int(np.sum(pyscf_elec.mo_occ))
 #These routines print the data to a MolEl.dat file for use in reruns 
 nTotEl = geo.nelec[0]+ geo.nelec[1]
 if pyscf_settings[0]=="mcpdft" and pyscf_settings[1]=="casscf":
  nAO = mc.mo_coeff.shape[1]
  mo_coef = mc.mo_coeff
 else:
  nAO= pyscf_elec.mo_coeff.shape[1]
  mo_coef=pyscf_elec.mo_coeff
  if pyscf_settings[0]!="mcpdft":
   nAct=0
   nActEl=0

 print_molel(h,s,norb,numelec,nTotEl,nAct,nActEl,nAO,mo_coef)
 h=h*27.211396641308

 return h,s,norb,numelec

def print_molel(h,s,norb,numelec,nTotEl,nAct,nActEl,nAO,mo_coef): 
 f = open("MolEl.dat", "w")
 f.write("Number of states,orbitals,electrons,ActOrb,ActEl \n")
 f.write("1 " + str(mo_coef.shape[1]) + " " + str(nTotEl) + " " + str(nAct) + " " + str(nActEl) + '\n')


 f.write("Overlap Matrix (AO) \n")
 for x in range(nAO):
    for y in range(x + 1):
        f.write(f'{x + 1} {y + 1} {s[x][y]} \n')

 f.write("Molecular orbital coefficients \n")
 for x in range(nAO):
    for y in range(nAO):
        f.write(f'{x + 1} {y + 1} {mo_coef[x][y]} \n')
 f.write("Effective Hamiltonian (AO) \n")
 f.write("State 1 \n")
 for x in range(nAO):
    for y in range(nAO):
        f.write(f'{x + 1} {y + 1} {h[x][y]} \n')

 f.write("Orbital Energies")

def esc_pyscf_wbl(geofile,dft_functional,basis_set,ecp,convtol,maxiter,num_elec_atoms,pyscf_settings):
 #from pyscf import gto,dft,scf
 geo=gto.M(atom=geofile,basis=basis_set,ecp=ecp,verbose=pyscf_settings[5])

 if pyscf_settings[0]=="dft":
  if pyscf_settings[3]=="rks":
   pyscf_elec=dft.RKS(geo).set(max_cycle=maxiter,conv_tol=convtol)
   pyscf_elec.xc=dft_functional
   pyscf_elec.kernel()
   if pyscf_elec.converged==False:
    pyscf_elec=dft.RKS(geo).set(max_cycle=2*maxiter,conv_tol=convtol,level_shift=0.2)
    pyscf_elec.damp=0.5
    pyscf_elec.diis_start_cycle=2
    pyscf_elec.kernel()
   if pyscf_elec.converged==False:
    print("Your SCF calculation did not converge after 2 attempts. Check your settings.")
    exit()
   h = pyscf_elec.get_fock()
   s = pyscf_elec.get_ovlp()

 elif pyscf_settings[0]=="mcpdft":
  #Pyscf mcpdft routine modified from original version created by Dr. Andrew Sand (Butler University)
  [nActEl,nAct]=pyscf_settings[2]
  if pyscf_settings[3]=="rks":
   #SCF DFT run
   pyscf_elec = dft.RKS(geo).set(max_cycle=maxiter,conv_tol=convtol)
   pyscf_elec.xc = dft_functional
   pyscf_elec.kernel()
   if pyscf_elec.converged==False:
    pyscf_elec=dft.RKS(geo).set(max_cycle=2*maxiter,conv_tol=convtol,level_shift=0.2)
    pyscf_elec.damp=0.5
    pyscf_elec.diis_start_cycle=2
    pyscf_elec.kernel()
   if pyscf_elec.converged==False:
    print("Your SCF RKS calculation did not converge after 2 attempts. Check your settings.")
    exit()

  elif pyscf_settings[3]=="rhf":
   #If we would rather, we could run Hartree-Fock as the pre-MR step.
   pyscf_elec = scf.RHF(geo).set(max_cycle=maxiter,conv_tol=convtol)
   pyscf_elec.init_guess = 'huckel'
   #pyscf_elec.xc = dft_functional
   pyscf_elec.kernel()
   if pyscf_elec.converged==False:
    pyscf_elec=scf.RHF(geo).set(max_cycle=2*maxiter,conv_tol=convtol,level_shift=0.2)
    pyscf_elec.damp=0.5
    pyscf_elec.diis_start_cycle=2
    pyscf_elec.kernel()
   if pyscf_elec.converged==False:
    print("Your SCF RHF calculation did not converge after 2 attempts. Check your settings.")
    exit()

  else:
    print("SCF method not recognized. Use rks or rhf keywords.")
    exit()

  #Now, we perform the CASSCF or CASCI.  The PDFT functional is tPBE by default and is changed in pyscf_settings not using dft_functional.
  print("Using t "+pyscf_settings[4]+" for MCPDFT functional")
  if pyscf_settings[1]=="casscf":
   mc = mcpdft.CASSCF(pyscf_elec, 't'+pyscf_settings[4], nAct, nActEl).run()
  elif pyscf_settings[1]=="casci":
   mc = mcpdft.CASCI(pyscf_elec, 't'+pyscf_settings[4], nAct, nActEl).run()
  else:
   print("MCSCF method not recognized. Use casscf or casci keywords.")
   exit()

  #The remainder of this file builds the PDFT fock matrix in an ao basis
  #The next two lines build the wave function-like parts of the fock matrix.
  dm = mc.make_rdm1()
  F = mc.get_hcore () + mc._scf.get_j(mc,dm)
  #The next lines handle the potential contributions (PDFT part) to the fock matrix.
  mc_pot, mc_pot2 = mc.get_pdft_veff()
  h = F + mc_pot
  #Now get overlap matrix
  s = pyscf_elec.get_ovlp()

 #This part determines the total number of orbitals, number of electrons, and orbitals in the electrodes (assumes symmetric electrodes)
 norb=len(h)
 numelec=int(np.sum(pyscf_elec.mo_occ))
 ao_data=gto.mole.ao_labels(geo,fmt=False)
 
 atom_num=0
 ao_index=0
 elec_orb=0
 #print(ao_data)
 ao_data_len=len(ao_data)
 while atom_num < num_elec_atoms:
  atom_num=int(ao_data[ao_index][0])
  ao_index+=1
  if ao_index == ao_data_len:
   print("The # of electrode atoms is incorrect")
   break
 elec_orb=ao_index-1
 print("printing h")
 #print(h)
 #print("printing s")
 #rint(s)
 #This section writes the H and S matrices to a standard ".scf_dat" file for RUQT-Fortran to use
 scffile=open("fort_ruqt"+".scf_dat",'w')
 mo_dim=pyscf_elec.mo_coeff.shape
 f_dim=h.shape
 o_dim=s.shape

 scffile.write('Molecular Orbital Coefficients'+"\n")
 for x in range(0,mo_dim[0]):
  for y in range(0,mo_dim[1]):
   scffile.write("{0}".format(x+1)+" "+"{0}".format(y+1)+" "+"{0}".format(pyscf_elec.mo_coeff[x,y])+"\n")

 scffile.write('Molecular Orbital Energies'+"\n")
 for x in range(0,mo_dim[0]):
  scffile.write("{0}".format(pyscf_elec.mo_energy[x])+"\n")

 scffile.write('Overlap Matrix'+"\n")
 for x in range(0,o_dim[0]):
  for y in range(0,o_dim[1]):
   scffile.write("{0}".format(x+1)+" "+"{0}".format(y+1)+" "+"{0}".format(s[x,y])+"\n")

 scffile.write('Fock Matrix'+"\n")
 for x in range(0,f_dim[0]):
  for y in range(0,f_dim[1]):
   scffile.write("{0}".format(x+1)+" "+"{0}".format(y+1)+" "+"{0}".format(h[x,y])+"\n")

 scffile.close()
 h=h*27.2114
 
 return h,s,norb,numelec,elec_orb

def esc_pyscf2(geofile,dft_functional,basis_set,ecp,num_elec_atoms,pyscf_settings,pyscf_conv_settings):
 #from pyscf import gto,dft,scf

 outputfile=open(pyscf_settings[9]+".out",'w')
 geo=gto.M(atom=geofile,basis=basis_set,ecp=ecp,verbose=pyscf_settings[5],output=pyscf_settings[9]+".log",spin=pyscf_conv_settings[11],charge=pyscf_conv_settings[10])
 h_final=0
 s_final=0
 norb=0
 numelec=0

 if pyscf_settings[0]=="dft":
  if pyscf_settings[3]=="rks":
   if "diis" in pyscf_conv_settings[5]:
    if pyscf_settings[12]=="no_df" or pyscf_settings[12]==None:
      print("Not using density fitting.",file=outputfile)
      pyscf_elec=dft.RKS(geo).set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],level_shift=pyscf_conv_settings[4])
    elif pyscf_settings[12]=="df_default":
      print("Using default density fitting auxiliary basis.",file=outputfile)
      pyscf_elec=dft.RKS(geo).density_fit().set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],level_shift=pyscf_conv_settings[4])
    else:
      print("Using given density fitting auxiliary basis.",file=outputfile)
      pyscf_elec=dft.RKS(geo).density_fit().set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],level_shift=pyscf_conv_settings[4])
      pyscf_elec.with_df.auxbasis=pyscf_settings[8]
    pyscf_elec.xc=dft_functional
    pyscf_elec.init_guess = pyscf_conv_settings[6]
    pyscf_elec.chkfile=pyscf_settings[9]+".chk"
    pyscf_elec.output=pyscf_settings[9]+".log"
    pyscf_elec.damp=pyscf_conv_settings[3]
    pyscf_elec.diis_start_cycle=pyscf_conv_settings[2]
    if pyscf_conv_settings[5]=="adiis":
     pyscf_elec.DIIS=scf.ADIIS
    elif pyscf_conv_settings[5]=="ediis":
     pyscf_elec.DIIS=scf.EDIIS
    if pyscf_conv_settings[9]==True:
     pyscf_elec=scf.addons.frac_occ(pyscf_elec)
    pyscf_elec.kernel()

    if pyscf_elec.converged==False:
     print("Your SCF calculation did not converge using"+str(pyscf_conv_settings[5])+" optimizer. Check your settings.",file=outputfile)
     exit()
    h = pyscf_elec.get_fock()
#    h=h*27.2114
    s = pyscf_elec.get_ovlp()
    h_final,s_final,norb,numelec=prepare_outputs(h,s,pyscf_settings,pyscf_elec,"None",geo)


   elif "soscf" in pyscf_conv_settings[5]:
    #print("Using SOSCF")
    #for i in range(1,pyscf_conv_settings[5]):
    if pyscf_settings[12]=="no_df" or pyscf_settings[12]==None:
      print("Not using density fitting.",file=outputfile)
      pyscf_elec=dft.RKS(geo).set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],level_shift=pyscf_conv_settings[4])
    elif pyscf_settings[12]=="df_default":
      print("Using default density fitting auxiliary basis.",file=outputfile)
      pyscf_elec=dft.RKS(geo).density_fit().set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],level_shift=pyscf_conv_settings[4])
    else:
      print("Using given density fitting auxiliary basis.",file=outputfile)
      pyscf_elec=dft.RKS(geo).density_fit().set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],level_shift=pyscf_conv_settings[4])
      pyscf_elec.with_df.auxbasis=pyscf_settings[8]
     #scf.addons.dynamic_level_shift_(pyscf_elec,factor=0.5)
    pyscf_elec.xc=dft_functional
    pyscf_elec=pyscf_elec.newton()
    pyscf_elec.init_guess = pyscf_conv_settings[6]
    pyscf_elec.chkfile=pyscf_settings[9]+".chk"
    pyscf_elec.output=pyscf_settings[9]+".log"
    pyscf_elec.damp=pyscf_conv_settings[3]
    if pyscf_conv_settings[9]==True:
     pyscf_elec=scf.addons.frac_occ(pyscf_elec)
#    pyscf_elec.newton()
    pyscf_elec.kernel()

    if pyscf_elec.converged==False:
     print("Your SCF calculation did not converge using"+str(pyscf_conv_settings[5])+" optimizer. Check your settings.",file=outputfile)
     exit()
    h = pyscf_elec.get_fock()
#    h=h*27.2114
    s = pyscf_elec.get_ovlp()
    h_final,s_final,norb,numelec=prepare_outputs(h,s,pyscf_settings,pyscf_elec,"None",geo)
  else:
   print("DFT/SCF choice not supported",file=pyscf_settings[9]+".out")
   exit()
   #norb=len(h)
   #numelec=int(np.sum(pyscf_elec.mo_occ))
 elif pyscf_settings[0]=="mcpdft":
  #Pyscf mcpdft routine modified from original version created by Dr. Andrew Sand (Butler University)
  [nActEl,nAct]=pyscf_settings[2]
  if pyscf_settings[3]=="rks":
   if "diis" in pyscf_conv_settings[5]:
    if pyscf_settings[12]=="no_df" or pyscf_settings[12]==None:
      print("Not using density fitting.",file=outputfile)
      pyscf_elec=dft.RKS(geo).set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],level_shift=pyscf_conv_settings[4])
    elif pyscf_settings[12]=="df_default":
      print("Using default density fitting auxiliary basis.",file=outputfile)
      pyscf_elec=dft.RKS(geo).density_fit().set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],level_shift=pyscf_conv_settings[4])
    else:
      print("Using given density fitting auxiliary basis.",file=outputfile)
      pyscf_elec=dft.RKS(geo).density_fit().set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],level_shift=pyscf_conv_settings[4])
      pyscf_elec.with_df.auxbasis=pyscf_settings[8]
    pyscf_elec.xc=dft_functional
    pyscf_elec.init_guess=pyscf_conv_settings[6]
    pyscf_elec.chkfile=pyscf_settings[9]+".chk"
    pyscf_elec.output=pyscf_settings[9]+".log"
    pyscf_elec.damp=pyscf_conv_settings[3]
    pyscf_elec.diis_start_cycle=pyscf_conv_settings[2]
    if pyscf_conv_settings[5]=="adiis":
     pyscf_elec.DIIS=scf.ADIIS
    elif pyscf_conv_settings[5]=="ediis":
     pyscf_elec.DIIS=scf.EDIIS
    if pyscf_conv_settings[9]==True:
     pyscf_elec=scf.addons.frac_occ(pyscf_elec)
    pyscf_elec.kernel()

    if pyscf_elec.converged==False:
     print("Your SCF calculation did not converge using"+str(pyscf_conv_settings[5])+" optimizer. Check your settings.",file=outputfile)
     exit()
    h = pyscf_elec.get_fock()
#    h=h*27.2114
    s = pyscf_elec.get_ovlp()
    h_final,s_final,norb,numelec=prepare_outputs(h,s,pyscf_settings,pyscf_elec,"None",geo)


   elif "soscf" in pyscf_conv_settings[5]:
    #print("Using SOSCF")
    #for i in range(1,pyscf_conv_settings[5]):
    if pyscf_settings[12]=="no_df" or pyscf_settings[12]==None:
      print("Not using density fitting.",file=outputfile)
      pyscf_elec=dft.RKS(geo).set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],level_shift=pyscf_conv_settings[4])
    elif pyscf_settings[12]=="df_default":
      print("Using default density fitting auxiliary basis.",file=outputfile)
      pyscf_elec=dft.RKS(geo).density_fit().set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],level_shift=pyscf_conv_settings[4])
    else:
      print("Using given density fitting auxiliary basis.",file=outputfile)
      pyscf_elec=dft.RKS(geo).density_fit().set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],level_shift=pyscf_conv_settings[4])
      pyscf_elec.with_df.auxbasis=pyscf_settings[8]
     #scf.addons.dynamic_level_shift_(pyscf_elec,factor=0.5)
    pyscf_elec.xc=dft_functional
    pyscf_elec=pyscf_elec.newton()
    pyscf_elec.init_guess=pyscf_conv_settings[6]
    pyscf_elec.chkfile=pyscf_settings[9]+".chk"
    pyscf_elec.output=pyscf_settings[9]+".log"
    pyscf_elec.damp=pyscf_conv_settings[3]
    if pyscf_conv_settings[9]==True:
     pyscf_elec=scf.addons.frac_occ(pyscf_elec)
#    pyscf_elec.newton()
    pyscf_elec.kernel()

   if pyscf_elec.converged==False:
    print("Your SCF calculation did not converge using"+str(pyscf_conv_settings[5])+" optimizer. Check your settings.",file=outputfile)
    exit()
    h = pyscf_elec.get_fock()
#    h=h*27.2114
    s = pyscf_elec.get_ovlp()
    h_final,s_final,norb,numelec=prepare_outputs(h,s,pyscf_settings,pyscf_elec,"None",geo)


  elif pyscf_settings[3]=="rhf":
   if "diis" in pyscf_conv_settings[5]:
    if pyscf_settings[12]=="no_df" or pyscf_settings[12]==None:
      print("Not using density fitting.",file=outputfile)
      pyscf_elec=scf.RHF(geo).set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],level_shift=pyscf_conv_settings[4])
    elif pyscf_settings[12]=="df_default":
      print("Using default density fitting auxiliary basis.",file=outputfile)
      pyscf_elec=scf.RHF(geo).density_fit().set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],level_shift=pyscf_conv_settings[4])
    else:
      print("Using given density fitting auxiliary basis.",file=outputfile)
      pyscf_elec=scf.RHF(geo).density_fit().set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],level_shift=pyscf_conv_settings[4])
      pyscf_elec.with_df.auxbasis=pyscf_settings[8]
    pyscf_elec.init_guess = pyscf_conv_settings[6]
    pyscf_elec.chkfile=pyscf_settings[9]+".chk"
    pyscf_elec.output=pyscf_settings[9]+".log"
    pyscf_elec.damp=pyscf_conv_settings[3]
    pyscf_elec.diis_start_cycle=pyscf_conv_settings[2]
    if pyscf_conv_settings[5]=="adiis":
     pyscf_elec.DIIS=scf.ADIIS
    elif pyscf_conv_settings[5]=="ediis":
     pyscf_elec.DIIS=scf.EDIIS
    if pyscf_conv_settings[9]==True:
     pyscf_elec=scf.addons.frac_occ(pyscf_elec)
    pyscf_elec.kernel()

    if pyscf_elec.converged==False:
     print("Your SCF calculation did not converge using"+str(pyscf_conv_settings[5])+" optimizer. Check your settings.",file=outputfile)
     exit()
    h = pyscf_elec.get_fock()
#    h=h*27.2114
    s = pyscf_elec.get_ovlp()
    h_final,s_final,norb,numelec=prepare_outputs(h,s,pyscf_settings,pyscf_elec,"None",geo)
   elif "soscf" in pyscf_conv_settings[5]:
    #print("Using SOSCF")
    #for i in range(1,pyscf_conv_settings[5]):
    if pyscf_settings[12]=="no_df" or pyscf_settings[12]==None:
      print("Not using density fitting.",file=outputfile)
      pyscf_elec=scf.RHF(geo).set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],level_shift=pyscf_conv_settings[4])
    elif pyscf_settings[12]=="df_default":
      print("Using default density fitting auxiliary basis.",file=outputfile)
      pyscf_elec=scf.RHF(geo).density_fit().set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],level_shift=pyscf_conv_settings[4])
    else:
      print("Using given density fitting auxiliary basis.",file=outputfile)
      pyscf_elec=scf.RHF(geo).density_fit().set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],level_shift=pyscf_conv_settings[4])
      pyscf_elec.with_df.auxbasis=pyscf_settings[8]
     #scf.addons.dynamic_level_shift_(pyscf_elec,factor=0.5)
    pyscf_elec.xc=dft_functional
    pyscf_elec=pyscf_elec.newton()
    pyscf_elec.init_guess=pyscf_conv_settings[6]
    pyscf_elec.chkfile=pyscf_settings[9]+".chk"
    pyscf_elec.output=pyscf_settings[9]+".log"
    pyscf_elec.damp=pyscf_conv_settings[3]
    if pyscf_conv_settings[9]==True:
     pyscf_elec=scf.addons.frac_occ(pyscf_elec)
#    pyscf_elec.newton()
    pyscf_elec.kernel()

    if pyscf_elec.converged==False:
     print("Your SCF calculation did not converge using"+str(pyscf_conv_settings[5])+" optimizer. Check your settings.",file=outputfile)
     exit()
    h = pyscf_elec.get_fock()
#    h=h*27.2114
    s = pyscf_elec.get_ovlp()
    h_final,s_final,norb,numelec=prepare_outputs(h,s,pyscf_settings,pyscf_elec,"None",geo)


  elif pyscf_settings[3]=="chkfile":
   if pyscf_conv_settings[5].lower()=="diis" or pyscf_conv_settings[5]==None:
    if pyscf_settings[12]=="no_df" or pyscf_settings[12]==None:
      print("Not using density fitting.",file=outputfile)
      pyscf_elec=dft.RKS(geo).set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],level_shift=pyscf_conv_settings[4])
    elif pyscf_settings[12]=="df_default":
      print("Using default density fitting auxiliary basis.",file=outputfile)
      pyscf_elec=dft.RKS(geo).density_fit().set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],level_shift=pyscf_conv_settings[4])
    else:
      print("Using given density fitting auxiliary basis.",file=outputfile)
      pyscf_elec=dft.RKS(geo).density_fit().set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],level_shift=pyscf_conv_settings[4])
      pyscf_elec.with_df.auxbasis=pyscf_settings[8]
    pyscf_elec=dft.RKS(geo).density_fit().set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],level_shift=pyscf_conv_settings[4])
    pyscf_elec.xc=dft_functional
    pyscf_elec.diis_start_cycle=pyscf_conv_settings[2]

   elif pyscf_conv_settings[5].lower=="soscf":
    if pyscf_settings[12]=="no_df"  or pyscf_settings[12]==None:
      print("Not using density fitting.",file=outputfile)
      pyscf_elec=dft.RKS(geo).set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],level_shift=pyscf_conv_settings[4])
    elif pyscf_settings[12]=="df_default":
      print("Using default density fitting auxiliary basis.",file=outputfile)
      pyscf_elec=dft.RKS(geo).density_fit().set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],level_shift=pyscf_conv_settings[4])
    else:
      print("Using given density fitting auxiliary basis.",file=outputfile)
      pyscf_elec=dft.RKS(geo).density_fit().set(max_cycle=pyscf_conv_settings[0],conv_tol=pyscf_conv_settings[1],level_shift=pyscf_conv_settings[4])
      pyscf_elec.with_df.auxbasis=pyscf_settings[8]
    pyscf_elec=pyscf_elec.newton()
    pyscf_elec.xc=dft_functional
   
   pyscf_elec.init_guess=pyscf_conv_settings[6]
   pyscf_elec.chkfile=pyscf_settings[9]+".chk"
   pyscf_elec.damp=pyscf_conv_settings[3]
   pyscf_elec.kernel()

  else:
   print("SCF method not recognized. Use rks or rhf keywords.",file=outputfile)
   exit()

  if pyscf_conv_settings[7]==True:
   mo_old=read_molel_orbs(shape(pyscf_elec.mo_coeff[1]),pyscf_conv_setting[8])
   pyscf_elec.mo_coeff=mo_old

  #Now, we perform the CASSCF or CASCI.  The PDFT functional is tPBE by default and is changed in pyscf_settings not using dft_functional.
  #print("Using t"+pyscf_settings[4]+" for MCPDFT functional")
  if pyscf_settings[1]=="casscf":
   if pyscf_settings[6] != [] and pyscf_settings[7]==False:
    mc2 = mcscf.CASSCF(pyscf_elec, nAct, nActEl)
    mo=mc2.sort_mo(pyscf_settings[6])
    mc2.kernel(mo)
    mc = mcpdft.CASSCF(pyscf_elec, 't'+pyscf_settings[4], nAct, nActEl)
    if pyscf_settings[11] != 1:
     mc.fcisolver.nroots = pyscf_settings[11]
    mc.kernel(mc2.mo_coeff)

   elif pyscf_settings[7]==True and pyscf_settings[6] != []:
    nAct, nActEl, orbs = avas.avas(pyscf_elec,pyscf_settings[6])
    mc2 = mcscf.CASSCF(pyscf_elec, nAct, nActEl)
    mc2.kernel(orbs)
    mc = mcpdft.CASSCF(pyscf_elec, 't'+pyscf_settings[4], nAct, nActEl)
    if pyscf_settings[11] != 1:
     mc.fcisolver.nroots = pyscf_settings[11]
    mc.kernel(mc2.mo_coeff)

   else:
    mc = mcpdft.CASSCF(pyscf_elec, 't'+pyscf_settings[4], nAct, nActEl)
    if pyscf_settings[11] != 1:
     mc.fcisolver.nroots = pyscf_settings[11]
    mc.kernel()

  elif pyscf_settings[1]=="casci":
   if pyscf_settings[6] != [] and pyscf_settings[7]==False:
    mc2 = mcscf.CASCI(pyscf_elec, nAct, nActEl)
    mo=mc2.sort_mo(pyscf_settings[6])
    mc2.kernel(mo)
    mc = mcpdft.CASCI(pyscf_elec, 't'+pyscf_settings[4], nAct, nActEl)
    if pyscf_settings[11] != 1:
     mc.fcisolver.nroots = pyscf_settings[11]
    mc.kernel(mc2.mo_coeff)

   if pyscf_settings[7]==True and pyscf_settings[6] != []:
    nAct, nActEl, orbs = avas.avas(pyscf_elec,pyscf_settings[6])
    mc2 = mcscf.CASCI(pyscf_elec, nAct, nActEl)
    mc2.kernel(orbs)
    mc = mcpdft.CASCI(pyscf_elec, 't'+pyscf_settings[4], nAct, nActEl)
    if pyscf_settings[11] != 1:
     mc.fcisolver.nroots = pyscf_settings[11]
    mc.kernel(mc2.mo_coeff)

   else:
    mc = mcpdft.CASCI(pyscf_elec, 't'+pyscf_settings[4], nAct, nActEl)
    if pyscf_settings[11] != 1:
     mc.fcisolver.nroots = pyscf_settings[11]
    mc.kernel()

  else:
   print("MCSCF method not recognized. Use casscf or casci keywords.",file=outputfile)
   exit()
  mc.analyze()
  
  #The next part builds the PDFT fock matrix in an ao basis
  #The next two lines build the wave function-like parts of the fock matrix.
  dm = mc.make_rdm1()
  F = mc.get_hcore () + mc._scf.get_j(mc,dm)
  #The next lines handle the potential contributions (PDFT part) to the fock matrix.
  mc_pot, mc_pot2 = mc.get_pdft_veff()
  h_new = F + mc_pot
  #Now get overlap matrix
  h=np.array(h_new)
  s = pyscf_elec.get_ovlp()

  h_final,s_final,norb,numelec=prepare_outputs(h,s,pyscf_settings,pyscf_elec,mc,geo)

# print scf and mcscf(if casscf is used) orbitals to molden files for visualization
 if pyscf_settings[8]==True:
  if pyscf_settings[0]=="mcpdft" and pyscf_settings[1]=="casscf":
    tools.molden.from_scf(pyscf_elec,pyscf_settings[9]+"_scf.molden")
    tools.molden.from_mcscf(mc, pyscf_settings[9]+"_mc.molden", ignore_h=True, cas_natorb=False)
   # tools.cubegen.orbital(geo,"mc_orbitals.cube",mc.mo_coeff,resolution=0.02,margin=6.0)
  else:
    tools.molden.from_scf(pyscf_elec,pyscf_settings[9]+"_scf.molden")

 ao_data=gto.mole.ao_labels(geo,fmt=False)

 atom_num=0
 ao_index=0
 elec_orb=0
 ao_data_len=len(ao_data)
 while atom_num < num_elec_atoms: 
  atom_num=ao_data[ao_index][0]
  ao_index+=1
  if ao_index == ao_data_len:
   print("The # of electrode atoms is incorrect",file=outputfile)
   break
 elec_orb=ao_index-1

 return h_final,s_final,norb,numelec,elec_orb

#The next 2 routines print pyscf data to a MolEl.dat file for use in reruns
def prepare_outputs(h,s,pyscf_settings,pyscf_elec,mc,geo):
 norb=len(h)
 numelec=int(np.sum(pyscf_elec.mo_occ))
 nTotEl = geo.nelec[0]+ geo.nelec[1]
 if mc != "None":
  nAO = mc.mo_coeff.shape[1]
  mo_coef = mc.mo_coeff
  nAct,nActEl= mc.ncas, mc.nelecas
 else:
  nAO= pyscf_elec.mo_coeff.shape[1]
  mo_coef=pyscf_elec.mo_coeff
  nAct=0
  nActEl=0
 print_molel(h,s,norb,numelec,nTotEl,nAct,nActEl,nAO,mo_coef)
 h=h*27.211396641308
 return h,s,norb,numelec

def print_molel(h,s,norb,numelec,nTotEl,nAct,nActEl,nAO,mo_coef):

 f = open("MolEl.dat", "w")
 f.write("Number of states,orbitals,electrons,ActOrb,ActEl \n")
 f.write("1 " + str(mo_coef.shape[1]) + " " + str(nTotEl) + " " + str(nAct) + " " + str(nActEl) + '\n')


 f.write("Overlap Matrix (AO) \n")
 for x in range(nAO):
    for y in range(x + 1):
        f.write(f'{x + 1} {y + 1} {s[x][y]} \n')

 f.write("Molecular orbital coefficients \n")
 for x in range(nAO):
    for y in range(nAO):
        f.write(f'{x + 1} {y + 1} {mo_coef[x][y]} \n')

 f.write("Effective Hamiltonian(s) (AO) \n")
 f.write("State 1 \n")
 for x in range(nAO):
    for y in range(nAO):
        f.write(f'{x + 1} {y + 1} {h[x][y]} \n')

 f.write("Orbital Energies")

#This routine is used by esc_pyscf2 to get orbital numbers from an old MolEl.dat
def read_molel_orbs(nAO,molel_read_dir):
 f = open(molel_read_dir+"/MolEl.dat", "r")

 content=f.readlines()
 mo_coeff=np.zeros((nAO,nAO),dtype=float)
 for row in f:
  word="Molecular orbital coefficients"
  if row.find(word) !=-1:
   mo_line=lines.index(row)+1
 for x in range(nAO):
  for y in range(nAO):
   line_data=line.split(content[mo_line])
   mo_coeff[int(line_data[0])-1][int(line_data[1])-1]=float(line_data[2])
   mo_line+=1

 return mo_coeff
                   
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

#These routines extract detailed paritioning data from RUQT-Fortan for debugging purposes (currently unused).
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
def fort_inputwrite(cal_typ,FermiE,Fermi_Den,temp,max_bias,min_bias,delta_bias,min_trans_energy,max_trans_energy,delta_energy,qc_method,rdm_type,exmol_dir,exmol_file,exmol_prog,num_elec_atoms,outputfile,state_num,norb,numelec,size_elec,molcas_supercell):
 import string
 import math

#This part converts the python variables to ones recognized by the Fortran code
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
  if molcas_supercell==True:
   cp_fock='cp '+exmol_dir+"/MolEl.dat"+' MolEl.dat'

   cpdata_1=subprocess.Popen(cp_fock,shell=True)
   cpdata_1.wait()
   #h,s,norb,numelec,actorb,actelec,states=esc_molcas2(exmol_dir,"MolEl.dat",state_num,outputfile) 
   size_ex,size_elec=ruqt.read_syminfo(exmol_dir,norb,num_elec_atoms,outputfile)  
  else:
   size_ex=norb-2*size_elec
  numocc=int(numelec/2)
  numvirt=norb-numocc
 elif exmol_prog=="maple":
  norb,numocc,numvirt,size_ex,size_elec=orb_read_scfdat(exmol_dir,exmol_file,num_elec,outputfile)
 elif exmol_prog=="pyscf":
  numocc=math.ceil(numelec/2)
  numvirt=norb-numocc
  size_ex=norb-2*size_elec
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
 negf_inp.write("{0}".format("molcas") + "\n")
 negf_inp.write("{0}".format(rdm_doubles) + "\n")
 negf_inp.write("{0}".format("dft") + "\n")
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
 #Make sure to have a working RUQT.x executable compiled from the Github source code (ruqt.engine branch)

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
