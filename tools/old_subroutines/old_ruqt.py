
import numpy as np
import scipy
from pyscf import gto,dft,scf,mcscf,mcpdft,lo,tools
from pyscf.mcscf import avas
from ase import transport,Atoms,units
import matplotlib.pyplot as plt
import string,subprocess
from functools import reduce

#This is a storage file for older subroutines that were used in previous versions of pyRUQT.
#These routines are kept here for reference, but have been replaced by newer subroutines and 
# are no longer called by the main pyRUQT code.

def esc_molcas(calc_file,calc_dir,data_dir,state_num,outputfile):
 #import numpy as np
 norb=basisfxn_read(calc_dir,calc_file,outputfile)
 h=np.zeros((norb,norb))
 s=np.zeros((norb,norb))
 molcas_matread(calc_dir,data_dir+"FOCK_AO_"+str(state_num),norb,h)
 h=h*27.2114
 molcas_matread_sym(calc_dir,data_dir+"Overlap",norb,s)
 return h,s

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