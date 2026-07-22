     MODULE RUQT_CALC
      IMPLICIT NONE

      PRIVATE
      PUBLIC :: WBL_NEGF_CALC 

      CONTAINS

      subroutine WBL_NEGF_CALC(inputfile,norb,numfcore,numfvirt,numocc,numvirt,size_l,size_r,size_c,energy_start,energy_end,delta_en,volt_start,volt_end,delta_volt,inputcode,KT,ElectrodeType,Fermi_enl,Fermi_enR,CalcType,localden_fermi_l,localden_fermi_r,doubles,numatomic,functional,num_threads,use_b0,b0_type,write_ruqt_data,state_num,eff_ham_mat,overlap_matrix,energy_val,current_values, trans_results,current_results)
      !Use InterfaceMod
      !Use FunctionMod
      Use TypeMod
      Use RUQT_IntRoutines
      implicit none
      
      ! f2py annotations
      !f2py intent(in) :: inputfile,norb,numfcore,numfvirt,numocc,numvirt,size_l,size_r,size_c,energy_start,energy_end,delta_en,volt_start,volt_end,delta_volt,inputcode,KT,ElectrodeType,Fermi_enl,Fermi_enR,CalcType,localden_fermi_l,localden_fermi_r,doubles,numatomic,functional,num_threads,use_b0,b0_type,write_ruqt_data,state_num,eff_ham_mat, overlap_matrix
      !f2py intent(out) :: trans_results,current_results
               
      real(8),allocatable,dimension(:,:) :: H_one,H_Two,OneInts,H_Two_le,H_Two_re,H_Two_cen,H_Two_le_trans,H_Two_re_trans
      real(8), INTENT(IN) :: eff_ham_mat(:,:), overlap_matrix(:,:)
      real(8), allocatable, dimension(:) :: TwoIntsCompact,transm,transm_curr,energy_list,mo_ener,B0_coeff
      real(8), allocatable, dimension(:,:) :: Smat_le,Smat_re,Smat_cen,Smat,coupling_mat,mo_coeff,mo_coeff2
      real(8), INTENT(OUT) :: trans_results(1:energy_val),current_results(1:current_values)
      complex(8), allocatable, dimension(:,:) :: gfc_r,gfc_a,current_temp,Sigma_l,Sigma_r,Gamma_L,Gamma_R
      complex(8),allocatable,dimension(:) :: voltage,current
      real(8) :: coupling_r,coupling_l,energy,energy_start,energy_end,delta_en,volt_start,volt_end,delta_volt,corr_ener
      integer :: size_l,size_r,size_c,size_lc,size_lcr,numatomic,num_threads,numfcore,numfvirt
      character(len=100) :: inputfile,outfile,option
      logical :: libint,gamess,rdm_flag,invert,doubles,currentflag,cisd_flag,qchem,hf_flag,pyruqt
      logical :: dft_flag,pyscf,maple,use_b0,molcas,write_ruqt_data
      integer :: i,j,k,counter,counter2,current_values,norb,numact,energy_val,ioerror,numocc,numvirt
      character(len=40) :: ElectrodeType,CalcType,functional,inputcode,b0_type
      real(8) :: KT,current_con,Fermi_enl,Fermi_enR,localden_fermi,localden_fermi_l,localden_fermi_r,temp
      real(8) :: fermi_l,fermi_r
      complex(8), allocatable, dimension(:,:) :: test
      type(B1) :: B1data,l1data
      type(B2) :: B2data,l2data
      type(energr) :: G_S
      real(8) :: time_start,time_end
      integer :: state_num

      call cpu_time(time_start)
      invert=.true.
      currentflag=.false.     
     outfile=trim(inputfile) // ".negf_dat"
     open(unit=9,file=outfile,action='write',iostat=ioerror)
     write(9,*) "Just getting started"

     write(9,*) "Using the following parameters for the transport calculation"
     write(9,*) "Data file: ",outfile
     write(9,*) "Number of OpenMP Threads:",num_threads
     write(9,*) "Number of  Molecular and Atomic Orbitals:",norb,numatomic
     write(9,*) "Number of Active Orbitals:",norb-numfcore-numfvirt
     write(9,*) "Number of Occupied Orbitals:",numocc
     write(9,*) "Number of Virtual Orbitals:",numvirt
     write(9,*) "Orbitals in left electrode:",size_l
     write(9,*) "Orbitals in device region:",size_c
     write(9,*) "Orbitals in right electrode:",size_r
     write(9,*) "Transmission Energy Window(eV) and dE:",energy_start,energy_end,delta_en
     write(9,*) "Voltage Window and dV:",volt_start,volt_end,delta_volt
     write(9,*) "Fermi Density:",localden_fermi_l,localden_fermi_r
     write(9,*) "KT:",KT
     write(9,*) "Transport Calculated for State ",state_num
     write(9,*) "Print RUQT Data",write_ruqt_data
      size_lc = size_l + size_c
      size_lcr = size_l + size_c + size_r
      libint=.false.
      Call Flag_set(inputcode,functional,cisd_flag,rdm_flag,hf_flag,dft_flag,qchem,gamess,pyscf,maple,molcas,pyruqt)

      if(qchem.eqv..true.) then
        write(9,*) 'Using Qchem data for this run'
        Call Get_HF_Qchem(inputfile,norb,H_Two,Smat)
       elseif(molcas.eqv..true.) then
        Call Get_HF_Molcas2("MolEl.dat",norb,H_Two,Smat,state_num)
       !elseif(libint.eqv..true.) then
       ! Call Get_HF_libint(inputfile,norb,numact,H_one,Smat,mo_coeff,OneInts,TwoIntsCompact)
       elseif(gamess.eqv..true.) then
        Call Get_HF_GAMESS(inputfile,numatomic,H_Two,Smat,norb)
        write(9,*) 'Using GAMESS data for this run'
       elseif(pyscf.eqv..true.) then
        write(9,*) 'Using PySCF data for this run'
        Call Get_HF_PySCF(inputfile,numatomic,H_Two,Smat,norb)
       elseif(pyruqt.eqv..true.) then
        write(9,*) 'Using pyRUQT data for this run'
        allocate(H_two(1:numatomic,1:numatomic))
        allocate(Smat(1:numatomic,1:numatomic))
        H_two=eff_ham_mat
        Smat=overlap_matrix
       elseif(maple.eqv..true.) then
        write(9,*) "Using Maple+QuantumChemistry data for this run"
        Call Get_HF_PySCF(inputfile,numatomic,H_Two,Smat,norb)
       end if

      if(rdm_flag.eqv..true.) then
        write(9,*) 'This run using:'
        write(9,*) 'The Lehmann representation of a'

         if(doubles.eqv..true.) then
            write(9,*) 'p2-RDM Greens function'
           elseif(doubles.eqv..false.) then
            write(9,*) 'HF Greens function'
          end if

        elseif(qchem.eqv..true.) then
           write(9,*) 'QCHEM HF/DFT Greens function'

        elseif(molcas.eqv..true.) then
           write(9,*) 'Using Molcas Fock Matrix: FOCK_AO'
      
        elseif(pyscf.eqv..true.) then
           write(9,*) 'PYSCF HF/DFT Greens Function'

       end if

!Here we want to parition the Smatrix and H matrix into electrodes and
!the device
      if(trim(ElectrodeType).eq."Metal_WBL") then
       write(9,*) 'Starting Metal WBL calculation'
       allocate(Sigma_l(1:size_c,1:size_c))
       allocate(Sigma_r(1:size_c,1:size_c))
        
       Call PartitionHS_MetalWBL(Smat,H_Two,size_l,size_r,size_c,size_lc,size_lcr,Smat_le,Smat_re,Smat_cen,H_Two_le,H_Two_re,H_Two_cen,H_Two_le_trans,H_Two_re_trans,write_ruqt_data,inputfile,inputcode)

       write(9,*) 'Getting Electrodes'
       Call Electrodes_MetalWBL(Sigma_l,Sigma_r,Smat_re,Smat_le,H_Two_le,H_Two_re,localden_fermi_l,localden_fermi_r,size_c,size_lc,size_lcr,size_l,size_r,H_Two_le_trans,H_Two_re_trans,write_ruqt_data,inputfile)

       allocate(Gamma_L(1:size_c,1:size_c))
       allocate(Gamma_R(1:size_c,1:size_c))
       Gamma_L=-(DIMAG(Sigma_L)-DIMAG(adjoint(Sigma_L,size_c)))!,1E-12,8)
       Gamma_R=-(DIMAG(Sigma_R)-DIMAG(adjoint(Sigma_R,size_c)))!,1E-12,8)



      if(trim(CalcType).eq."current".or.trim(CalcType).eq."Current") then
       write(9,*) "Starting Current Calculation"
       allocate(gfc_r(1:size_c,1:size_c))
       allocate(gfc_a(1:size_c,1:size_c))
       allocate(current(1:current_values))
       allocate(voltage(1:current_values))
       allocate(transm(1:energy_val))
       allocate(transm_curr(1:energy_val))
       allocate(current_temp(1:size_c,1:size_c))
       allocate(energy_list(1:energy_val))
       transm=0
       transm_curr=0
       current_temp=0

     
       energy = energy_start
       current_con = 2*1.6021766E-19*(4.135667E-15)**(-1)
       counter = 1
      do k=1,energy_val
          transm(k) = 0
          current_temp=0
          energy = energy_start + (k-1)*delta_en
          energy_list(k) = energy
         if((hf_flag.eqv..true.).or.(dft_flag.eqv..true.)) then
          gfc_r = 0
          gfc_a = 0
          gfc_r = energy*Smat_cen-H_Two_cen
          gfc_r = gfc_r - Sigma_l - Sigma_r
          gfc_r = inv(gfc_r)
          gfc_a = adjoint(gfc_r,size_c)
         else if((cisd_flag.eqv..true.).or.(rdm_flag.eqv..true.)) then
            gfc_r = 0
            gfc_a = 0
            Call Build_G_SD_Invert(gfc_r,Sigma_l,Sigma_r,energy,size_l,size_c,size_lc,size_lcr,norb,inputfile,numocc,numvirt,counter,B1data,B2data,mo_ener,mo_coeff,mo_coeff2,doubles,currentflag,energy_val,k,G_S,corr_ener,numatomic,B0_coeff,use_b0,gamess,maple,numfcore,numfvirt,b0_type)
            gfc_a = adjoint(gfc_r,size_c)
            counter=2
         end if
         current_temp = matmul_zgemm(Gamma_R,gfc_a)
         current_temp = matmul_zgemm(gfc_r,current_temp)
         current_temp = matmul_zgemm(Gamma_L,current_temp)
         do i=1,size_c
           transm(k) = transm(k) + real(current_temp(i,i))
         end do
      end do
    
      do j=1,energy_val
         write(9,*) 'Transm vs Energy curve',real(energy_list(j)),real(transm(j))
       end do

       do j=1,energy_val
         write(9,*) real(energy_list(j)),real(transm(j))
         trans_results(j)=real(transm(j))
       end do

     currentflag=.true.

    if (pyruqt.eqv..true.) then
     fermi_enL=0.0
     fermi_enR=0.0
    end if

    do j=1,current_values
       temp = (j-1)*delta_volt + volt_start
       voltage(j) = temp
       current(j) = 0
       transm_curr=0
      do k=1,energy_val
          energy = energy_start + (k-1)*delta_en
          fermi_l=fermi_function(energy-temp*0.5,fermi_enL,KT)
          fermi_r=fermi_function(energy+temp*0.5,fermi_enR,KT)
          transm_curr(k) = transm(k)*(fermi_l-fermi_r)
         end do
        do k=1,energy_val
         current(j) = current(j) + delta_en*current_con*transm_curr(k)
        end do
        !write(9,*) 'Done with current at voltage:',real(voltage(j))
       end do
       
       do j=1,current_values
         write(9,*) 'IV curve',real(voltage(j)),real(current(j))
       end do

       do j=1,current_values
         !write(9,*) real(voltage(j)),real(current(j))
         current_results(j)=real(current(j))
       end do

      elseif(trim(CalcType).eq."Transmission".or.trim(CalcType).eq."transmission") then

       write(*,*) "Starting Transmission Calculation at Energy"
       allocate(gfc_r(1:size_c,1:size_c))
       allocate(gfc_a(1:size_c,1:size_c))
       allocate(transm(1:energy_val))
       allocate(energy_list(1:energy_val))
       allocate(current_temp(1:size_c,1:size_c))


       energy = energy_start
       counter=1
      do k=1,energy_val
          transm(k) = 0
          current_temp = 0
          energy = energy_start + (k-1)*delta_en
          energy_list(k) = energy

         if((hf_flag.eqv..true.).or.(dft_flag.eqv..true.)) then
            gfc_r = 0
            gfc_a = 0
            gfc_r = energy*Smat_cen-H_Two_cen
            gfc_r = gfc_r - Sigma_l - Sigma_r
            gfc_r = inv(gfc_r)
            gfc_a = adjoint(gfc_r,size_c)

         else if((cisd_flag.eqv..true.).or.(rdm_flag.eqv..true.)) then
            gfc_r = 0
            gfc_a = 0
            Call Build_G_SD_Invert(gfc_r,Sigma_l,Sigma_r,energy,size_l,size_c,size_lc,size_lcr,norb,inputfile,numocc,numvirt,counter,B1data,B2data,mo_ener,mo_coeff,mo_coeff2,doubles,currentflag,energy_val,k,G_S,corr_ener,numatomic,B0_coeff,use_b0,gamess,maple,numfcore,numfvirt,b0_type)
            gfc_a = adjoint(gfc_r,size_c)
            counter=2


         end if

         current_temp = matmul_zgemm(Gamma_L,gfc_r)
         current_temp = matmul_zgemm(current_temp,Gamma_R)
         current_temp = matmul_zgemm(current_temp,gfc_a)
         do i=1,size_c
            transm(k) = transm(k) + real(current_temp(i,i))
          end do
         ! write(9,*) 'Done with transmission function at energy:',transm(k)
        end do

       do j=1,energy_val
         write(9,*) 'Transm vs Energy curve',real(energy_list(j)),real(transm(j))
       end do
 
       write(9,*) energy_val
       do j=1,energy_val
         !write(9,*) real(energy_list(j)),real(transm(j))
         trans_results(j)=real(transm(j))
       end do

      end if

      else if(trim(ElectrodeType).eq."Molecule_WBL") then
       write(9,*) 'Using Molecule WBL Electrodes'
       write(9,*) "***This option was outdated/buggy and has been removed***"
       write(9,*) "***Please use standalone versions of RUQT to access this feature (not recommended)***"
     end if

      call cpu_time(time_end)
      write(9,*) 'CPU Time:',time_end-time_start
      close(9)
      !contains
      end subroutine
      end module
