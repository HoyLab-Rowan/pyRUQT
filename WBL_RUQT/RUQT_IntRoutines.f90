     MODULE RUQT_IntRoutines
      IMPLICIT NONE

      !PRIVATE
      PUBLIC 

      CONTAINS

      function adjoint(A,norb)
      implicit none
 
      complex(8),allocatable,dimension(:,:) :: A,adjoint
      integer :: i,j,norb

       allocate(adjoint(1:norb,1:norb))

       do i=1,norb
        do j=1,norb
         adjoint(j,i) =  DCONJG(A(i,j))
        end do
       end do
     
      end function

      subroutine Get_HF_QChem(inputfile,norb,H_two,Smat)
      !Use InterfaceMod 
      implicit none

      character(len=100) :: inputfile,datafile
      real(8),allocatable,dimension(:,:) :: H_Two,Smat
      integer :: norb,ioerror,i,j

      20 format(A)
      !write(*,*) inputfile
      allocate(H_two(1:norb,1:norb))
      allocate(Smat(1:norb,1:norb))
      H_two=0
      Smat=0

      datafile = trim(inputfile) // "_Smat"
      open(unit=2,file=datafile,action='READ', iostat = ioerror)

      do i=1,norb
       do j=1,norb
        read(2,*) Smat(i,j)
       end do
      end do
      close(2)

       write(*,*) 'Start Htwo'
      datafile = trim(inputfile) // "_Htwo"
      open(unit=3,file=datafile,action='READ', iostat = ioerror)

      do i=1,norb
       do j=1,norb
        read(3,*) H_Two(i,j)
       end do
      end do
      close(3)
      write(*,*) 'Done Getting HF Values'
      end subroutine

      subroutine Get_HF_Molcas(inputfile,norb,H_two,Smat,state_num)
      !Use InterfaceMod
      implicit none

      character(len=100) :: inputfile,datafile
      real(8),allocatable,dimension(:,:) :: H_Two,Smat
      integer :: norb,ioerror,i,j,x,y
      real(8) :: readtemp
      integer :: state_num
      character(len=100) :: state_char
      20 format(A)
      !write(*,*) inputfile
      allocate(H_two(1:norb,1:norb))
      allocate(Smat(1:norb,1:norb))
      H_two=0
      Smat=0

      datafile = "Overlap"
      open(unit=2,file=datafile,action='READ', iostat = ioerror)

      do i=1,norb
       do j=i,norb
        read(2,*) x,y,readtemp
         Smat(x,y)=readtemp
         Smat(y,x)=readtemp
       end do
      end do
      close(2)

       write(*,*) 'Start Fock Matrix'
      
      write(*,*) state_num
      if(state_num.lt.10) then
       write(state_char,"(I1)") state_num
      else
       write(state_char,"(I2)") state_num
      end if

      datafile = "FOCK_AO_"//trim(state_char)
      write(*,*) datafile
      open(unit=3,file=datafile,action='READ', iostat = ioerror)

      do i=1,norb
       do j=1,norb
        read(3,*) x,y,readtemp
        H_Two(x,y)=readtemp
       end do
      end do
      close(3)
      write(*,*) 'Done Getting HF Values'
      end subroutine

      subroutine Get_HF_Molcas2(datafile,norb,H_two,Smat,state_num)
      !Use InterfaceMod
      implicit none

      character(len=9) :: datafile
      real(8),allocatable,dimension(:,:) :: H_Two,Smat
      integer :: norb,ioerror,i,j,x,y
      real(8) :: readtemp
      integer :: state_num,num_states,norb_2,nelec_2,actorb,actel
      character(len=100) :: state_char,readtemp_str
      20 format(A)
      !write(*,*) inputfile
      allocate(H_two(1:norb,1:norb))
      allocate(Smat(1:norb,1:norb))
      H_two=0
      Smat=0

      open(unit=2,file=datafile,action='READ', iostat = ioerror)

      read(2,*) readtemp_str
      read(2,*) num_states,norb_2,nelec_2,actorb,actel
      read(2,*) readtemp_str

      if(norb_2.ne.norb) then
        write(*,*) "Your orbital count is incorrect. Please check your input and MolEl.dat files"
        stop
       end if

      do i=1,norb
       do j=i,norb
        read(2,*) x,y,readtemp
         Smat(x,y)=readtemp
         Smat(y,x)=readtemp
       end do
      end do

      do while (trim(readtemp_str).ne."Effective")
       read(2,*) readtemp_str
      end do
      
      x=0
      do while (trim(readtemp_str).ne."State".and.x.ne.state_num)
       read(2,*) readtemp_str,x
      end do

      write(*,*) "Reading Fock Matrix for ",state_num

      do i=1,norb
       do j=1,norb
        read(2,*) x,y,readtemp
        H_Two(x,y)=readtemp
       end do
      end do
      close(2)
      write(*,*) 'Done Getting HF Values'
      end subroutine

 
      
      Subroutine Get_HF_GAMESS(inputfile,numatomic,H_two,Smat,norb)
      !use FunctionMod
      implicit none

      character(len=100) :: inputfile,datafile
      real(8),allocatable,dimension(:,:) :: H_Two,Smat,mo_data,mo_data2,ECP_m,ECP_a,ECP_temp
      integer :: numatomic,ioerror,i,j,norb

      20 format(A)
      allocate(H_two(1:numatomic,1:numatomic))
      allocate(Smat(1:numatomic,1:numatomic))
      H_two=0
      Smat=0

      datafile = trim(inputfile) // "_Smat"
      open(unit=2,file=datafile,action='READ', iostat = ioerror)

      do i=1,numatomic
       do j=1,numatomic
        read(2,*) Smat(i,j)
       end do
      end do
      close(2)

       write(*,*) 'Start Htwo'
      datafile = trim(inputfile) // "_Htwo"
      open(unit=3,file=datafile,action='READ', iostat = ioerror)

      do i=1,numatomic
       do j=1,numatomic
        read(3,*) H_Two(i,j)
       end do
      end do
      close(3)

      write(*,*) 'Get ECP'
      datafile = trim(inputfile) // "_ecp"
      open(unit=4,file=datafile,status='OLD',action='READ', iostat = ioerror)

     if(ioerror.eq.0) then
     write(*,*) 'ECP found'
     allocate(ECP_m(1:norb,1:norb))
     allocate(ECP_temp(1:numatomic,1:norb))
     allocate(ECP_a(1:numatomic,1:numatomic))
     allocate(mo_data(1:numatomic,1:norb))
     allocate(mo_data2(1:norb,1:numatomic))
 
      do i=1,norb
       do j=1,norb
        read(4,*) ECP_m(i,j)
       end do
      end do

      close(5)

      datafile = trim(inputfile) // ".mo_dat"
      open(unit=6,file=datafile,action='READ', iostat = ioerror)

      do i=1,numatomic
       do j=1,norb
        read(6,*) mo_data(i,j)
       end do
      end do
      close(6)
      mo_data2 = transpose(mo_data)

      ECP_temp=matmul_dgemm(mo_data,ECP_m)
      ECP_a=matmul_dgemm(ECP_temp,mo_data2)
      H_two=H_two-ECP_a
      else
       write(*,*) "No removal of ECP"
      end if
      write(*,*) 'H_Two',size(H_Two)
      write(*,*) 'Smat',size(Smat)
      !stop
      write(*,*) 'Done Getting HF Values from GAMESS'
      end subroutine

      Subroutine Get_HF_PySCF(inputfile,numatomic,H_two,Smat,norb)
      !Use InterfaceMod
      !use FunctionMod
      implicit none

      character(len=100) :: inputfile,datafile,readtemp
      real(8),allocatable,dimension(:,:) :: H_Two,Smat,mo_data,mo_data2,ECP_m,ECP_a,ECP_temp
      integer :: numatomic,ioerror,i,j,norb,tempx,tempy
      real(8) :: tempval

      20 format(A)
      allocate(H_two(1:numatomic,1:numatomic))
      allocate(Smat(1:numatomic,1:numatomic))
      H_two=0
      Smat=0

      datafile = trim(inputfile) // ".scf_dat"
      open(unit=2,file=datafile,action='READ', iostat = ioerror)

      do while(trim(readtemp)/="Overlap Matrix")
       read(2,'(A)') readtemp
       end do
      do i=1,numatomic
       do j=1,numatomic
        read(2,*) tempx,tempy,tempval
        Smat(tempx,tempy)=tempval
       end do
      end do

      do while(trim(readtemp)/="Fock Matrix")
        read(2,'(A)') readtemp
       end do

      do i=1,numatomic
       do j=1,numatomic
        read(2,*) tempx,tempy,tempval
        H_Two(tempx,tempy)=tempval
       end do
      end do
      close(2)
     end subroutine


      subroutine Get_HF_libint(inputfile,norb,numact,H_one,Smat,mo_coeff,OneInts,TwoIntsCompact)
      use TypeMod
      !use FunctionMod
      implicit none

      character(len=100) :: inputfile,datafile
      real(8),allocatable,dimension(:,:) :: H_one,H_Two,Smat,OneInts,mo_coeff
      real(8),allocatable,dimension(:) :: TwoIntsCompact
      integer :: norb,numact,ioerror,i,j,k,l
      integer(8) :: index1,index2,compindex1
      integer, dimension(1:4) :: orbind
      real(8) :: integral  

      allocate(H_one(1:norb,1:norb))
      allocate(Smat(1:norb,1:norb))
      allocate(mo_coeff(1:norb,1:norb))
      allocate(OneInts(1:numact,1:numact))
      allocate(TwoIntsCompact(1:numact*(numact+1)/2*(numact*(numact+1)/2+1)/2))

      20 format(A)
      datafile = inputfile // ".Hone"
      open(unit=1,file=datafile,action='READ', iostat = ioerror)

      do i=1,norb
       do j=1,norb
        read(1,20) H_one(i,j)
       end do
      end do
      close(1)

      datafile = inputfile // ".Smat"
      open(unit=2,file=datafile,action='READ', iostat = ioerror)

      do i=1,norb
       do j=1,norb
        read(2,20) Smat(i,j)
       end do
      end do
      close(2)


      open(unit=6,file=datafile,action='READ',iostat= ioerror)

      do i=1,norb
       do j=1,norb
        read(6,20) mo_coeff(i,j)
       end do
      end do

 
      datafile = inputfile // ".OneInts"
      open(unit=4,file=datafile,action='READ', iostat = ioerror)

      do i=1,norb
       do j=1,norb
        read(4,20) OneInts(i,j)
       end do
      end do
      close(4)

      datafile = inputfile // ".TwoInts"
      open(unit=5,file=datafile,action='READ', iostat = ioerror)

       do while(orbind(1).ne.0)

         read(5,*) orbind(1),orbind(2),orbind(3),orbind(4),integral
                     i = orbind(1)
                     k = orbind(2)
                     j = orbind(3)
                     l = orbind(4)
                     index1 = FirstIndex(i,k)
                     index2 = FirstIndex(j,l)
                     compindex1 = CompositeIndex(index1,index2)

                    TwoIntsCompact(compindex1) = integral

        end do


      end subroutine


      subroutine IntTransform(TwoIntsCompact,mo_coeff)
      use TypeMod
      implicit none

      real(8), allocatable,dimension(:,:) :: mo_coeff
      real(8), allocatable,dimension(:) :: TwoIntsCompact
      !There is nothing here yet!      

      end subroutine

      subroutine Calculate_Coupling_MoleculeWBL(Coupling_R,Coupling_L,localden_fermi)
      implicit none    

      real(8) :: Coupling_R,Coupling_L,localden_fermi

      Coupling_R =  0.2
      Coupling_L =  0.2
      end subroutine


      subroutine Electrodes_MoleculeWBL(Sigma_l,Sigma_r,Smat,Coupling_R,Coupling_L,size_c,size_lc,size_lcr)
      implicit none
      integer :: size_lc,size_c,size_lcr,i,j
      real(8) :: Coupling_R,Coupling_L,temp
      complex(8), allocatable, dimension(:,:) :: Sigma_L,Sigma_R
      real(8), allocatable, dimension(:,:) :: Smat_LE,Smat_RE,Smat

      do i = size_c+1,size_lc
       do j = 1,size_c
      Sigma_L(i,j) = CMPLX(0,-0.5*Coupling_L*Smat(i,j),8)
      Sigma_L(j,i) = CMPLX(0,-0.5*Coupling_L*Smat(j,i),8)
       end do
      end do
      do i=size_lc+1,size_lcr
       do j= 1,size_c
      Sigma_R(i,j) = CMPLX(0,-0.5*Coupling_R*Smat(i,j),8)
      Sigma_R(j,i) = CMPLX(0,-0.5*Coupling_R*Smat(j,i),8)
       end do
      end do
      end subroutine


      subroutine Electrodes_MetalWBL(Sigma_l,Sigma_r,Smat_re,Smat_le,H_Two_le,H_Two_re,localden_fermi_l,localden_fermi_r,size_c,size_lc,size_lcr,size_l,size_r,H_Two_le_trans,H_Two_re_trans,write_ruqt_data,inputfile)
      !use FunctionMod
      implicit none
      integer :: size_lc,size_c,size_lcr,i,j,size_l,size_r,ioerror
      real(8) :: Coupling_R,Coupling_L,temp,localden_fermi_l,localden_fermi_r,pi
      complex(8), allocatable, dimension(:,:) :: Sigma_L,Sigma_R
      real(8), allocatable, dimension(:,:) :: Smat_LE,Smat_RE,Smat,H_Two_re,H_Two_le,Sigma_L_temp,Sigma_R_temp,Sigma_R_temp2,Sigma_L_temp2,sigma_r_temp3,sigma_l_temp3,H_Two_le_trans,H_Two_re_trans
      logical :: write_ruqt_data
      character(len=100) :: inputfile,outfile

      pi = 3.14159265359

      allocate(sigma_l_temp(1:size_l,1:size_l))
      allocate(sigma_l_temp2(1:size_c,1:size_l))
      allocate(sigma_l_temp3(1:size_c,1:size_c))

       !write(*,*) 'alloc l done'
      Sigma_L_temp = -pi*localden_fermi_l*Smat_le

       call matmul_dgemm2(H_Two_le_trans,Sigma_L_temp,Sigma_L_temp2)
       call matmul_dgemm2(Sigma_L_temp2,H_Two_le,Sigma_L_temp3)
      Sigma_L = CMPLX(0,sigma_l_temp3,8)

      !write(*,*) 'l done'
      deallocate(sigma_l_temp)
      deallocate(sigma_l_temp2)
      deallocate(sigma_l_temp3)

      allocate(sigma_r_temp(1:size_r,1:size_r))
      allocate(sigma_r_temp2(1:size_c,1:size_r))
      allocate(sigma_r_temp3(1:size_c,1:size_c))

      Sigma_R_temp = -pi*localden_fermi_r*Smat_re

       call matmul_dgemm2(H_Two_re_trans,Sigma_R_temp,Sigma_R_temp2)
       call matmul_dgemm2(Sigma_R_temp2,H_Two_re,Sigma_R_temp3)
      Sigma_R = CMPLX(0,Sigma_R_temp3,8)

      deallocate(Sigma_R_temp)
      deallocate(Sigma_R_temp2)
      deallocate(Sigma_R_temp3)

      if(write_ruqt_data) then
        outfile = trim(inputfile) // ".Sigma"
        open(unit=8,file=outfile,action='write',iostat=ioerror)

        write(8,*) size_l,size_c,size_r
        write(8,*) "RUQT Sigma Matrices"
        do j=1,size_c
          do i=1,size_c
             write(8,*) j,i,Sigma_L(j,i)
             end do
            end do
         do j=1,size_c
           do i=1,size_c
              write(8,*) j,i,Sigma_R(j,i)
             end do
            end do
       close(8)
      end if
      !write(*,*) 'Sigma Calculated'

      end subroutine
      !contains

      !subroutine matmul_dgemm2(leftmatrix,rightmatrix,outmat)
      !Implicit NONE
      !real(8), allocatable, dimension(:,:) :: leftmatrix,rightmatrix
      !real(8), allocatable, dimension(:,:) :: outmat
      !integer :: lmr, lmc,cmc
      !integer :: rmr, rmc,cmr

      !lmr = size(leftmatrix,1)  !left matrix row size
      !lmc = size(leftmatrix,2)  !left matrix col size
      !rmr = size(rightmatrix,1) !right matrix row size
      !rmc = size(rightmatrix,2) !right matrix col size
      !cmr= size(outmat,1)
      !1cmc=size(outmat,2)
      !outmat = 0.0
      !if(lmr.ne.0.and.lmc.ne.0.and.rmr.ne.0.and.rmc.ne.0)  then
      !Call DGEMM("N","N",lmr,rmc,lmc,1.d0,leftmatrix,lmr,rightmatrix,rmr,0.d0,outmat,cmr)
      !end if
      !end subroutine matmul_dgemm2

     ! end subroutine

              subroutine PartitionHS_MetalWBL(Smat,H_Two,size_l,size_r,size_c,size_lc,size_lcr,Smat_le,Smat_re,Smat_cen,H_Two_le,H_Two_re,H_Two_cen,H_Two_le_trans,H_Two_re_trans,write_ruqt_data,inputfile,qc_code)
              implicit none

              real(8), allocatable, dimension(:,:) :: Smat,Smat_le,Smat_re,Smat_cen,H_Two,H_Two_cen,H_Two_re,H_Two_le,temp,H_Two_le_trans,H_Two_re_trans
              real(8), allocatable, dimension(:) :: eigen,work
              integer :: size_l,size_c,size_r,size_lc,size_lcr,info,lwork,i,j,ioerror
              logical :: write_ruqt_data
              character(len=100) :: inputfile,outfile
              character(len=40) :: qc_code

              allocate(Smat_le(1:size_l,1:size_l))
              allocate(Smat_re(1:size_r,1:size_r))
              allocate(Smat_cen(1:size_c,1:size_c))
              Smat_le(1:size_l,1:size_l)=Smat(1:size_l,1:size_l)
              Smat_re(1:size_r,1:size_r)=Smat(size_lc+1:size_lcr,size_lc+1:size_lcr)
              Smat_cen(1:size_c,1:size_c)=Smat(size_l+1:size_lc,size_l+1:size_lc)

              ! write(*,*) 'Done partitioning S Matrix'
              !deallocate(Smat)

              allocate(H_Two_le(1:size_l,1:size_c))
              allocate(H_Two_re(1:size_r,1:size_c))
              allocate(H_Two_cen(1:size_c,1:size_c))
              allocate(H_Two_le_trans(1:size_c,1:size_l))
              allocate(H_Two_re_trans(1:size_c,1:size_r))

              if(trim(qc_code).eq."pyruqt") then
               H_Two_le(1:size_l,1:size_c)=H_Two(1:size_l,size_l+1:size_lc)
               H_Two_re(1:size_r,1:size_c)=H_Two(size_lc+1:size_lcr,size_l+1:size_lc)
               H_Two_cen(1:size_c,1:size_c)=H_Two(size_l+1:size_lc,size_l+1:size_lc)
               H_Two_le_trans(1:size_c,1:size_l) = H_Two(size_l+1:size_lc,1:size_l) !transpose(H_Two_le)
               H_Two_re_trans(1:size_c,1:size_r) = H_Two(size_l+1:size_lc,size_lc+1:size_lcr)!transpose(H_Two_re)
              else
               H_Two_le(1:size_l,1:size_c)=27.2114*H_Two(1:size_l,size_l+1:size_lc)
               H_Two_re(1:size_r,1:size_c)=27.2114*H_Two(size_lc+1:size_lcr,size_l+1:size_lc)
               H_Two_cen(1:size_c,1:size_c)=27.2114*H_Two(size_l+1:size_lc,size_l+1:size_lc)
               H_Two_le_trans(1:size_c,1:size_l) = 27.2114*H_Two(size_l+1:size_lc,1:size_l) !transpose(H_Two_le)
               H_Two_re_trans(1:size_c,1:size_r) = 27.2114*H_Two(size_l+1:size_lc,size_lc+1:size_lcr)!transpose(H_Two_re)
              end if
              !write(*,*) 'Done partitioning Fock Matrix'
              !deallocate(H_Two) 
              if(write_ruqt_data) then
               outfile = trim(inputfile) // ".partdat"
               open(unit=8,file=outfile,action='write',iostat=ioerror)


               write(8,*) size_l,size_c,size_r
               write(8,*) "RUQT Overlap Matrices"
               do j=1,size_l
                 do i=1,size_l
                  write(8,*) j,i,Smat_le(j,i)
                end do
               end do
               do j=1,size_c
                do i=1,size_c
                 write(8,*) j,i,Smat_cen(j,i)
                end do
               end do
               do j=1,size_r
                do i=1,size_r
                 write(8,*) j,i, Smat_re(j,i)
                end do
               end do

               write(8,*) "RUQT Fock Matrices in Hartrees"
               do j=1,size_l
                do i=1,size_c
                  write(8,*) j,i,H_Two_le(j,i)/27.2114
                end do
               end do
               do j=1,size_r
                do i=1,size_c
                 write(8,*) j,i,H_Two_re(j,i)/27.2114
                end do
               end do
               do j=1,size_c
                do i=1,size_c
                 write(8,*) j,i,H_Two_cen(j,i)/27.2114
                end do
               end do

               close(8)
              end if
              end subroutine


              function fermi_function(energy,fermi_energy,KT)
              implicit none
              real(8) :: energydiff,KT,energy,fermi_energy,fermi_function

              energydiff = energy - fermi_energy
              fermi_function = 1.00/(exp(energydiff/KT)+1)
              end function

              function inv(A) result(Ainv)
               implicit none
               complex(8),allocatable, dimension(:,:), intent(in) :: A
               complex(8), allocatable, dimension(:,:) :: Ainv

               complex(8),allocatable, dimension(:) :: work  ! work array for LAPACK
               integer,allocatable, dimension(:) :: ipiv   ! pivot indices
               integer :: n, info

               allocate(Ainv(1:size(A,1),1:size(A,2)))
               allocate(work(1:size(A,1)))
               allocate(ipiv(1:size(A,1)))

               Ainv = A
               n = size(A,1)

               call ZGETRF(n, n, Ainv, n, ipiv, info)

               if (info /= 0) then
                 write(*,*) info!,A
                 stop 'Matrix is numerically singular!'
                end if

               call ZGETRI(n, Ainv, n, ipiv, work, n, info)

               if (info /= 0) then
                 write(*,*) info
                 stop 'Matrix inversion failed!'
                end if

                deallocate(work)
                deallocate(ipiv)
               end function inv     


              function inv_real(A) result(Ainv)
               implicit none
               real(8),allocatable, dimension(:,:), intent(in) :: A
               real(8), allocatable, dimension(:,:) :: Ainv

               real(8),allocatable, dimension(:) :: work  ! work array for LAPACK
               integer,allocatable, dimension(:) :: ipiv   ! pivot indices
               integer :: n, info


               allocate(Ainv(1:size(A,1),1:size(A,2)))
               allocate(work(1:size(A,1)))
               allocate(ipiv(1:size(A,1)))

               Ainv = A
               n = size(A,1)

               call DGETRF(n, n, Ainv, n, ipiv, info)

               if (info /= 0) then
                 write(*,*) info!,A
                 stop 'Matrix is numerically singular!'
                end if

               call DGETRI(n, Ainv, n, ipiv, work, n, info)

               if (info /= 0) then
                 write(*,*) info
                 stop 'Matrix inversion failed!'
                end if

                deallocate(work)
                deallocate(ipiv)
               end function inv_real

               function FirstIndex(i,k)

               Implicit None
               integer :: i,k
               integer(8) :: FirstIndex

               if(i.lt.k) then
                 FirstIndex = (k-1)*k/2 + i
               else
                 FirstIndex = (i-1)*i/2 + k
               end if

               end function FirstIndex


      function matmul_zgemm(leftmatrix,rightmatrix)
      Implicit NONE
      complex(8), allocatable, dimension(:,:) :: leftmatrix,rightmatrix
      complex(8), allocatable, dimension(:,:) :: matmul_zgemm
      integer :: lmr, lmc
      integer :: rmr, rmc

      lmr = size(leftmatrix,1)  !left matrix row size
      lmc = size(leftmatrix,2)  !left matrix col size
      rmr = size(rightmatrix,1) !right matrix row size
      rmc = size(rightmatrix,2) !right matrix col size

      allocate(matmul_zgemm(1:lmr,1:rmc))
      matmul_zgemm(1:lmr,1:rmc) = 0.0
      if(lmr.ne.0.and.lmc.ne.0.and.rmr.ne.0.and.rmc.ne.0)  then
      Call ZGEMM('N','N',lmr,rmc,lmc,1.d0,leftmatrix,lmr,rightmatrix,rmr,0.d0,matmul_zgemm,lmr)
      end if
      end function matmul_zgemm

      function matmul_dgemm(leftmatrix,rightmatrix)
      Implicit NONE
      real(8), allocatable, dimension(:,:) :: leftmatrix,rightmatrix
      real(8), allocatable, dimension(:,:) :: matmul_dgemm
      integer :: lmr, lmc
      integer :: rmr, rmc

      lmr = size(leftmatrix,1)  !left matrix row size
      lmc = size(leftmatrix,2)  !left matrix col size
      rmr = size(rightmatrix,1) !right matrix row size
      rmc = size(rightmatrix,2) !right matrix col size

      matmul_dgemm = 0.0
      if(lmr.ne.0.and.lmc.ne.0.and.rmr.ne.0.and.rmc.ne.0)  then
      Call DGEMM('N','N',lmr,rmc,lmc,1.d0,leftmatrix,lmr,rightmatrix,rmr,0.d0,matmul_dgemm,lmr)
      end if
      end function matmul_dgemm

      subroutine matmul_dgemm2(leftmatrix,rightmatrix,outmat)
      Implicit NONE
      real(8), allocatable, dimension(:,:) :: leftmatrix,rightmatrix
      real(8), allocatable, dimension(:,:) :: outmat
      integer :: lmr, lmc,cmc
      integer :: rmr, rmc,cmr

      lmr = size(leftmatrix,1)  !left matrix row size
      lmc = size(leftmatrix,2)  !left matrix col size
      rmr = size(rightmatrix,1) !right matrix row size
      rmc = size(rightmatrix,2) !right matrix col size
      cmr= size(outmat,1)
      cmc=size(outmat,2)
      outmat = 0.0
      !write(*,*) lmr,lmc,rmr,rmc,cmr,cmc
      !write(*,*) "lmat",leftmatrix
      !write(*,*) "rmat",rightmatrix
      !write(*,*) "outmat",outmat
      if(lmr.ne.0.and.lmc.ne.0.and.rmr.ne.0.and.rmc.ne.0)  then
      Call DGEMM("N","N",lmr,rmc,lmc,1.d0,leftmatrix,lmr,rightmatrix,rmr,0.d0,outmat,cmr)
      end if
      end subroutine matmul_dgemm2


              function CompositeIndex(ik,jl)
              Implicit NONE

              integer(8) :: ik, jl
              integer(8) :: CompositeIndex

               CompositeIndex = 0

              if(ik.lt.jl) then
                CompositeIndex =  (jl-1)*jl/2 +ik
              else
                CompositeIndex =  (ik-1)*ik/2 + jl
              end if

              end function CompositeIndex
         !This subroutine reads in the input file and assigns all variables for the calculation for standalone RUQT. Not used in pip version of pyRUQT.
          !Left for reference as it contains all input variables in one place.
     subroutine ReadInput(inputfile,norb,numfcore,numfvirt,numocc,numvirt,size_l,size_r,size_c,energy_start,energy_end,delta_en,volt_start,volt_end,delta_volt,inputcode,KT,Electrode_Type,Fermi_enl,Fermi_enr,CalcType,localden_fermi_l,localden_fermi_r,doubles,numatomic,functional,num_threads,use_b0,b0_type,write_ruqt_data,state_num)
     implicit none
     character(len=100) :: inputfile
     character(len=40) :: inputcode,filename,Electrode_Type,CalcType,functional,b0_type
     integer :: norb, size_c,size_r,size_l,numfcore,numfvirt,numocc,numvirt,numatomic,num_threads
     real(8) :: energy_start,energy_end,delta_en,volt_start,volt_end,delta_volt,KT
     logical :: libint,doubles,use_b0,write_ruqt_data
     real(8) :: Fermi_enl,Fermi_enr,localden_fermi_l,localden_fermi_r
     integer :: state_num
    
              filename = trim(inputfile)
              open(unit=1,file=filename,action="read")
              20 format(A) 
              num_threads=0
              read(1,20) CalcType
              read(1,20) Electrode_Type
              read(1,*) Fermi_enl
              read(1,*) Fermi_enr
              read(1,*) localden_fermi_l
              read(1,*) localden_fermi_r
              read(1,*) norb
              read(1,*) numatomic
              read(1,*) numfcore
              read(1,*) numfvirt
              read(1,*) numocc
              read(1,*) numvirt
              read(1,*) size_c
              read(1,*) size_l
              read(1,*) size_r
              read(1,*) energy_start
              read(1,*) energy_end
              read(1,*) delta_en
              read(1,*) volt_start
              read(1,*) volt_end
              read(1,*) delta_volt
              read(1,*) KT
              read(1,*) inputcode
              read(1,*) doubles
              read(1,*) functional
              read(1,*) use_b0
              read(1,*) b0_type
              read(1,*) write_ruqt_data
              read(1,*) num_threads
              read(1,*) state_num
                      
              close(1)
              if(num_threads.eq.0) then
                 num_threads=1
                end if
              end subroutine 

                      subroutine Flag_set(inputcode,functional,cisd_flag,rdm_flag,hf_flag,dft_flag,qchem,gamess,pyscf,maple,molcas,pyRUQT)
                      implicit none

                      character(len=40) :: inputcode,functional
                      logical :: rdm_flag,cisd_flag,dft_flag,hf_flag
                      logical :: qchem,gamess,pyscf,maple,molcas,pyRUQT

                       rdm_flag=.false.
                       cisd_flag=.false.
                       dft_flag=.false.
                       hf_flag=.false.
                       qchem=.false.
                       gamess=.false.
                       pyscf=.false.
                       maple=.false.
                       molcas=.false.
                       pyRUQT=.false.

                      if(inputcode.eq."qchem") then
                         qchem=.true.
                       if(functional.eq."dft") then
                         dft_flag=.true. 
                        elseif(functional.eq."hf") then
                         hf_flag=.true.
                        else
                         write(*,*) "Your method and QC code choice do not work together. Exiting"
                         stop
                        end if

                       elseif(inputcode.eq."molcas") then
                         molcas=.true.
                       if(functional.eq."dft") then
                         dft_flag=.true.
                        elseif(functional.eq."hf") then
                         hf_flag=.true.
                        else
                         write(*,*) "Your method and QC code choice do not work together. Exiting"
                         stop
                        end if

                       elseif(inputcode.eq."pyruqt") then
                         pyRUQT=.true.
                       if(functional.eq."dft") then
                         dft_flag=.true.
                        elseif(functional.eq."hf") then
                         hf_flag=.true.
                        else
                         write(*,*) "Your method and QC code choice do not work together. Exiting"
                         stop
                        end if


                       elseif(inputcode.eq."gamess") then
                         gamess=.true.
                       if(functional.eq."rdm") then
                         rdm_flag=.true.
                elseif(functional.eq."hf") then
                 hf_flag=.true.
                elseif(functional.eq."cisd") then
                 cisd_flag=.true.
                else
                 write(*,*) "Your method and QC code choice do not work together. Exiting"
                 stop
                end if

               elseif(inputcode.eq."pyscf") then
                 pyscf=.true.
                 if(functional.eq."hf") then
                  hf_flag=.true.
                 elseif(functional.eq."dft") then
                  dft_flag=.true.
                 else
                  write(*,*) "Your method and QC code choice do not work together. Exiting"
                  stop
                 end if


               elseif(inputcode.eq."maple") then
                 maple=.true.
               if(functional.eq."rdm") then
                 rdm_flag=.true.
                elseif(functional.eq."hf") then
                 hf_flag=.true.
                elseif(functional.eq."cisd") then
                 cisd_flag=.true.
                elseif(functional.eq."dft") then
                 dft_flag=.true.
                else
                 write(*,*) "Your method and QC code choice do not work together. Exiting"
                 stop
                end if


               else
                write(*,*) "Your QC code choice is not supported. Exiting"
                stop
               end if
              end subroutine
             !end module
            !end subroutine
      subroutine Build_G_SD_Invert(G_C,Sigma_l,Sigma_r,energy,size_l,size_c,size_lc,size_lcr,norb,inputfile,numocc,numvirt,iter,B1data,B2data,mo_ener,mo_coeff,mo_coeff2,doubles,currentflag,energy_values,ener_val,G_S,corr_ener,numatomic,B0_Coeff,use_b0,gamess,maple,numfcore,numfvirt,b0_type)
      !use FunctionMod
      use TypeMod
      implicit none
      character(len=40) :: inputfile,Bfile,Bfile2,readtemp,cstring,b0_type
      integer :: size_l,size_lc,size_c,size_lcr,i,j,ioerror,norb,iter,a,b,counter
      integer :: numocc,numvirt,k,p,q,z,y,x,r,s,t,energy_values,ener_val,numatomic
      integer :: aa,bb,pp,qq,ii,jj,tt,rr,ss,xx,yy,zz
      integer :: tempx, tempy,numvirt_act,numocc_act,numfvirt,numfcore
      real(8) :: temp,corr_ener,hf_energy,energy,tempval,dft_energy,use_dft_mo
      complex(8), allocatable, dimension(:,:) :: G_C,G_temp,Sigma_L,Sigma_R
      type(B2) :: B2data
      type(B1) :: B1data
      real(8) :: cisd_b0
      real(8), allocatable, dimension(:,:) :: mo_coeff,mo_coeff2,temp_gf,temp_gf2
      real(8), allocatable, dimension(:) :: mo_ener,B0_coeff
      logical :: singles,doubles,currentflag,use_b0,maple,gamess
      type(energr) :: G_S
      integer(8) :: rt1,rt2,rt3,rt4,rt5,io 
      numocc_act=numocc-numfcore
      numvirt_act=numvirt-numfvirt
      if(doubles.eqv..true.) then
        singles = .false.
       else if(doubles.eqv..false.) then
        singles = .true.
      end if

      10 format(A)
      if(iter.eq.1) then
       if(doubles.eqv..true.) then
       allocate(B1data%a%o(numocc+1:numocc+numvirt_act,numfcore+1:numocc))
       allocate(B1data%b%o(numocc+1:numocc+numvirt_act,numfcore+1:numocc))
       allocate(B2data%aa%m(numocc+1:numocc+numvirt_act,numfcore+1:numocc))
       allocate(B2data%ab%m(numocc+1:numocc+numvirt_act,numfcore+1:numocc))
       B1data%a%o=0
       B1data%b%o=0
       end if
       allocate(mo_ener(1:norb))
       allocate(mo_coeff(1:numatomic,1:norb))
       allocate(mo_coeff2(1:norb,1:numatomic))
       mo_coeff = 0
       mo_ener = 0
       mo_coeff2 = 0
       if(doubles.eqv..true.) then
       counter=1
       do j=numfcore+1,numocc
        do a = numocc+1,numocc+numvirt_act
          allocate(B2data%aa%m(a,j)%n(numocc+1:numocc+numvirt_act,numfcore+1:numocc))
          allocate(B2data%ab%m(a,j)%n(numocc+1:numocc+numvirt_act,numfcore+1:numocc))
            B2data%aa%m(a,j)%n=0
            B2data%ab%m(a,j)%n=0
         end do
        end do
       if(gamess.eqv..true.) then
         Bfile = trim(inputfile) // "T2"
         open(unit=4,file=Bfile,action='READ',iostat = ioerror)
         read(4,*) readtemp
         write(*,*) "GAMESS version does not work with frozen orbitals."
         write(*,*) "Do not use any frozen or core orbitals. ECP only."
         write(*,*) 'Total Number of T2(ab) and T2(aa) Elements',readtemp
        do i=1,numocc_act
         do a=numocc_act+1,numocc_act+numvirt_act
          do j=1,numocc_act
            do b=numocc_act+1,numvirt_act+numocc_act
              read(4,*) B2data%aa%m(a,i)%n(b,j)
              counter=counter+1
             end do
            end do
           end do
         end do
        do i=1,numocc_act
         do a=numocc_act+1,numocc_act+numvirt_act
          do j=1,numocc_act
            do b=numocc_act+1,numvirt_act+numocc_act
              read(4,*) B2data%ab%m(a,i)%n(b,j)
              counter=counter+1
             end do
            end do
           end do
         end do

      ! do i=1,numocc
      !   do a=numocc+1,numocc+numvirt
      !    do j=1,numocc
      !     do b=numocc+1,numvirt+numocc
             !write(*,*) i,a,j,b
       !      read(4,*) readtemp!B2data%bb%m(a,i)%n(b,j)
       !     end do
       !    end do
       !   end do
       ! end do


       do i=1,numocc_act
        do a=numocc_act+1,numvirt_act+numocc_act
          read(4,*) B1data%a%o(a,i)
             counter=counter+1
         end do
       end do
       write(*,*) 'Number of T2(ab) and T1(a) Values in T2 file',counter-1

        elseif(maple.eqv..true.) then
         Bfile = trim(inputfile) // ".T2"
         open(unit=4,file=Bfile,action='READ',iostat = ioerror)
         read(4,*) cstring,readtemp,corr_ener,use_dft_mo,hf_energy,dft_energy
         read(4,*) use_dft_mo,hf_energy,dft_energy

          write(*,*) 'Max Number of T2(ab) and T2(aa) Elements',readtemp
          if(use_dft_mo.eq.1) then
            write(*,*) "Using DFT MO Energies"
            corr_ener=hf_energy+corr_ener-dft_energy
           end if
          write(*,*) 'Difference in Energy',corr_ener

          counter=0

         do
           counter=counter+1
           read(4,*,IOSTAT=io) rt5,rt1,rt2,rt3,rt4,tempval
           if(io.eq.0..and.rt5.ne.-1) then
            if(rt5.eq.2.and.rt1.ne.0) then
              B2data%aa%m(rt4+numfcore,rt1+numfcore)%n(rt3+numfcore,rt2+numfcore)=tempval
            elseif(rt5.eq.1.and.rt1.ne.0) then
              B2data%ab%m(rt4+numfcore,rt1+numfcore)%n(rt3+numfcore,rt2+numfcore)=tempval
            elseif(rt5.eq.0.and.rt1.ne.0) then
              !add T2(bb) here once open shell is complete
            elseif(rt5.eq.1.and.rt1.eq.0) then
              B1data%a%o(rt4+numfcore,rt3+numfcore)=tempval
            elseif(rt5.eq.0.and.rt1.eq.0) then
              B1data%b%o(rt4+numfcore,rt3+numfcore)=tempval
            end if
           else
            write(*,*) "Done reading T2 file. ",counter," terms read."
            exit
          end if
         end do



       end if

       close(4)

       end if
       if(doubles.eqv..true.) then
        if(use_b0.eqv..false.) then
          write(*,*) "Using CEPA conventation for B0"
         elseif(use_b0.eqv..true.) then
         if(trim(b0_type).eq."cisd") then
          write(*,*) "Using CISD conventation for B0"
          Call Build_B0_CISD(cisd_b0,norb,numfcore,numvirt_act,numocc,numvirt,B1data,B2data)
         end if
        end if
       end if

       if(gamess.eqv..true.) then
       write(*,*) 'Reading MO info'
       Bfile2 = trim(inputfile) // ".mo_dat"
       open(unit=5,file=Bfile2,action='READ',iostat = ioerror)
        do i=1,numatomic
         do j=1,norb
           read(5,*) mo_coeff(i,j)
           mo_coeff2(j,i) = mo_coeff(i,j)
          end do
         end do

        do i=1,norb
          read(5,*) mo_ener(i)
         end do      
        read(5,*) corr_ener
        write(*,*) 'Correlation Energy',corr_ener

        elseif(maple.eqv..true.) then
         write(*,*) "Reading MO data from .scf_dat file"
         Bfile2 = trim(inputfile) // ".scf_dat"
         open(unit=5,file=Bfile2,action='READ',iostat = ioerror)
         do while(trim(readtemp)/="Molecular Orbital Coefficients")
            read(5,'(A)') readtemp
           end do
         do i=1,numatomic
          do j=1,norb
            read(5,*) tempx,tempy,tempval
            mo_coeff(tempx,tempy)=tempval
            mo_coeff2(j,i) = mo_coeff(i,j)
           end do
          end do
        read(5,'(A)')
        do i=1,norb
          read(5,*) tempx,mo_ener(i)
         end do

       end if

      allocate(G_S%en(1:energy_values))
      do i = 1,energy_values
      allocate(G_S%en(i)%gf(1:numatomic,1:numatomic))
       G_S%en(i)%gf = 0.0
      end do
      end if

      if(currentflag.eqv..false.) then
        allocate(temp_gf(1:norb,1:norb))
       if(singles.eqv..true.) then
        corr_ener = 0
       end if
       temp_gf = 0.0
      if(doubles.eqv..true.) then
       do i=numfcore+1,numocc
          if(use_b0.eqv..true..and.trim(b0_type).eq."cisd") then
           temp_gf(i,i) = cisd_b0*(energy-(corr_ener+mo_ener(i))*27.211396132)**(-1)
            else if(use_b0.eqv..true..and.trim(b0_type).eq."rdm") then
            temp_gf(i,i) = b0_coeff(i)**(2)*((energy-(corr_ener+mo_ener(i))*27.211396132)**(-1))
            else if(use_b0.eqv..false.) then
           temp_gf(i,i) =(energy-(corr_ener+mo_ener(i))*27.211396132)**(-1)
          end if
        end do 
       do a=numocc+1,numvirt_act+numocc
          if(use_b0.eqv..true..and.trim(b0_type).eq."cisd") then
           temp_gf(a,a) = cisd_b0*(energy+(corr_ener-mo_ener(a))*27.211396132)**(-1)
           else if(use_b0.eqv..true..and.trim(b0_type).eq."rdm") then
            temp_gf(a,a) = b0_coeff(a)**(2)*((energy+(corr_ener-mo_ener(a))*27.211396132)**(-1))
           else if(use_b0.eqv..false.) then
           temp_gf(a,a) = (energy+(corr_ener-mo_ener(a))*27.211396132)**(-1)
          end if
       end do 

       if(numfcore.gt.0) then
        do i=1,numfcore
           temp_gf(i,i) = (energy-(corr_ener+mo_ener(i))*27.211396132)**(-1)
         end do
       end if

       if(numfvirt.gt.0) then
       do a=numocc+1+numvirt_act,numocc+numvirt
          temp_gf(a,a) = (energy+(corr_ener-mo_ener(a))*27.211396132)**(-1)
        end do
       end if

       else

        do i=1,numocc
           temp_gf(i,i) = (energy-(mo_ener(i))*27.211396132)**(-1)
         end do

        do a=numocc+1,numvirt+numocc
          temp_gf(a,a) = (energy-(mo_ener(a))*27.211396132)**(-1)
        end do

      end if

      if(singles.eqv..true.) then
        goto 100
       else if(doubles.eqv..true.) then

       do a=numocc+1,numocc+numvirt_act
        do b=numocc+1,numocc+numvirt_act
         do q=numfcore+1,numocc
 
           temp_gf(a,b) = temp_gf(a,b) + B1data%a%o(a,q)*B1data%a%o(b,q)*(energy-(corr_ener+mo_ener(q))*27.211396132)**(-1)

          end do
         end do
        end do


        do a=numfcore+1,numocc
        do b=numfcore+1,numocc
         do p=numocc+1,numocc+numvirt_act

           temp_gf(a,b) = temp_gf(a,b) + B1data%a%o(p,a)*B1data%a%o(p,b)*(energy+(corr_ener-mo_ener(p))*27.211396132)**(-1)

          end do
         end do
        end do

       do a=numfcore+1,numocc
        do y=numfcore+1,numocc
         do z=numocc+1,numocc+numvirt_act

           temp_gf(a,a) = temp_gf(a,a) + B1data%a%o(z,y)*B1data%a%o(z,y)*(energy+(-corr_ener-mo_ener(a)-mo_ener(y)+mo_ener(z))*27.211396132)**(-1)

          end do
         end do
        end do

       do a=numocc+1,numocc+numvirt_act
        do b=numocc+1,numocc+numvirt_act
         do z=numocc+1,numocc+numvirt_act
          do x=numfcore+1,numocc
           do y=numfcore+1,numocc

           temp_gf(a,b) = temp_gf(a,b) + (B2data%ab%m(a,x)%n(z,y)*B2data%ab%m(b,x)%n(z,y))*((energy+(-corr_ener-mo_ener(x)-mo_ener(y)+mo_ener(z))*27.211396132)**(-1))
           if(x.gt.y.and.a.gt.z.and.b.gt.z) then
             temp_gf(a,b) = temp_gf(a,b) + (B2data%aa%m(a,x)%n(z,y)*B2data%aa%m(b,x)%n(z,y))*(energy+(-corr_ener-mo_ener(x)-mo_ener(y)+mo_ener(z))*27.211396132)**(-1)
            end if
            
          end do
         end do
        end do
       end do
      end do

      do a=numocc+1,numocc+numvirt_act
       do t=numfcore+1,numocc
         do r=numocc+1,numocc+numvirt_act

           temp_gf(a,a) = temp_gf(a,a) + B1data%a%o(r,t)*B1data%a%o(r,t)*(energy+(corr_ener-mo_ener(r)-mo_ener(a)+mo_ener(t))*27.211396132)**(-1)

          end do
         end do
        end do

       do a=numfcore+1,numocc
        do b=numfcore+1,numocc
         do t=numfcore+1,numocc
          do r=numocc+1,numocc+numvirt_act
           do s=numocc+1,numocc+numvirt_act

           if(r.gt.s.and.a.gt.t.and.b.gt.t) then
            temp_gf(a,b) = temp_gf(a,b) + (B2data%aa%m(r,t)%n(s,b)*B2data%aa%m(r,t)%n(s,a))*(energy+(corr_ener-mo_ener(r)-mo_ener(s)+mo_ener(t))*27.211396132)**(-1)
            end if

          end do
         end do
        end do
       end do
      end do

        end if
100     allocate(temp_gf2(1:norb,1:numatomic))

        temp_gf2 = 0
        call matmul_dgemm2(temp_gf,mo_coeff2,temp_gf2)
        call matmul_dgemm2(mo_coeff,temp_gf2,G_S%en(ener_val)%gf)
        G_S%en(ener_val)%gf = inv_real(G_S%en(ener_val)%gf)
        deallocate(temp_gf)
        deallocate(temp_gf2)

       end if

       G_C = CMPLX(0,0)
       G_C=G_S%en(ener_val)%gf(size_l+1:size_lc,size_l+1:size_lc)
       G_C = G_C - Sigma_l - Sigma_r
       G_C = inv(G_C)
      ! contains

      end subroutine

      subroutine Build_B0_CISD(cisd_b0,norb,numfcore,numfvirt,numocc,numvirt,B1data,B2data)
      !use InterfaceMod
      use TypeMod
      !use FunctionMod
      implicit none
      integer :: size_l,size_lc,size_c,size_lcr,i,j,ioerror,norb,iter,a,b,numfvirt
      integer :: numocc,numvirt,p,k,q,z,y,x,r,s,t,energy_values,numatomic,numfcore
      type(B2) :: B2data
      type(B1) :: B1data
      real(8) :: cisd_b0
      real(8) :: sum_B0,sum_B0_2,sum_B0_1

      write(*,*) 'Building B0'

      sum_B0_1=0

         do b=numocc+1,numocc+numfvirt
           do k=numfcore+1,numocc
 
           sum_B0_1 = sum_B0_1 - B1data%a%o(b,k)*B1data%a%o(b,k)

          end do
         end do


        sum_B0_2=0
        do a=numocc+1,numocc+numfvirt
         do k=numfcore+1,numocc
           do b=a,numfvirt+numocc
            do j=k,numocc
             sum_B0_2 = sum_B0_2 - B2data%ab%m(a,k)%n(b,j)*B2data%ab%m(a,k)%n(b,j)
              !write(*,*) a,k,b,j 
             if(k.ne.j.and.a.ne.b) then
              sum_B0_2 = sum_B0_2 - B2data%aa%m(a,k)%n(b,j)*B2data%aa%m(a,k)%n(b,j)
             end if

             end do
            end do
           end do
          end do

          cisd_b0 =1+sum_B0_1+sum_B0_2

          write(*,*) "B0 Final",cisd_b0
       end subroutine
      end module
