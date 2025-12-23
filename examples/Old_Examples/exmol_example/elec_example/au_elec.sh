#!/bin/bash -l


export CPUS=4
export MOLCAS_MEM=80000
export MOLCAS=/opt/app1/sandx_fock2
export MOLCAS_SOURCE=/opt/app1/sandx_fock2
export MOLCAS_PRINT=6
export PATH=~/bin:$PATH
export Project=au_elec
export Scratch=$PWD
export WorkDir=$Scratch/$Project
export CurrDir=$PWD
export OPENBLAS_NUM_THREADS=4


#rm $Scratch/$Project/*
pymolcas2 $CurrDir/$Project.inp &> $CurrDir/$Project.out 2>$CurrDir/$Project.err &
