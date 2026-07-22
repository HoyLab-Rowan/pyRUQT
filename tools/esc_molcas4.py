#!/usr/bin/env python

# This is a new code we are working on to optimize the esc_molcas calculator
# for easier, convinient and faster calculations and subsequently run pyRUQT after the presence of the MolEL.dat file to make the transmission graphs.
from ase.io import read
import os
import subprocess
import time


def main():
    # Take input from user
    filename = input("Enter the base filename for .inp and .sh files: ")
    xyzfile = input("Enter xyz file: ")
    basis = input("Enter preferred basis set: ")
    

    # Validate XYZ file existence
    if not os.path.isfile(xyzfile):
        print("Error: XYZ file does not exist.")
        return
    

    # Calculate inactive orbitals
    inactive_orbitals = calculate_inactive_orbitals(xyzfile)

    # Generate .inp and .sh files
    generate_input_files(filename, xyzfile, basis)

    # Run the .sh file
    run_sh_file(filename)

    # Wait for mole.dat file to be generated
    wait_for_mole_dat(filename)

    # Read the mole.dat file
    read_mole_dat(filename)

def calculate_inactive_orbitals(xyzfile):
    atoms = read(xyzfile, format='xyz')
    atomic_numbers = atoms.get_atomic_numbers()
    inactive_orbitals = int(sum(atomic_numbers) / 2 - active_space / 2)
    return inactive_orbitals


def generate_input_files(filename, xyzfile, basis):
    # Generate content for .inp file
    inactive_orbitals = calculate_inactive_orbitals(xyzfile)
    active_space = 8
    inp_lines = generate_inp_lines(xyzfile, basis, inactive_orbitals, active_space)

    # Generate content for .sh file
    sh_lines = generate_sh_lines(filename)

    # Write files
    write_file(f"{filename}.inp", inp_lines)
    write_file(f"{filename}.sh", sh_lines)
    os.chmod(f"{filename}.sh", 0o755)

def generate_inp_lines(xyzfile, basis, inactive_orbitals, active_space):
    inp_lines = [
        "&GATEWAY",
        f"coord = {xyzfile}",
        f"basis = {basis}",
        "group = nosym",
        "RICD",
        "",
        "&SEWARD",
        "NODEleted",
        "grid input",
        "grid = fine",
        "end of grid input",
        "",
        "&SCF",
        "",
        "&RASSCF &END",
        "nActEl",
        str(inactive_orbitals),
        "RAS2",
        str(active_space),
        "DELEted",
        "0 0 0 0",
        "Symmetry",
        "1",
        "Spin",
        "1",
        "CIROot",
        "1 1; 1",
        "",
        "&GRID_IT",
        " ALL",
        "",
        "&MCPDFT",
        " KSDFT=T:PBE",
        " GRADIENT",
        " End of input"
    ]
    return inp_lines

def generate_sh_lines(filename):
    sh_lines = [
        "#!/bin/bash -l",
        "export CPUS=4",
        "export MOLCAS_MEM=20000",
        "export MOLCAS=/opt/app1/sandx_fock2",
        "export MOLCAS_SOURCE=/opt/app1/sandx_fock2",
        "export MOLCAS_PRINT=6",
        "export PATH=~/bin:$PATH",
        f"export Project={filename}",
        "export Scratch=$PWD",
        "export WorkDir=$Scratch/$Project",
        "export CurrDir=$PWD",
        "export OPENBLAS_NUM_THREADS=4",
        "",
        "# rm $Scratch/$Project/*",
        f"pymolcas {filename}.inp &> {filename}.out 2>{filename}.err &"
    ]
    return sh_lines

def write_file(filename, lines):
    # Write lines to file
    with open(filename, "w") as f:
        f.write("\n".join(lines))

def run_sh_file(filename):
    calc_dir = "."  # Set the directory where the .sh file is located
    subprocess.run([f"./{filename}.sh"], cwd=calc_dir)

def wait_for_mole_dat(filename):
    calc_dir = "."  # Set the directory where the .sh file is located
    mole_dat_file = "MolEl.dat"
    while not os.path.exists(mole_dat_file):
        time.sleep(1)  # Wait for 1 second
    print("mole.dat file found.")

def read_mole_dat(filename):
    # code to read mole.dat file will go  here(call molel.matread)
    pass

if __name__ == "__main__":
    main()

