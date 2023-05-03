import ruqt
import pyruqt

trans_calc=pyruqt.sie_negf(output="sie_1",temp=300,exmol_dir="../exmol_example/ben/",elec_dir="../elec_example/au_elec/",ase_current=False)
trans_calc.current()

#To run this file put the following into the command line: python test_run.py &
#If the sie_1.err file is empty, your test was successful, check results against those in ref_data folder
