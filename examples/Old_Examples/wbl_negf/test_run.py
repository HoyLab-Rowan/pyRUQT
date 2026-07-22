import ruqt
import pyruqt

#Example transmission calcualtion using RUQT-Fortran (metal wide band limit)
#Uses only the required keywords. Check documtation for optional keywords.
trans_calc=pyruqt.wbl_negf(output="wbl_example",exmol_dir="../exmol_example/ben/",num_elec_atoms=2)
trans_calc.transmission()
