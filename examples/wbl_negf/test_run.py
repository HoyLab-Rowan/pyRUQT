import ruqt
import pyruqt

trans_calc=pyruqt.wbl_negf(output="wbl_example",exmol_dir="../exmol_example/ben/",num_elec_atoms=2)
trans_calc.transmission()
