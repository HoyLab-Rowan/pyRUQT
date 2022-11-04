import ruqt
import pyruqt

trans_calc=pyruqt.sie_negf(output="sie_example",exmol_dir="../exmol_example/ben/",elec_dir="../elec_example/au_elec/")
trans_calc.current()
