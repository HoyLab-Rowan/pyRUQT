import ruqt
import pyruqt

trans_calc=pyruqt.sie_negf(output="stb_example",exmol_dir="../exmol_example/ben/",elec_dir="../elec_example/au_elec/")
#print(trans_calc.input_parameters['exmol_dir'])
trans_calc.transmission()
