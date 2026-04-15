project('ruqt_python', 'c','fortran')

py_mod = import('python')
py = py_mod.find_installation(pure: false)

# Find NumPy and f2py headers dynamically
incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

# Find Intel MKL (Required by RUQT)
fc = meson.get_compiler('fortran')
mkl_dep = dependency('mkl-static-ilp64-iomp', required: true)
#add_global_arguments('-ffree-line-length-none', language : 'fortran')
#mkl_dep = fc.find_library('mkl-static-ilp64-iomp', required: true) 

# Define Fortran source files IN DEPENDENCY ORDER (Modules first!)
#fortran_srcs = [
  #    'WBL_RUQT/TypeMod.f90',
  #  'WBL_RUQT/Build_B0_CISD.f90',
  #  'WBL_RUQT/Build_G_SD_Invert.f90',
  #]

# Generate the C wrapper for the main subroutine using f2py
#f2py_source = custom_target('f2py_source',
#  input : 'WBL_RUQT/RUQT.f90', # The file with your RUN_RUQT subroutine
#  output : ['wbl_ruqtmodule.c', 'wbl_ruqt-f2pywrappers2.f90'],
#  command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'wbl_ruqt', '--lower']
#)

# Compile everything together into a Python extension
#py.extension_module('wbl_ruqt',
#  fortran_srcs + ['WBL_RUQT/RUQT.f90'],
#  f2py_source,
#  incdir_f2py / 'fortranobject.c',
#  include_directories: [incdir_numpy, incdir_f2py],
#  dependencies : [py.dependency(), mkl_dep],
#  install : true
#)

py.install_sources(
    'src/pyruqt.py', 
    'src/ruqt.py',
)
