project('ruqt_python', 'c','fortran')

py_mod = import('python')
py = py_mod.find_installation(pure: false)

# Find NumPy and f2py headers dynamically
incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

# Find Intel MKL (Required by RUQT)
fc = meson.get_compiler('fortran')
lapack_dep = dependency('lapack', required: true)
blas_dep = dependency('blas', required: true)
#mkl_dep = dependency('mkl-static-lp64-iomp', required: true)
add_global_arguments('-ffree-line-length-none', language : 'fortran')

#This next part is only for debugging the fortran libraries (if you are rebuilding them yourself)
if get_option('buildtype') == 'debug'
  add_project_arguments(
    '-fcheck=all', 
    '-fbacktrace', 
    '-O0',            # Turn off optimization so variables aren't optimized away
    language: 'fortran'
  )
  add_project_arguments(
    '-g3',
    '-O0',
    language: 'c'
  )
endif
# Define Fortran source files IN DEPENDENCY ORDER (Modules first!)
fortran_srcs = [
    'WBL_RUQT/TypeMod.f90',
    'WBL_RUQT/RUQT_IntRoutines.f90',
    'WBL_RUQT/RUQT.f90',
  ]

# Generate the C wrapper for the main subroutine using f2py
f2py_source = custom_target('f2py_source',
  input : 'WBL_RUQT/RUQT.f90' , # The file with your RUN_RUQT subroutine
  output : ['wbl_ruqtmodule.c', 'wbl_ruqt-f2pywrappers2.f90'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'wbl_ruqt', '--lower']#,'only:','wbl_negf_calc',':']
)

# Compile everything together into a Python extension
py.extension_module('wbl_ruqt',
  fortran_srcs ,
  f2py_source,
  incdir_f2py / 'fortranobject.c',
  include_directories: [incdir_numpy, incdir_f2py],
  dependencies : [py.dependency(), lapack_dep, blas_dep],
  install : true,
  subdir: 'pyruqt'
)

py.install_sources(
    'pyruqt/__init__.py',
    'pyruqt/pyruqt.py', 
    'pyruqt/ruqt.py',
    subdir: 'pyruqt'
)
