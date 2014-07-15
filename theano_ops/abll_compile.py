import errno
import logging
import os
import shutil
import stat
import sys

from theano import config
from theano.gof.cmodule import get_lib_extension
from theano.gof.compilelock import get_lock, release_lock
from theano.sandbox import cuda
from theano.sandbox.cuda import nvcc_compiler
import theano_ops

_logger_name = 'deeplearn.theano_ops.abll_compile'
_logger = logging.getLogger(_logger_name)
_logger.debug('importing')

root_dir = os.path.join(theano_ops.__path__[0], 'abll')
include_dir = os.path.join(root_dir, 'include')
src_dir = os.path.join(root_dir, 'src')
loc = os.path.join(config.compiledir, 'abll')
libs = [
    'cublas',
    'curand',
    'cufft',
]
srcs = (
    # In partial dependency order: the last ones depend on the first ones
    'abll/conv_bc01_fft.cu',
)

abll_so = os.path.join(loc, 'abll.' + get_lib_extension())
libabll_so = os.path.join(loc, 'libabll.' + get_lib_extension())


def is_available():
    # If already compiled, OK
    if is_available.compiled:
        _logger.debug('already compiled')
        return True

    # If there was an error, do not try again
    if is_available.compile_error:
        _logger.debug('error last time')
        return False

    # Else, we need CUDA
    if not cuda.cuda_available:
        is_available.compile_error = True
        _logger.debug('cuda unavailable')
        return False

    # Try to actually compile
    success = abll_compile()
    if success:
        is_available.compiled = True
    else:
        is_available.compile_error = False
    _logger.debug('compilation success: %s', success)

    return is_available.compiled

# Initialize variables in is_available
is_available.compiled = False
is_available.compile_error = False


def should_recompile():
    """
    Returns True if the .so files are not present or outdated.
    """
    # The following list is in alphabetical order.
    mtimes = [os.stat(os.path.join(src_dir, source_file))[stat.ST_MTIME]
              for source_file in srcs]
    date = max(mtimes)
    _logger.debug('max date: %f', date)
    if (not os.path.exists(abll_so) or
            date >= os.stat(abll_so)[stat.ST_MTIME]):
        return True
    return False


def symlink_ok():
    """
    Check if an existing library exists and can be read.
    """
    try:
        open(libabll_so).close()
        return True
    except IOError:
        return False


def abll_compile():
    # Compile .cu files in abll
    _logger.debug('nvcc_compiler.rpath_defaults: %s',
                  str(nvcc_compiler.rpath_defaults))
    import time
    t1 = time.time()
    if should_recompile():
        _logger.debug('should recompile')

        # Concatenate all .cu files into one big mod.cu
        code = []
        for source_file in srcs:
            code.append(open(os.path.join(src_dir, source_file)).read())
        code = '\n'.join(code)

        get_lock()
        try:
            # Check if the compilation has already been done by another process
            # while we were waiting for the lock
            if should_recompile():
                _logger.debug('recompiling')

                try:
                    compiler = nvcc_compiler.NVCC_compiler()
                    args = compiler.compile_args()

                    # compiler.compile_args() can execute a
                    # compilation This currently will remove empty
                    # directory in the compile dir.  So we must make
                    # destination directory after calling it.
                    if not os.path.exists(loc):
                        os.makedirs(loc)
                    compiler.compile_str(
                        'abll',
                        code,
                        location=loc,
                        include_dirs=[include_dir],
                        lib_dirs=nvcc_compiler.rpath_defaults + [loc],
                        libs=libs,
                        preargs=['-O3'] + args,
                        py_module=False,)
                except Exception as e:
                    _logger.error('Failed to compile %s %s: %s',
                                  os.path.join(loc, 'mod.cu'),
                                  srcs, str(e))
                    return False
            else:
                _logger.debug('already compiled by another process')

        finally:
            release_lock()
    else:
        _logger.debug('not recompiling')

    # If necessary, create a symlink called libabll.so
    if not symlink_ok():
        if sys.platform == 'win32':
            # The Python `os` module does not support symlinks on win32.
            shutil.copyfile(abll_so, libabll_so)
        else:
            try:
                os.symlink(abll_so, libabll_so)
            except OSError as e:
                # This may happen for instance when running multiple
                # concurrent jobs, if two of them try to create the
                # symlink simultaneously.
                # If that happens, we verify that the existing symlink is
                # indeed working.
                if (getattr(e, 'errno', None) != errno.EEXIST
                        or not symlink_ok()):
                    raise

    # Raise an error if libabll_so is still not available
    open(libabll_so).close()

    # Add abll to the list of places that are hard-coded into
    # compiled modules' runtime library search list.
    nvcc_compiler.add_standard_rpath(loc)

    t2 = time.time()
    _logger.debug('successfully imported. Compiled in %fs', t2 - t1)

    return True
