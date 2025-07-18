# mypy: ignore-errors
from __future__ import annotations


import copy
import dataclasses
from distutils import ccompiler
from distutils import sysconfig
import logging
import os
import os.path
import pickle
import shutil
import sys

import setuptools

import cupy_builder.install_build as build
from cupy_builder._context import Context
from cupy_builder.install_build import PLATFORM_LINUX
from cupy_builder.install_build import PLATFORM_WIN32


def ensure_module_file(file):
    if isinstance(file, tuple):
        return file
    else:
        return file, []


def module_extension_name(file):
    return ensure_module_file(file)[0]


def module_extension_sources(file, use_cython, no_cuda):
    pyx, others = ensure_module_file(file)
    base = os.path.join(*pyx.split('.'))
    pyx = base + ('.pyx' if use_cython else '.cpp')

    # If CUDA SDK is not available, remove CUDA C files from extension sources
    # and use stubs defined in header files.
    if no_cuda:
        others1 = []
        for source in others:
            base, ext = os.path.splitext(source)
            if ext == '.cu':
                continue
            others1.append(source)
        others = others1

    return [pyx] + others


def get_required_modules(MODULES):
    return [m['name'] for m in MODULES if m.required]


def check_library(compiler, includes=(), libraries=(),
                  include_dirs=(), library_dirs=(), define_macros=None,
                  extra_compile_args=()):

    source = ''.join(['#include <%s>\n' % header for header in includes])
    source += 'int main() {return 0;}'
    try:
        # We need to try to build a shared library because distutils
        # uses different option to build an executable and a shared library.
        # Especially when a user build an executable, distutils does not use
        # LDFLAGS environment variable.
        build.build_shlib(compiler, source, libraries,
                          include_dirs, library_dirs, define_macros,
                          extra_compile_args)
    except Exception as e:
        print(e)
        sys.stdout.flush()
        return False
    return True


def canonicalize_hip_libraries(hip_version, libraries):
    def ensure_tuple(x):
        return x if isinstance(x, tuple) else (x, None)
    new_libraries = []
    for library in libraries:
        lib_name, pred = ensure_tuple(library)
        if pred is None:
            new_libraries.append(lib_name)
        elif pred(hip_version):
            new_libraries.append(lib_name)
    libraries.clear()
    libraries.extend(new_libraries)


def preconfigure_modules(ctx: Context, MODULES, compiler, settings):
    """Returns a list of modules buildable in given environment and settings.

    For each module in MODULES list, this function checks if the module
    can be built in the current environment and reports it.
    Returns a list of module names available.
    """

    nvcc_path = build.get_nvcc_path()
    hipcc_path = build.get_hipcc_path()
    summary = [
        '',
        '************************************************************',
        '* CuPy Configuration Summary                               *',
        '************************************************************',
        '',
        'Build Environment:',
        '  Include directories: {}'.format(str(settings['include_dirs'])),
        '  Library directories: {}'.format(str(settings['library_dirs'])),
        '  nvcc command       : {}'.format(
            nvcc_path if nvcc_path else '(not found)'),
        '  hipcc command      : {}'.format(
            hipcc_path if hipcc_path else '(not found)'),
        '',
        'Environment Variables:',
    ]

    for key in ['CFLAGS', 'LDFLAGS', 'LIBRARY_PATH',
                'CUDA_PATH', 'NVCC', 'HIPCC',
                'ROCM_HOME']:
        summary += ['  {:<16}: {}'.format(key, os.environ.get(key, '(none)'))]

    summary += [
        '',
        'Modules:',
    ]

    ret = []
    for module in MODULES:
        installed = False
        status = 'No'
        errmsg = []

        if module['name'] == 'cutensor':
            cutensor_path = os.environ.get('CUTENSOR_PATH', '')
            inc_path = os.path.join(cutensor_path, 'include')
            if os.path.exists(inc_path):
                settings['include_dirs'].append(inc_path)
            lib_path = os.path.join(cutensor_path, 'lib')
            if os.path.exists(lib_path):
                settings['library_dirs'].append(lib_path)
            cuda_version = ctx.features['cuda'].get_version()
            cuda_major = str(cuda_version // 1000)
            cuda_major_minor = cuda_major + '.' + \
                str((cuda_version // 10) % 100)
            for cuda_ver in (cuda_major_minor, cuda_major):
                lib_path = os.path.join(cutensor_path, 'lib', cuda_ver)
                if os.path.exists(lib_path):
                    settings['library_dirs'].append(lib_path)
                    break

        # In ROCm 4.1 and later, we need to use the independent version of
        # hipfft as well as rocfft. We configure the lists of include
        # directories and libraries to link here depending on ROCm version
        # before the configuration process following.
        if ctx.use_hip and module['name'] == 'cuda':
            if module.configure(compiler, settings):
                hip_version = module.get_version()
                if hip_version >= 401:
                    rocm_path = build.get_rocm_path()
                    inc_path = os.path.join(rocm_path, 'hipfft', 'include')
                    settings['include_dirs'].insert(0, inc_path)
                    lib_path = os.path.join(rocm_path, 'hipfft', 'lib')
                    settings['library_dirs'].insert(0, lib_path)
                # n.b., this modifieds MODULES['cuda']['libraries'] inplace
                canonicalize_hip_libraries(hip_version, module['libraries'])

        print('')
        print('-------- Configuring Module: {} --------'.format(
            module['name']))
        sys.stdout.flush()
        if not check_library(
                compiler,
                includes=module['include'],
                include_dirs=settings['include_dirs'],
                define_macros=settings['define_macros'],
                extra_compile_args=settings['extra_compile_args']):
            errmsg = ['Include files not found: %s' % module['include'],
                      'Check your CFLAGS environment variable.']
        elif not check_library(
                compiler,
                libraries=module['libraries'],
                library_dirs=settings['library_dirs'],
                define_macros=settings['define_macros'],
                extra_compile_args=settings['extra_compile_args']):
            errmsg = ['Cannot link libraries: %s' % module['libraries'],
                      'Check your LDFLAGS environment variable.']
        elif not module.configure(compiler, settings):
            # Fail on per-library condition check (version requirements etc.)
            installed = True
            errmsg = ['The library is installed but not supported.']
        elif (module['name'] in ('thrust', 'cub', 'random')
                and (nvcc_path is None and hipcc_path is None)):
            installed = True
            cmd = 'nvcc' if not ctx.use_hip else 'hipcc'
            errmsg = ['{} command could not be found in PATH.'.format(cmd),
                      'Check your PATH environment variable.']
        else:
            installed = True
            status = 'Yes'
            ret.append(module['name'])

        if installed:
            version = module.get_version()
            if version is not None:
                status += f' (version {version})'

        summary += [
            '  {:<10}: {}'.format(module['name'], status)
        ]

        # If error message exists...
        if len(errmsg) != 0:
            summary += ['    -> {}'.format(m) for m in errmsg]

            # Skip checking other modules when CUDA is unavailable.
            if module['name'] == 'cuda':
                break

    # Get a list of the CC of the devices connected to this node
    if not ctx.use_hip:
        build.check_compute_capabilities(compiler, settings)

    if len(ret) != len(MODULES):
        if 'cuda' in ret:
            lines = [
                'WARNING: Some modules could not be configured.',
                'CuPy will be installed without these modules.',
            ]
        else:
            lines = [
                'ERROR: CUDA could not be found on your system.',
                '',
                'HINT: You are trying to build CuPy from source, '
                'which is NOT recommended for general use.',
                '      Please consider using binary packages instead.',
                '',

            ]
        summary += [
            '',
        ] + lines + [
            'Please refer to the Installation Guide for details:',
            'https://docs.cupy.dev/en/stable/install.html',
            '',
        ]

    summary += [
        '************************************************************',
        '',
    ]

    print('\n'.join(summary))
    return ret, settings


def _rpath_base():
    if PLATFORM_LINUX:
        return '$ORIGIN'
    else:
        raise Exception('not supported on this platform')


def _find_static_library(name: str) -> str:
    if PLATFORM_LINUX:
        filename = f'lib{name}.a'
        if (int(os.environ.get('CONDA_BUILD_CROSS_COMPILATION', 0)) == 1 and
                os.environ.get('CONDA_OVERRIDE_CUDA', '0').startswith('11')):
            # CUDA 11 on conda-forge has an ad hoc layout to support cross
            # compiling
            libdirs = ['lib']
            cuda_path = (f'{build.get_cuda_path()}/targets/'
                         f'{build.conda_get_target_name()}/')
        else:
            libdirs = ['lib64', 'lib']
            cuda_path = build.get_cuda_path()
    elif PLATFORM_WIN32:
        filename = f'{name}.lib'
        libdirs = ['lib\\x64', 'lib']
        cuda_path = build.get_cuda_path()
    else:
        raise Exception('not supported on this platform')

    logging.debug(f"{cuda_path=}")
    if cuda_path is None:
        raise Exception(f'Could not find {filename}: CUDA path unavailable')
    for libdir in libdirs:
        path = os.path.join(cuda_path, libdir, filename)
        if os.path.exists(path):
            return path
    else:
        raise Exception(f'Could not find {filename}: {path} does not exist')


def make_extensions(ctx: Context, compiler, use_cython):
    """Produce a list of Extension instances which passed to cythonize()."""

    ctx.calculate_cupy_cache_key()
    CACHE_FILE = f"{ctx.source_root}/.cupy_builder.cache"
    if ctx.dev_configure_cache and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            (prev_ctx, ret) = pickle.load(f)
        if (ctx.dev_configure_cache_key == prev_ctx.dev_configure_cache_key and
                ctx.cupy_cache_key == prev_ctx.cupy_cache_key):
            print("***************************************************")
            print("*** NOTICE: Reusing build configuration from previous "
                  f"run. Remove the configuration cache ({CACHE_FILE}) "
                  "if you intend to reconfigure.")
            print("***************************************************")
            for f in dataclasses.fields(prev_ctx):
                setattr(ctx, f.name, getattr(prev_ctx, f.name))
            return ret
        print("*** NOTICE: Cache key has changed, ignoring config cache.")

    MODULES = ctx.features.values()

    no_cuda = ctx.use_stub
    use_hip = not no_cuda and ctx.use_hip
    settings = build.get_compiler_setting(ctx, use_hip)

    include_dirs = settings['include_dirs']

    settings['include_dirs'] = [
        x for x in include_dirs if os.path.exists(x)]
    settings['library_dirs'] = [
        x for x in settings['library_dirs'] if os.path.exists(x)]

    # Adjust rpath to use CUDA libraries in `cupy/.data/lib/*.so`) from CuPy.
    use_wheel_libs_rpath = (
        0 < len(ctx.wheel_libs) and not PLATFORM_WIN32)

    # In the environment with CUDA 7.5 on Ubuntu 16.04, gcc5.3 does not
    # automatically deal with memcpy because string.h header file has
    # been changed. This is a workaround for that environment.
    # See details in the below discussions:
    # https://github.com/BVLC/caffe/issues/4046
    # https://groups.google.com/forum/#!topic/theano-users/3ihQYiTRG4E
    settings['define_macros'].append(('_FORCE_INLINES', '1'))

    # Ensure all "cdef public" APIs have C linkage.
    settings['define_macros'].append(('CYTHON_EXTERN_C', 'extern "C"'))

    if ctx.linetrace:
        settings['define_macros'].append(('CYTHON_TRACE', '1'))
        settings['define_macros'].append(('CYTHON_TRACE_NOGIL', '1'))
    if no_cuda:
        settings['define_macros'].append(('CUPY_NO_CUDA', '1'))
    if use_hip:
        settings['define_macros'].append(('CUPY_USE_HIP', '1'))
        # introduced since ROCm 4.2.0
        settings['define_macros'].append(('__HIP_PLATFORM_AMD__', '1'))
        # deprecated since ROCm 4.2.0
        settings['define_macros'].append(('__HIP_PLATFORM_HCC__', '1'))
        # Fix for ROCm 6.3.0, See https://github.com/ROCm/rocThrust/issues/502
        settings['define_macros'].append(
            ('THRUST_DEVICE_SYSTEM', 'THRUST_DEVICE_SYSTEM_HIP'))
    settings['define_macros'].append(('CUPY_CACHE_KEY', ctx.cupy_cache_key))

    try:
        host_compiler = compiler
        if int(os.environ.get('CONDA_BUILD_CROSS_COMPILATION', 0)) == 1:
            os.symlink(f'{os.environ["BUILD_PREFIX"]}/x86_64-conda-linux-gnu/'
                       'bin/x86_64-conda-linux-gnu-ld',
                       f'{os.environ["BUILD_PREFIX"]}/bin/ld')
        if (PLATFORM_LINUX and (
                int(os.environ.get('CONDA_BUILD_CROSS_COMPILATION', 0)) == 1 or
                os.environ.get('CONDA_OVERRIDE_CUDA', '0').startswith('12'))):
            # If cross-compiling, we need build_and_run() & build_shlib() to
            # use the compiler on the build platform to generate stub files
            # that are executable in the build environment, not the target
            # environment.
            compiler = ccompiler.new_compiler()
            sysconfig.customize_compiler(compiler)
            # Need to match and replace these
            # https://github.com/pypa/distutils/blob/30b7331b07fbc404959cb37ac311afdfb90813be/distutils/unixccompiler.py#L117-L129
            cc = os.environ['CC_FOR_BUILD']
            cxx = os.environ['CXX_FOR_BUILD']
            ar = os.environ['BUILD'] + "-ar"
            compiler.preprocessor = None
            compiler.compiler = [cc,]
            compiler.compiler_so = [cc,]
            compiler.compiler_cxx = [cxx,]
            compiler.compiler_so_cxx = [cxx,]
            compiler.linker_so = [cc, f'-B{os.environ["BUILD_PREFIX"]}/bin',
                                  '-shared']
            compiler.linker_so_cxx = [cxx,
                                      f'-B{os.environ["BUILD_PREFIX"]}/bin',
                                      '-shared']
            compiler.linker_exe = [cc, f'-B{os.environ["BUILD_PREFIX"]}/bin']
            compiler.linker_exe_cxx = [cxx,
                                       f'-B{os.environ["BUILD_PREFIX"]}/bin']
            compiler.archiver = [ar, 'rcs']
            compiler.ranlib = None

        available_modules = []
        if no_cuda:
            available_modules = [m['name'] for m in MODULES]
        else:
            available_modules, settings = preconfigure_modules(
                ctx, MODULES, compiler, settings)
            required_modules = get_required_modules(MODULES)
            if not (set(required_modules) <= set(available_modules)):
                raise Exception('Your CUDA environment is invalid. '
                                'Please check above error log.')
    finally:
        compiler = host_compiler
        if int(os.environ.get('CONDA_BUILD_CROSS_COMPILATION', 0)) == 1:
            os.remove(f'{os.environ["BUILD_PREFIX"]}/bin/ld')

    ret = []
    for module in MODULES:
        if module['name'] not in available_modules:
            continue

        s = copy.deepcopy(settings)
        if not no_cuda:
            s['libraries'] = module.libraries
            s['extra_objects'] = [
                _find_static_library(name) for name in module.static_libraries
            ]

        compile_args = s.setdefault('extra_compile_args', [])
        link_args = s.setdefault('extra_link_args', [])

        if module['name'] == 'cusolver':
            # cupy_backends/cupy_lapack.h has C++ template code
            compile_args.append('--std=c++11')
            # openmp is required for cusolver
            if use_hip:
                pass
            elif compiler.compiler_type == 'unix':
                compile_args.append('-fopenmp')
                link_args.append('-fopenmp')
            elif compiler.compiler_type == 'msvc':
                compile_args.append('/openmp')

        if module['name'] == 'random':
            if compiler.compiler_type == 'msvc':
                compile_args.append('-D_USE_MATH_DEFINES')

        if module['name'] == 'jitify':
            # this fixes RTD (no_cuda) builds...
            compile_args.append('--std=c++11')
            # suppress printing Jitify logging to stdout
            compile_args.append('-DJITIFY_PRINT_LOG=0')
            # Uncomment to diagnose Jitify issues.
            # compile_args.append('-DJITIFY_PRINT_ALL')

            # if any change is made to the Jitify header, we force recompiling
            s['depends'] = ['./cupy/_core/include/cupy/_jitify/jitify.hpp']

        if module['name'] == 'dlpack':
            # if any change is made to the DLPack header, we force recompiling
            s['depends'] = ['./cupy/_core/include/cupy/_dlpack/dlpack.h']

        for f in module['file']:
            s_file = copy.deepcopy(s)
            name = module_extension_name(f)

            if name.endswith('fft._callback') and not PLATFORM_LINUX:
                continue

            rpath = []
            if not ctx.no_rpath:
                # Add library directories (e.g., `/usr/local/cuda/lib64`) to
                # RPATH.
                rpath += s_file['library_dirs']

            if use_wheel_libs_rpath:
                # Add `cupy/.data/lib` (where shared libraries included in
                # wheels reside) to RPATH.
                # The path is resolved relative to the module, e.g., use
                # `$ORIGIN/../cupy/.data/lib` for `cupy/cudnn.so` and
                # `$ORIGIN/../../../cupy/.data/lib` for
                # `cupy_backends/cuda/libs/cudnn.so`.
                depth = name.count('.')
                rpath.append(
                    '{}{}/cupy/.data/lib'.format(_rpath_base(), '/..' * depth))

            if (PLATFORM_LINUX and len(rpath) != 0):
                ldflag = '-Wl,'
                if PLATFORM_LINUX:
                    ldflag += '--disable-new-dtags,'
                ldflag += ','.join('-rpath,' + p for p in rpath)
                args = s_file.setdefault('extra_link_args', [])
                args.append(ldflag)

            sources = module_extension_sources(f, use_cython, no_cuda)
            extension = setuptools.Extension(name, sources, **s_file)
            ret.append(extension)

    if ctx.dev_configure_cache:
        print(f"Persisting build configuration cache: {CACHE_FILE}")
        with open(CACHE_FILE, "wb") as f:
            pickle.dump((ctx, ret), f)
    return ret


def prepare_wheel_libs(ctx: Context):
    """Prepare shared libraries and include files for wheels.

    Shared libraries are placed under `cupy/.data/lib` and
    RUNPATH will be set to this directory later (Linux only).
    Include files are placed under `cupy/.data/include`.

    Returns the list of files (path relative to `cupy` module) to add to
    the sdist/wheel distribution.
    """
    data_dir = os.path.abspath(os.path.join('cupy', '.data'))
    if os.path.exists(data_dir):
        print('Clearing directory: {}'.format(data_dir))
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)

    # Generate list files to copy
    # tuple of (src_path, dst_path)
    files_to_copy = []

    # Library files
    for srcpath in ctx.wheel_libs:
        relpath = os.path.basename(srcpath)
        dstpath = os.path.join(data_dir, 'lib', relpath)
        files_to_copy.append((srcpath, dstpath))

    # Include files
    for include_path_spec in ctx.wheel_includes:
        srcpath, relpath = include_path_spec.rsplit(':', 1)
        dstpath = os.path.join(data_dir, 'include', relpath)
        files_to_copy.append((srcpath, dstpath))

    # Wheel meta data
    if ctx.wheel_metadata_path:
        files_to_copy.append(
            (ctx.wheel_metadata_path, os.path.join(data_dir, '_wheel.json')))

    # Copy
    for srcpath, dstpath in files_to_copy:
        # Note: symlink is resolved by shutil.copy2.
        print('Copying file for wheel: {}'.format(srcpath))
        dirpath = os.path.dirname(dstpath)
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
        shutil.copy2(srcpath, dstpath)

    package_files = [x[1] for x in files_to_copy] + [
        'cupy/.data/_depends.json',
    ]
    return [os.path.relpath(f, 'cupy') for f in package_files]


def get_ext_modules(use_cython: bool, ctx: Context):
    # We need to call get_config_vars to initialize _config_vars in distutils
    # see #1849
    sysconfig.get_config_vars()
    compiler = ccompiler.new_compiler()
    sysconfig.customize_compiler(compiler)

    extensions = make_extensions(ctx, compiler, use_cython)

    return extensions
