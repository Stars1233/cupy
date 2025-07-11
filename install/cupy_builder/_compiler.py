from __future__ import annotations

import distutils.ccompiler
import os
import os.path
import platform
import shutil
import sys
import subprocess
from typing import Any

from setuptools import Extension

from cupy_builder._context import Context
import cupy_builder.install_build as build


def _nvcc_gencode_options(cuda_version: int) -> list[str]:
    """Returns NVCC GPU code generation options."""

    if sys.argv == ['setup.py', 'develop']:
        return []

    envcfg = os.getenv('CUPY_NVCC_GENERATE_CODE', None)
    if envcfg is not None and envcfg != 'current':
        return ['--generate-code={}'.format(arch)
                for arch in envcfg.split(';') if len(arch) > 0]
    if envcfg == 'current' and build.get_compute_capabilities() is not None:
        ccs = build.get_compute_capabilities()
        arch_list = [
            f'compute_{cc}' if cc < 60 else (f'compute_{cc}', f'sm_{cc}')
            for cc in ccs]
    else:
        # The arch_list specifies virtual architectures, such as 'compute_61',
        # and real architectures, such as 'sm_61', for which the CUDA
        # input files are to be compiled.
        #
        # The syntax of an entry of the list is
        #
        #     entry ::= virtual_arch | (virtual_arch, real_arch)
        #
        # where virtual_arch is a string which means a virtual architecture and
        # real_arch is a string which means a real architecture.
        #
        # If a virtual architecture is supplied, NVCC generates a PTX code
        # the virtual architecture. If a pair of a virtual architecture and a
        # real architecture is supplied, NVCC generates a PTX code for the
        # virtual architecture as well as a cubin code for the real one.
        #
        # For example, making NVCC generate a PTX code for 'compute_60' virtual
        # architecture, the arch_list has an entry of 'compute_60'.
        #
        #     arch_list = ['compute_60']
        #
        # For another, making NVCC generate a PTX code for 'compute_61' virtual
        # architecture and a cubin code for 'sm_61' real architecture, the
        # arch_list has an entry of ('compute_61', 'sm_61').
        #
        #     arch_list = [('compute_61', 'sm_61')]
        #
        # See the documentation of each CUDA version for the list of supported
        # architectures:
        #
        #   https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation
        #
        # CuPy utilizes CUDA Minor Version Compatibility to support all CUDA
        # minor versions in a single binary package (e.g., `cupy-cuda12x`). To
        # achieve this, CUBIN must be generated for all supported compute
        # capabilities instead of PTX. This is because executing PTX requires
        # CUDA driver newer than the one used to compile the code, and we often
        # use the latest CUDA Driver to build our binary package. See also:
        #
        #   https://docs.nvidia.com/deploy/cuda-compatibility/index.html#application-considerations-for-minor-version-compatibility
        #
        # In addition, to allow running CuPy with future (not yet released)
        # GPUs, PTX for the latest architecture is also included as a
        # fallback. c.f.:
        #
        #   https://forums.developer.nvidia.com/t/software-migration-guide-for-nvidia-blackwell-rtx-gpus-a-guide-to-cuda-12-8-pytorch-tensorrt-and-llama-cpp/321330
        #
        # Jetson platforms are also targetted when built under aarch64. c.f.:
        #
        #   https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#deployment-considerations-for-cuda-upgrade-package

        aarch64 = (platform.machine() == 'aarch64')
        if cuda_version >= 12000:
            arch_list = [('compute_50', 'sm_50'),
                         ('compute_52', 'sm_52'),
                         ('compute_60', 'sm_60'),
                         ('compute_61', 'sm_61'),
                         ('compute_70', 'sm_70'),
                         ('compute_75', 'sm_75'),
                         ('compute_80', 'sm_80'),
                         ('compute_86', 'sm_86'),
                         ('compute_89', 'sm_89'),
                         ('compute_90', 'sm_90'),]
            if cuda_version < 12080:
                arch_list.append('compute_90')
            elif 12080 <= cuda_version < 12090:
                arch_list += [('compute_100', 'sm_100'),
                              ('compute_120', 'sm_120'),
                              'compute_100']
            elif 12090 <= cuda_version:
                arch_list += [('compute_100f', 'sm_100'),
                              ('compute_120f', 'sm_120'),
                              'compute_100']

            if aarch64:
                # JetPack 5 (CUDA 12.0-12.2) or JetPack 6 (CUDA 12.2+)
                arch_list += [
                    ('compute_72', 'sm_72'),  # Jetson (Xavier)
                    ('compute_87', 'sm_87'),  # Jetson (Orin)
                ]
        elif cuda_version >= 11080:
            arch_list = [('compute_35', 'sm_35'),
                         ('compute_37', 'sm_37'),
                         ('compute_50', 'sm_50'),
                         ('compute_52', 'sm_52'),
                         ('compute_60', 'sm_60'),
                         ('compute_61', 'sm_61'),
                         ('compute_70', 'sm_70'),
                         ('compute_75', 'sm_75'),
                         ('compute_80', 'sm_80'),
                         ('compute_86', 'sm_86'),
                         ('compute_89', 'sm_89'),
                         ('compute_90', 'sm_90'),
                         'compute_90']
            if aarch64:
                # JetPack 5 (CUDA 11.4/11.8)
                arch_list += [
                    ('compute_72', 'sm_72'),  # Jetson (Xavier)
                    ('compute_87', 'sm_87'),  # Jetson (Orin)
                ]
        elif cuda_version >= 11040:
            arch_list = [('compute_35', 'sm_35'),
                         ('compute_37', 'sm_37'),
                         ('compute_50', 'sm_50'),
                         ('compute_52', 'sm_52'),
                         ('compute_60', 'sm_60'),
                         ('compute_61', 'sm_61'),
                         ('compute_70', 'sm_70'),
                         ('compute_75', 'sm_75'),
                         ('compute_80', 'sm_80'),
                         ('compute_86', 'sm_86'),
                         'compute_86']
            if aarch64:
                # JetPack 5 (CUDA 11.4/11.8)
                arch_list += [
                    ('compute_72', 'sm_72'),  # Jetson (Xavier)
                    ('compute_87', 'sm_87'),  # Jetson (Orin)
                ]
        elif cuda_version >= 11020:
            arch_list = ['compute_35',
                         'compute_50',
                         ('compute_60', 'sm_60'),
                         ('compute_61', 'sm_61'),
                         ('compute_70', 'sm_70'),
                         ('compute_75', 'sm_75'),
                         ('compute_80', 'sm_80'),
                         ('compute_86', 'sm_86'),
                         'compute_86']
        else:
            # This should not happen.
            assert False

    options = []
    for arch in arch_list:
        if type(arch) is tuple:
            virtual_arch, real_arch = arch
            options.append('--generate-code=arch={},code={}'.format(
                virtual_arch, real_arch))
        else:
            options.append('--generate-code=arch={},code={}'.format(
                arch, arch))

    return options


class DeviceCompilerBase:
    """A class that invokes NVCC or HIPCC."""
    _context: Context

    def __init__(self, ctx: Context) -> None:
        self._context = ctx

    def _get_preprocess_options(self, ext: Extension) -> list[str]:
        # https://setuptools.pypa.io/en/latest/deprecated/distutils/apiref.html#distutils.core.Extension
        # https://github.com/pypa/setuptools/blob/v60.0.0/setuptools/_distutils/command/build_ext.py#L524-L526
        incdirs = ext.include_dirs[:]
        macros: list[Any] = ext.define_macros[:]
        for undef in ext.undef_macros:
            macros.append((undef,))
        return distutils.ccompiler.gen_preprocess_options(macros, incdirs)

    def spawn(self, commands: list[str]) -> None:
        print('Command:', commands)
        subprocess.check_call(commands)


class DeviceCompilerUnix(DeviceCompilerBase):

    def compile(self, obj: str, src: str, ext: Extension) -> None:
        if self._context.use_hip:
            self._compile_unix_hipcc(obj, src, ext)
        else:
            self._compile_unix_nvcc(obj, src, ext)

    def _compile_unix_nvcc(self, obj: str, src: str, ext: Extension) -> None:
        cc_args = self._get_preprocess_options(ext) + ['-c']

        # For CUDA C source files, compile them with NVCC.
        nvcc_path = build.get_nvcc_path()
        base_opts = build.get_compiler_base_options(nvcc_path)
        compiler_so = nvcc_path

        cuda_version = self._context.features['cuda'].get_version()
        postargs = _nvcc_gencode_options(cuda_version) + [
            '-Xfatbin=-compress-all', '-O2', '--compiler-options="-fPIC"',
            '--expt-relaxed-constexpr']
        num_threads = int(os.environ.get('CUPY_NUM_NVCC_THREADS', '2'))
        # Note: we only support CUDA 11.2+ since CuPy v13.0.0.
        # Bumping C++ standard from C++14 to C++17 for "if constexpr"
        postargs += ['--std=c++17',
                     f'-t{num_threads}',
                     '-Xcompiler=-fno-gnu-unique']
        print('NVCC options:', postargs)
        self.spawn(compiler_so + base_opts + cc_args + [src, '-o', obj] +
                   postargs)

    def _compile_unix_hipcc(self, obj: str, src: str, ext: Extension) -> None:
        cc_args = self._get_preprocess_options(ext) + ['-c']

        # For CUDA C source files, compile them with HIPCC.
        rocm_path = build.get_hipcc_path()
        base_opts = build.get_compiler_base_options(rocm_path)
        compiler_so = rocm_path

        postargs = ['-O2', '-fPIC', '--include', 'hip_runtime.h']
        # Note: we only support ROCm 4.3+ since CuPy v11.0.0.
        # Bumping C++ standard from C++14 to C++17 for "if constexpr"
        postargs += ['--std=c++17']
        print('HIPCC options:', postargs)
        self.spawn(compiler_so + base_opts + cc_args + [src, '-o', obj] +
                   postargs)


class DeviceCompilerWin32(DeviceCompilerBase):

    def compile(self, obj: str, src: str, ext: Extension) -> None:
        if self._context.use_hip:
            raise RuntimeError('ROCm is not supported on Windows')

        compiler_so = build.get_nvcc_path()
        cc_args = self._get_preprocess_options(ext) + ['-c']
        cuda_version = self._context.features['cuda'].get_version()
        postargs = _nvcc_gencode_options(cuda_version) + [
            '-Xfatbin=-compress-all', '-O2']
        # Note: we only support CUDA 11.2+ since CuPy v13.0.0.
        # MSVC 14.0 (2015) is deprecated for CUDA 11.2 but we need it
        # to build CuPy because some Python versions were built using it.
        # REF: https://wiki.python.org/moin/WindowsCompilers
        postargs += ['-allow-unsupported-compiler']
        # "/bigobj" to silence `fatal error C1128: number of sections exceeded
        # object file format limit`
        postargs += ['-Xcompiler', '/MD /bigobj', '-D_USE_MATH_DEFINES']
        # Bumping C++ standard from C++14 to C++17 for "if constexpr"
        num_threads = int(os.environ.get('CUPY_NUM_NVCC_THREADS', '2'))
        postargs += ['--std=c++17',
                     f'-t{num_threads}']
        cl_exe_path = self._find_host_compiler_path()
        if cl_exe_path is not None:
            print(f'Using host compiler at {cl_exe_path}')
            postargs += ['--compiler-bindir', cl_exe_path]
        print('NVCC options:', postargs)
        self.spawn(compiler_so + cc_args + [src, '-o', obj] + postargs)

    def _find_host_compiler_path(self) -> str | None:
        # c.f. cupy.cuda.compiler._get_extra_path_for_msvc
        cl_exe = shutil.which('cl.exe')
        if cl_exe:
            # The compiler is already on PATH, no extra path needed.
            return None

        if self._context.win32_cl_exe_path is not None:
            return self._context.win32_cl_exe_path

        try:
            # See #8568, #8574, #8583.
            import setuptools.msvc
        except Exception:
            print('Warning: cl.exe could not be auto-detected; '
                  'setuptools.msvc could not be imported')
            return None

        vctools: list[str] = setuptools.msvc.EnvironmentInfo(
            platform.machine()).VCTools
        for path in vctools:
            cl_exe = os.path.join(path, 'cl.exe')
            if os.path.exists(cl_exe):
                return path
        print(f'Warning: cl.exe could not be found in {vctools}')
        return None
