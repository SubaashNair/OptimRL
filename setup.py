from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
import os
import platform
import shutil
import versioneer


class CustomBuildExt(build_ext):
    """Custom build command to handle library file copying and platform-specific compilation."""

    def build_extensions(self):
        # Set platform-specific compiler flags
        compiler_type = self.compiler.compiler_type
        for ext in self.extensions:
            if compiler_type == 'unix':
                # macOS and Linux specific flags
                if platform.system() == 'Darwin':
                    # macOS specific flags
                    ext.extra_compile_args += ['-O3', '-fPIC']
                    # Support both Intel and Apple Silicon
                    ext.extra_compile_args += ['-arch', 'x86_64', '-arch', 'arm64']
                    ext.extra_link_args += ['-arch', 'x86_64', '-arch', 'arm64']
                else:
                    # Linux specific flags
                    ext.extra_compile_args += ['-O3', '-fPIC']
            elif compiler_type == 'msvc':
                # Windows specific flags
                ext.extra_compile_args += ['/O2']

        # Build the extensions
        super().build_extensions()

        # After building, copy the library to the package directory
        self.copy_extensions_to_package()

    def copy_extensions_to_package(self):
        """Copy the built extension to the package directory."""
        for ext in self.extensions:
            # Get the full path of the built library
            built_lib = self.get_ext_fullpath(ext.name)

            # Determine the destination directory within the package
            dest_dir = os.path.join('optimrl', 'c_src')
            os.makedirs(dest_dir, exist_ok=True)

            # Get the filename only
            filename = os.path.basename(built_lib)

            # Create platform-specific library name
            if platform.system() == 'Darwin':
                lib_name = 'libgrpo.dylib'
            elif platform.system() == 'Linux':
                lib_name = 'libgrpo.so'
            else:
                lib_name = 'libgrpo.dll'

            # Copy the file to the package directory with the correct name
            dest_path = os.path.join(dest_dir, lib_name)
            shutil.copy2(built_lib, dest_path)
            print(f"Copied {built_lib} to {dest_path}")


class BDistWheel(_bdist_wheel):
    """Custom bdist_wheel to ensure platform-specific tagging."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False  # Indicate this is not a pure Python package
        if platform.system() == 'Linux':
            self.plat_name = 'manylinux2014_x86_64'  # Use a compliant platform tag
        elif platform.system() == 'Darwin':
            self.plat_name = 'macosx_11_0_arm64'  # Example for macOS
        elif platform.system() == 'Windows':
            self.plat_name = 'win_amd64'  # Example for Windows


# Define the extension module
grpo_module = Extension(
    'optimrl.c_src.libgrpo',
    sources=['optimrl/c_src/grpo.c'],
    include_dirs=['optimrl/c_src'],
    libraries=['m'] if platform.system() != 'Windows' else [],
    extra_compile_args=['-O3', '-fPIC'] if platform.system() != 'Windows' else ['/O2']
)

# Read the README file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="optimrl",
    version=versioneer.get_version(),
    cmdclass={
        **versioneer.get_cmdclass(),
        'build_ext': CustomBuildExt,
        'bdist_wheel': BDistWheel
    },
    author="Subashanan Nair",
    author_email="subaashnair12@gmail.com",
    description="Group Relative Policy Optimization for Efficient RL Training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SubaashNair/OptimRL",
    packages=find_packages(include=["optimrl", "optimrl.*"]),
    ext_modules=[grpo_module],
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.8.0"
    ],
    extras_require={
        'test': ['pytest>=6.0'],
        'dev': ['pytest>=6.0', 'black', 'isort', 'flake8']
    },
    python_requires=">=3.8",
    include_package_data=True,
    package_data={
        'optimrl': ['c_src/*.h', 'c_src/*.c']
    },
)
