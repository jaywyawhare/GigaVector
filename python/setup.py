import shutil
import subprocess
from pathlib import Path
import os
import sys
import glob

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
from setuptools.command.sdist import sdist as _sdist

# Directory inside the python/ tree where C sources are bundled for sdist builds.
_CSRC_DIR = Path(__file__).resolve().parent / "_csrc"


def _find_repo_root() -> Path:
    """Locate the directory containing CMakeLists.txt.

    Handles two layouts:
      - git checkout: setup.py lives in <repo>/python/, CMakeLists.txt is at <repo>/
      - sdist:        C sources are bundled into python/_csrc/ by the custom sdist command
    """
    setup_dir = Path(__file__).resolve().parent
    for candidate in [setup_dir.parent, _CSRC_DIR]:
        if (candidate / "CMakeLists.txt").exists():
            return candidate
    raise FileNotFoundError(
        "CMakeLists.txt not found. When building on Windows from an sdist, "
        "the sdist must have been created from a full git checkout so that C "
        "sources are bundled inside _csrc/."
    )


def _cmake_generator(env: dict) -> str:
    """Return the CMake generator to use on Windows, or '' to let CMake decide.

    Prefer MSVC (Visual Studio) over MinGW to avoid shipping MinGW runtime DLLs
    (libgcc_s_seh-1.dll, libwinpthread-1.dll, etc.) alongside the package.
    Users can always override via the CMAKE_GENERATOR environment variable.
    """
    forced = env.get("CMAKE_GENERATOR", "")
    if forced:
        return forced

    # If MSVC cl.exe is on PATH (or a VS dev-prompt set VCINSTALLDIR /
    # VCToolsInstallDir), let CMake auto-select the Visual Studio generator.
    if shutil.which("cl") or env.get("VCINSTALLDIR") or env.get("VCToolsInstallDir"):
        return ""  # CMake will pick the VS generator automatically

    # No MSVC found — fall back to MinGW Makefiles if gcc is available.
    if shutil.which("gcc") or shutil.which("mingw32-make"):
        return "MinGW Makefiles"

    return ""  # let CMake figure it out


class BuildPyWithMake(build_py):
    """Run `make lib` in the repository root and copy the .so into the package."""

    def run(self):
        pkg_dir = Path(__file__).resolve().parent / "src" / "gigavector"
        if os.name == "nt":
            lib_filename = "GigaVector.dll"
        elif sys.platform == "darwin":
            lib_filename = "libGigaVector.dylib"
        else:
            lib_filename = "libGigaVector.so"

        package_lib_path = pkg_dir / lib_filename

        # Prefer an already-packaged library to avoid rebuilding inside sdist.
        if package_lib_path.exists():
            self.announce(f"Found packaged lib at {package_lib_path}", level=3)
            lib_path = package_lib_path
        else:
            lib_path = None

            if os.name == "nt":
                self.announce("Building GigaVector shared library via CMake (Windows)", level=3)
                repo_root = _find_repo_root()
                build_dir = Path(__file__).resolve().parent / "build-cmake-py"
                build_dir.mkdir(parents=True, exist_ok=True)

                env = os.environ.copy()
                hardening = env.get("HARDENING_FLAGS", "")
                extra_cflags = hardening.strip()
                cmake_c_flags = env.get("CMAKE_C_FLAGS", "").strip()
                combined_c_flags = " ".join([x for x in [cmake_c_flags, extra_cflags] if x])

                cmake_gen = _cmake_generator(env)
                # Multi-config generators (Visual Studio, Ninja Multi-Config) pass
                # --config at build time; single-config generators (MinGW Makefiles,
                # Ninja, NMake) require CMAKE_BUILD_TYPE at configure time.
                _MULTI_CONFIG = {"Visual Studio", "Ninja Multi-Config", "Xcode"}
                is_multi_config = not cmake_gen or any(
                    cmake_gen.startswith(mc) for mc in _MULTI_CONFIG if mc
                ) or cmake_gen == ""

                cmake_args = [
                    "cmake", "-S", str(repo_root), "-B", str(build_dir),
                    "-DBUILD_TESTS=OFF", "-DBUILD_BENCHMARKS=OFF",
                ]
                if cmake_gen:
                    cmake_args += ["-G", cmake_gen]
                if not is_multi_config:
                    # Single-config generators need CMAKE_BUILD_TYPE at configure time.
                    cmake_args += ["-DCMAKE_BUILD_TYPE=Release"]
                if combined_c_flags:
                    cmake_args.append(f"-DCMAKE_C_FLAGS={combined_c_flags}")

                subprocess.check_call(cmake_args, env=env)

                build_cmd = ["cmake", "--build", str(build_dir)]
                if is_multi_config:
                    # Multi-config generators select the config at build time.
                    build_cmd += ["--config", "Release"]
                subprocess.check_call(build_cmd, env=env)

                # MSVC puts the DLL in <build>/Release/; MinGW puts it at <build>/ with a lib prefix.
                candidates = [
                    build_dir / "Release" / "GigaVector.dll",
                    build_dir / "GigaVector.dll",
                    build_dir / "libGigaVector.dll",
                ]
                lib_path = next((p for p in candidates if p.exists()), None)
                if lib_path is None:
                    hits = glob.glob(str(build_dir / "**" / "*.dll"), recursive=True)
                    dll_hits = [h for h in hits if "GigaVector" in Path(h).name]
                    if dll_hits:
                        lib_path = Path(dll_hits[0])
                if lib_path is None:
                    raise FileNotFoundError(f"{lib_filename} not found after CMake build in {build_dir}")

            else:
                repo_root = _find_repo_root()
                # POSIX: build via Makefile
                candidate = repo_root / "build" / "lib" / lib_filename
                if candidate.exists():
                    lib_path = candidate
                    self.announce("Using prebuilt GigaVector shared library", level=3)
                elif (repo_root / "Makefile").exists():
                    self.announce("Building GigaVector shared library via make", level=3)
                    env = os.environ.copy()
                    subprocess.check_call(["make", "-C", str(repo_root), "lib"], env=env)
                    candidate = repo_root / "build" / "lib" / lib_filename
                    if candidate.exists():
                        lib_path = candidate
                if lib_path is None:
                    raise FileNotFoundError(f"{lib_filename} not found and build failed at {repo_root}")

        # Avoid copying onto itself inside sdist build trees.
        if lib_path.resolve() != package_lib_path.resolve():
            package_lib_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(lib_path, package_lib_path)
            self.announce(f"Copied {lib_path} -> {package_lib_path}", level=3)
        else:
            self.announce(f"Library already present at {package_lib_path}", level=3)

        super().run()


class sdist(_sdist):
    """Bundle C sources into the sdist so Windows can build from source."""

    def run(self):
        repo_root = Path(__file__).resolve().parent.parent
        try:
            if (repo_root / "CMakeLists.txt").exists():
                if _CSRC_DIR.exists():
                    shutil.rmtree(_CSRC_DIR)
                shutil.copytree(repo_root / "src", _CSRC_DIR / "src")
                shutil.copytree(repo_root / "include", _CSRC_DIR / "include")
                shutil.copy2(repo_root / "CMakeLists.txt", _CSRC_DIR / "CMakeLists.txt")
            super().run()
        finally:
            if _CSRC_DIR.exists():
                shutil.rmtree(_CSRC_DIR)


class bdist_wheel(_bdist_wheel):
    def initialize_options(self):
        super().initialize_options()
        # Force a platform wheel; we bundle native binaries.
        self.root_is_pure = False

    def finalize_options(self):
        super().finalize_options()
        # This wheel bundles a native ELF shared library (libGigaVector.so),
        # so it must not be tagged as "py3-none-any".
        self.root_is_pure = False


setup(cmdclass={
    "build_py": BuildPyWithMake,
    "bdist_wheel": bdist_wheel,
    "sdist": sdist,
})
