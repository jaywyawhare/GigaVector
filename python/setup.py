import shutil
import subprocess
from pathlib import Path
import os
import sys
import glob

from setuptools import setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class BuildPyWithMake(build_py):
    """Run `make lib` in the repository root and copy the .so into the package."""

    def run(self):
        repo_root = Path(__file__).resolve().parent.parent
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
                build_dir = repo_root / "build-cmake-py"
                build_dir.mkdir(parents=True, exist_ok=True)

                env = os.environ.copy()
                hardening = env.get("HARDENING_FLAGS", "")
                extra_cflags = hardening.strip()
                cmake_c_flags = env.get("CMAKE_C_FLAGS", "").strip()
                combined_c_flags = " ".join([x for x in [cmake_c_flags, extra_cflags] if x])

                cmake_args = [
                    "cmake", "-S", str(repo_root), "-B", str(build_dir),
                    "-DBUILD_TESTS=OFF", "-DBUILD_BENCHMARKS=OFF",
                    "-DCMAKE_BUILD_TYPE=Release",
                ]
                if combined_c_flags:
                    cmake_args.append(f"-DCMAKE_C_FLAGS={combined_c_flags}")

                subprocess.check_call(cmake_args, env=env)
                subprocess.check_call(["cmake", "--build", str(build_dir), "--config", "Release"], env=env)

                candidates = [
                    build_dir / "Release" / "GigaVector.dll",
                    build_dir / "GigaVector.dll",
                ]
                lib_path = next((p for p in candidates if p.exists()), None)
                if lib_path is None:
                    hits = glob.glob(str(build_dir / "**" / "GigaVector.dll"), recursive=True)
                    if hits:
                        lib_path = Path(hits[0])
                if lib_path is None:
                    raise FileNotFoundError(f"{lib_filename} not found after CMake build in {build_dir}")

            else:
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


class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        # This wheel bundles a native ELF shared library (libGigaVector.so),
        # so it must not be tagged as "py3-none-any".
        self.root_is_pure = False


setup(cmdclass={"build_py": BuildPyWithMake, "bdist_wheel": bdist_wheel})


