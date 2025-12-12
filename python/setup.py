import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyWithMake(build_py):
    """Run `make lib` in the repository root and copy the .so into the package."""

    def run(self):
        repo_root = Path(__file__).resolve().parent.parent
        package_lib_path = Path(__file__).resolve().parent / "src" / "gigavector" / "libGigaVector.so"

        # Prefer an already-packaged library to avoid rebuilding inside sdist.
        if package_lib_path.exists():
            self.announce(f"Found packaged lib at {package_lib_path}", level=3)
            lib_path = package_lib_path
        else:
            lib_path = repo_root / "build" / "lib" / "libGigaVector.so"
            if lib_path.exists():
                self.announce("Using prebuilt GigaVector shared library", level=3)
            elif (repo_root / "Makefile").exists():
                self.announce("Building GigaVector shared library via make", level=3)
                subprocess.check_call(["make", "-C", str(repo_root), "lib"])
            else:
                raise FileNotFoundError(f"libGigaVector.so not found and Makefile missing at {repo_root}")

        # Avoid copying onto itself inside sdist build trees.
        if lib_path.resolve() != package_lib_path.resolve():
            package_lib_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(lib_path, package_lib_path)
            self.announce(f"Copied {lib_path} -> {package_lib_path}", level=3)
        else:
            self.announce(f"Library already present at {package_lib_path}", level=3)

        super().run()


setup(cmdclass={"build_py": BuildPyWithMake})


