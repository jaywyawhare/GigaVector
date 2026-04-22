from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import zipfile
from pathlib import Path


def _is_within(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def fix_wheel(wheel_path: Path) -> None:
    wheel_path = wheel_path.resolve()
    if not wheel_path.name.endswith(".whl"):
        raise ValueError(f"Not a wheel: {wheel_path}")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        unpack_dir = td_path / "unpack"
        unpack_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(wheel_path, "r") as zf:
            zf.extractall(unpack_dir)

        # Find the one *.data/purelib tree and move it to *.data/platlib.
        purelib_dirs = list(unpack_dir.glob("*.data/purelib"))
        if not purelib_dirs:
            # Nothing to fix.
            return
        if len(purelib_dirs) != 1:
            raise RuntimeError(f"Expected 1 purelib dir, found {len(purelib_dirs)}")

        purelib_dir = purelib_dirs[0]
        data_dir = purelib_dir.parent
        platlib_dir = data_dir / "platlib"

        if platlib_dir.exists():
            # If both exist, refuse to guess merges.
            raise RuntimeError(f"platlib already exists in wheel: {platlib_dir}")

        # Move purelib -> platlib
        platlib_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(purelib_dir), str(platlib_dir))

        # Repack wheel deterministically-ish: zipfile doesn't preserve order anyway,
        # but that's fine for our use (auditwheel runs next).
        tmp_wheel = td_path / wheel_path.name
        with zipfile.ZipFile(tmp_wheel, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(unpack_dir.rglob("*")):
                if p.is_dir():
                    continue
                if not _is_within(unpack_dir, p):
                    continue
                arcname = str(p.relative_to(unpack_dir)).replace(os.sep, "/")
                zf.write(p, arcname)

        shutil.copy2(tmp_wheel, wheel_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wheel", type=Path)
    args = ap.parse_args()
    fix_wheel(args.wheel)


if __name__ == "__main__":
    main()

