"""Compare solution dumps written by `poisson -o` against the 1x1 run.

Usage: python3 tests/compare.py DIR
DIR holds u_<tag>_<Px>x<Py>_P<P>.bin.  Prints max |u - u_1x1| for every
other grid; Jacobi must be bitwise identical, CG within 1e-12.  Exit status
is non-zero on failure.
"""
import glob, os, re, sys
import numpy as np

d = sys.argv[1]
ok = True
for f in sorted(glob.glob(os.path.join(d, "u_*_P*.bin"))):
    tag, grid, P = re.match(r"u_(.+)_(\d+x\d+)_P(\d+)\.bin", os.path.basename(f)).groups()
    if grid == "1x1":
        continue
    ref = np.fromfile(os.path.join(d, f"u_{tag}_1x1_P1.bin"))
    diff = float(np.abs(np.fromfile(f) - ref).max())
    lim = 0.0 if tag == "jacobi" else 1e-12
    ok &= diff <= lim
    print(f"COMPARE tag={tag} grid={grid} P={P} maxdiff={diff:.3e} {'ok' if diff <= lim else 'FAIL'}")
print("COMPARE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
