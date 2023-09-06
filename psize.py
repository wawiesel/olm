from pathlib import Path
import sys

for p in [Path(x) for x in sys.argv[1:]]:
    print(p, p.stat().st_size)
