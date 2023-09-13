import json
import sys
from pathlib import Path


def find_replace(xstr):
    xstr = xstr.replace(".type", "_type")
    xstr = xstr.replace("generate:all_permutations", "generate:full_hypercube")
    xstr = xstr.replace("wtpt_pu239", "pu239_frac")
    xstr = xstr.replace("wtpt_pu", "pu_frac")
    xstr = xstr.replace("common", "internal")
    xstr = xstr.replace("build", "assemble")
    xstr = xstr.replace("fuelcomp", "comp")
    xstr = xstr.replace("params", "static")
    xstr = xstr.replace("fuel_density", "density")
    xstr = xstr.replace("scale.olm.generate:expander", "scale.olm.generate:jt_expander")
    xstr = xstr.replace("generate:comp_mox_ornltm2003_2", "complib:mox_multizone_2023")
    xstr = xstr.replace("triton_constpower_burndata", "constpower_burndata")
    return xstr


config = Path(sys.argv[1])
with open(config, "r") as f:
    x = json.load(f)
xstr = json.dumps(x)
xstr = find_replace(xstr)
x = json.loads(xstr)

model = config.parent / x["generate"]["template"]
with open(model, "r") as f:
    xstr = f.read()
    xstr = find_replace(xstr)
with open(model, "w") as f:
    f.write(xstr)

x["generate"]["comp"].pop("nuclide_prefix", None)
x["model"].pop("work_dir", None)
x["model"].pop("scale_env_var", None)
x["run"].pop("nprocs", None)
x["run"]["dry_run"] = False
x["generate"]["static"].pop("density_Am", None)
x["assemble"].pop("suffix", None)

is_bwr = len(x["generate"]["states"]["coolant_density"]) > 1

if x["generate"]["comp"]["_type"] == "scale.olm.complib:mox_multizone_2023":
    comp = x["generate"]["comp"]
    comp["_type"] = "scale.olm.complib:mox_multizone_2023"
    comp["zone_pins"] = comp.pop("pins_zone", comp.get("zone_pins", None))
    comp["zone_names"] = "BWR2016" if is_bwr else "PWR2016"
    comp["gd2o3_pins"] = comp.pop("pins_gd", comp.get("gd2o3_pins", None))
    comp["gd2o3_wtpct"] = comp.pop("pct_gd", comp.get("gd2o3_wtpct", None))
    x["generate"]["comp"] = comp

from collections import OrderedDict

y = OrderedDict()
for key in ["model", "generate", "run", "assemble", "check", "report"]:
    y[key] = x[key]


with open(sys.argv[1], "w") as f:
    json.dump(y, f, indent=4)
