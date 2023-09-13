import json
import sys
from pathlib import Path


def find_replace_input(xstr):
    import re

    zone = ""
    comp = ""

    # Global replacements to normalize input to expected below.
    xstr = xstr.replace("fuelcomps.", "fuelcomp.")

    # Define a regular expression pattern to match the multiline string
    pattern = r"{% if params\.density_Am > 0\.0 -%}(.*?){% endif -%}"

    # Use re.sub() with re.DOTALL to match across multiple lines and replace
    xstr = re.sub(pattern, "AM241", xstr, flags=re.DOTALL)

    # Define a regular expression pattern to match the temperature value
    # Use re.search to find the match in the text
    pattern = r"(\d+(\.\d*)?)\s+92235\s"
    match = re.search(pattern, xstr)
    if match:
        am_temp = match.group(1)

    lines = ""
    mix = None
    for line in xstr.split("\n"):
        if line.find("' fuel inner") != -1:
            zone = "inner"
        elif line.find("' fuel inside edge") != -1:
            zone = "iedge"
        elif line.find("' fuel edge") != -1:
            zone = "iedge"
        elif line.find("' fuel corner") != -1:
            zone = "corner"
        m = re.search("^uo2\s+(\d+)", line)
        if m:
            comp = "uo2"
            mix = m.group(1)
        m = re.search("^puo2\s+(\d+)", line)
        if m:
            comp = "puo2"
            mix = m.group(1)

        if line.find("AM241") != -1:
            line = line.replace(
                "AM241",
                f"""
  am   {{mix}} den={{{{comp.{zone}.density}}}}
        {{{{comp.{zone}.amo2.dens_frac*comp.{zone}.info.amo2_hm_frac}}}} {am_temp}
        95241 {{{{comp.{zone}.amo2.iso.am241}}}} end
  o    {{mix}} den={{{{comp.{zone}.density}}}}
        {{{{comp.{zone}.amo2.dens_frac*(1.0-comp.{zone}.info.amo2_hm_frac)}}}} {am_temp} end""",
            )

        line = line.replace("fuelcomp.fuel_density", f"comp.{zone}.density")
        m = re.search(r"({{100.0-fuelcomp.wtpt_[^ 0-9]*?}})", line)
        if m:
            line = line.replace(m.group(1), f"{{{{comp.{zone}.uo2.dens_frac}}}}")
        m = re.search(r"({{fuelcomp.wtpt_[^ 0-9]*?}})", line)
        if m:
            line = line.replace(m.group(1), f"{{{{comp.{zone}.puo2.dens_frac}}}}")

        # handle all isotopic entry
        m = re.search(r"fuelcomp.wtpt_([^ ]*)", line)
        if m:
            line = line.replace("fuelcomp.wtpt_", f"comp.{zone}.{comp}.iso.")

        lines += line + "\n"

        # Final global replaces.
        lines = lines.replace("params.", "static.")
    return lines


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
    xstr = xstr.replace("generate_rst", "rst2pdf")
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
    xstr = find_replace_input(xstr)
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
