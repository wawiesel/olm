import scale.olm as olm
from pathlib import Path
import json

this_dir = Path(__file__).parent.resolve()

all = [
    olm.generate.time,
    olm.generate.root,
    olm.generate.comp,
    olm.generate.static,
    olm.generate.dynamic,
    olm.generate.states,
    olm.run,
    olm.assemble,
    olm.check,
    olm.report,
]


def go(modules=all):
    for m in modules:
        for x in m.__all__:
            s = olm.internal.schema(m.__name__ + ":" + x, color=False, description=True)
            jj = m.__name__ + "." + x + ".json"
            j = this_dir / jj
            with open(j, "w") as f:
                f.write(json.dumps(s, indent=4))
            rr = m.__name__ + "." + x + ".rst"
            r = this_dir / rr
            with open(r, "w") as f:
                f.write(
                    f"""
.. jsonschema:: {jj}
    :lift_description:
    :auto_reference:

:download:`{jj} </create/schema/{jj}>`
    """
                )


if __name__ == "__main__":
    go()
