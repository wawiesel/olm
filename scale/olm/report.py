from pathlib import Path
import scale.olm.common as common
import json


def __stub1_summary_run():
    return "Hello run!"


def __stub1_summary_check():
    return "Hello check!"


def __stub1_summary_lib():
    return "Hello lib!"


def stub1(model, template):
    common.logger.info(f"reading RST template={template}")

    # Load the template file.
    with open(Path(model["dir"]) / template, "r") as f:
        template_text = f.read()

    # Load data files.
    work_dir = Path(model["work_dir"])
    data = {"model": model}
    for x in ["generate", "build", "run"]:
        j = work_dir / (x + ".json")
        with open(j, "r") as f:
            data[x] = json.load(f)

    # Populate basic section summaries.
    data["summary"] = {
        "run": __stub1_summary_run(),
        "check": __stub1_summary_check(),
        "lib": __stub1_summary_lib(),
    }
    filled_text = common.expand_template(template_text, data)

    # Fill template.
    rst = Path(model["work_dir"]) / (model["name"] + ".rst")
    with open(rst, "w") as f:
        common.logger.info(f"writing RST report {rst}")
        f.write(filled_text)

    # Generate PDF.
    report = rst.with_suffix(".pdf")
    common.run_command(f"rst2pdf {rst}")
    common.logger.info(f"Generated PDF report {report}")

    return {
        "template": str(template),
        "report": str(report),
        "rst": str(rst),
        "data": data,
    }
