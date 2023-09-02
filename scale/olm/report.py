from pathlib import Path
import scale.olm.common as common
import scale.olm.core as core
import json


def __stub1_summary_check():
    return "Hello check!"


def __stub1_summary_lib():
    return "Hello lib!"


def stub1(model, template):
    core.logger.info(f"reading RST template={template}")

    # Load the template file.
    with open(Path(model["dir"]) / template, "r") as f:
        template_text = f.read()

    # Load data files.
    work_dir = Path(model["work_dir"])
    data = {"model": model}
    for x in ["generate", "build", "run", "check"]:
        j = work_dir / (x + ".json")
        with open(j, "r") as f:
            data[x] = json.load(f)

    # Expand template.
    filled_text = common.expand_template(template_text, data)

    # Fill template.
    rst = Path(model["work_dir"]) / (model["name"] + ".rst")
    with open(rst, "w") as f:
        core.logger.info(f"writing RST report {rst}")
        f.write(filled_text)

    # Generate PDF.
    pdf = rst.with_suffix(".pdf")
    common.run_command(f"rst2pdf {rst}")
    core.logger.info(f"Generated PDF report {pdf}")

    return {
        "work_dir": str(work_dir),
        "template": str(template),
        "pdf": str(pdf.relative_to(work_dir)),
        "rst": str(rst.relative_to(work_dir)),
        "data": data,
    }
