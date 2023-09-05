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
    rst = work_dir / (model["name"] + ".rst")
    pdf = rst.with_suffix(".pdf")
    data.update(
        {
            "_work_dir": str(work_dir),
            "_template": str(template),
            "_pdf": str(pdf.relative_to(work_dir)),
            "_rst": str(rst.relative_to(work_dir)),
        }
    )
    with open(work_dir / "report.json", "w") as f:
        core.logger.info(f"writing data for report.json")
        json.dump(data, f, indent=4)

    # Expand template.
    filled_text = common.expand_template(template_text, data)
    with open(rst, "w") as f:
        core.logger.info(f"writing RST report {rst}")
        f.write(filled_text)

    # Generate PDF.
    common.run_command(f"rst2pdf -s twocolumn {rst}", check_return_code=False)
    core.logger.info(f"Generated PDF report {pdf}")

    return data
