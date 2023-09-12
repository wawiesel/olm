from pathlib import Path
import scale.olm.internal as internal
import scale.olm.core as core
import json


def rst2pdf(template, _model, _env):
    """Generate a report using rst2pdf.

    Report templates are provided in restructured text. This function
    expands those templates with a report.olm.json file that has data
    from all the other stages.

        1. generate
        2. run
        3. assemble
        4. check

    """

    # Load the template file.
    template_path = Path(_env["config_file"]).parent / template
    internal.logger.info("Initializing report", template=str(template_path))
    with open(template_path, "r") as f:
        template_text = f.read()

    # Load data files.
    work_path = Path(_env["work_dir"])
    data = {"model": _model}
    for x in ["generate", "run", "assemble", "check"]:
        j = work_path / (x + ".olm.json")
        with open(j, "r") as f:
            data[x] = json.load(f)
    rst = work_path / (_model["name"] + ".rst")
    pdf = rst.with_suffix(".pdf")
    report_data_path = work_path / "report.olm.json"
    with open(report_data_path, "w") as f:
        internal.logger.info("Writing report data", report_data=str(report_data_path))
        json.dump(data, f, indent=4)

    # Expand template.
    filled_text = core.TemplateManager.expand_text(template_text, data)
    with open(rst, "w") as f:
        internal.logger.info("Writing report content", rst_file=str(rst))
        f.write(filled_text)

    # Generate PDF.
    internal.run_command(f"rst2pdf -s twocolumn {rst}", check_return_code=False)
    internal.logger.info("Generated report", pdf_file=str(pdf))

    data["_"] = {
        "work_dir": str(work_path),
        "template": str(template_path),
        "pdf_file": str(pdf),
        "rst_file": str(rst),
    }

    return data
