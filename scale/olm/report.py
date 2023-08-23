from pathlib import Path
import scale.olm.common as common
import json


def __stub1_params_summary(params):
    summary = ""
    for x in params:
        summary += "- {}: {}\n".format(x, params[x])
    return summary


def __stub1_library_summary():
    return "Hello library.summary!"


def __stub1_generate_summary():
    return "Hello generate.summary!"


def __stub1_run_summary():
    return "Hello run.summary!"


def __stub1_build_summary():
    return "Hello build.summary!"


def __stub1_check_summary():
    return "Hello check.summary!"


def stub1(model, template):
    common.logger.info(f"reading RST template={template}")

    # Load the template file.
    with open(Path(model["dir"]) / template, "r") as f:
        template_text = f.read()

    # Load a data file.
    work_dir = Path(model["work_dir"])
    generate_json = work_dir / "generate.json"
    with open(generate_json, "r") as f:
        generate_d = json.load(f)

    data = {
        "params": {"summary": __stub1_params_summary(generate_d["params"])},
        "library": {"summary": __stub1_library_summary()},
        "generate": {"summary": __stub1_generate_summary()},
        "run": {"summary": __stub1_run_summary()},
        "build": {"summary": __stub1_build_summary()},
        "check": {"summary": __stub1_check_summary()},
        "model": model,
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

    return {"template": str(template), "report": str(report), "rst": str(rst)}
