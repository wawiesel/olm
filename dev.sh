_olm_dev_run() {
    if [ ! -d venv ]
    then
        _olm_python=${OLM_PYTHON:-python3.12}
        echo "Setting up virtual environment with $_olm_python..."
        "$_olm_python" -m venv venv || return
    fi

    echo "Activating virtual environment..."
    source venv/bin/activate || return

    python -c 'import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else "OLM dev setup requires Python 3.9-3.12")' || return

    echo "Installing OLM with development extras..."
    python -m pip install --editable ".[test,docs,notebooks,dev]" || return

    echo "Updating data submodule..."
    git submodule update --init data || return
    (cd data && git lfs pull .) || return

    echo "Running tests"
    python -m pytest -n 6 .
}

_olm_dev_run
_olm_dev_status=$?
unset -f _olm_dev_run
return $_olm_dev_status 2>/dev/null || exit $_olm_dev_status
