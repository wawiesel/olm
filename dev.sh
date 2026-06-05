if [ ! -e venv ]
then
    echo "Setting up virtual environment..."
    virtualenv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing pinned development snapshot..."
pip install -r requirements.txt

echo "Installing OLM with development extras..."
pip install --editable ".[test,docs,notebooks]"

echo "Updating data submodule..."
git submodule update --init data
pushd data
git lfs pull .
popd

echo "Running tests"
pytest -n 6 .
