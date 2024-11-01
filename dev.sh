if [ ! -e venv ]
then
    echo "Setting up virtual environment..."
    virtualenv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing requirements..."
pip install -r requirements.txt

echo "Installing OLM..."
pip install --editable .

echo "Updating data submodule..."
git submodule update --init data
pushd data
git lfs pull .
popd

echo "Running tests"
pytest -n 6 .

