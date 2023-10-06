if [ ! -e venv ]
then
    virtualenv venv
fi

source venv/bin/activate
pip install -r requirements-dev.txt
pip install --editable .

echo running tests

pytest -n 6 .

