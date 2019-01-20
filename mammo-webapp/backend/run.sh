export FLASK_APP=./mammo-backend/index.py
export FLASK_ENV=development
export GOOGLE_APPLICATION_CREDENTIALS=./credentials.json
source $(pipenv --venv)/bin/activate
flask run