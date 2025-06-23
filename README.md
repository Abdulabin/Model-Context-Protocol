pip install uv
uv venv --python 3.11

.venv\Scripts\activate

uv add -r requirements.txt

RUN COMMAND: uvicorn app:app