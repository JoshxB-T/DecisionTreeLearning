PY=python3

all: run

run: main.py
	$(PY) $<

clean:
	rm -rf __pycache__ *.pyc