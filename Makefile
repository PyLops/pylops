.PHONY: install dev-install install_conda dev-install_conda tests doc docupdate

install:
	pip install -r requirements.txt && pip install .

dev-install:
	pip install -r requirements-dev.txt && pip install -e .

install_conda:
	conda env create -f environment.yml && source activate lops && pip install .

dev-install_conda:
	conda env create -f environment-dev.yml && source activate lops && pip install -e .

tests:
	python setup.py test

doc:
	cd docs  && rm -rf source/api/generated && rm -rf source/gallery &&\
	rm -rf source/tutorials && rm -rf build && make html && cd ..

docupdate:
	cd docs && make html && cd ..