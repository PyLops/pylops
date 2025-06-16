PIP := $(shell command -v pip3 2> /dev/null || command which pip 2> /dev/null)
PYTHON := $(shell command -v python3 2> /dev/null || command which python 2> /dev/null)

.PHONY: install dev-install dev-install_gpu install_conda dev-install_conda dev-install_conda_arm tests tests_cpu_ongpu tests_gpu doc docupdate servedoc lint typeannot coverage

pipcheck:
ifndef PIP
	$(error "Ensure pip or pip3 are in your PATH")
endif
	@echo Using pip: $(PIP)

pythoncheck:
ifndef PYTHON
	$(error "Ensure python or python3 are in your PATH")
endif
	@echo Using python: $(PYTHON)

install:
	make pipcheck
	$(PIP) install -r requirements.txt && $(PIP) install .

dev-install:
	make pipcheck
	$(PIP) install -r requirements-dev.txt &&\
	$(PIP) install -r requirements-torch.txt && $(PIP) install -e .

dev-install_gpu:
	make pipcheck
	$(PIP) install -r requirements-dev-gpu.txt &&\
	$(PIP) install -e .

install_conda:
	conda env create -f environment.yml && conda activate pylops && pip install .

dev-install_conda:
	conda env create -f environment-dev.yml && conda activate pylops && pip install -e .

dev-install_conda_arm:
	conda env create -f environment-dev-arm.yml && conda activate pylops && pip install -e .

dev-install_conda_gpu:
	conda env create -f environment-dev-gpu.yml && conda activate pylops_gpu && pip install -e .

tests:
	# Run tests with CPU
	make pythoncheck
	pytest

tests_cpu_ongpu:
	# Run tests with CPU on a system with GPU (and CuPy installed)
	make pythoncheck
	export CUPY_PYLOPS=0 && export TEST_CUPY_PYLOPS=0 && pytest

tests_gpu:
	# Run tests with GPU (requires CuPy to be installed)
	make pythoncheck
	export TEST_CUPY_PYLOPS=1 && pytest

doc:
	cd docs  && rm -rf source/api/generated && rm -rf source/gallery &&\
	rm -rf source/tutorials && rm -rf build && make html && cd ..

docupdate:
	cd docs && make html && cd ..

servedoc:
	$(PYTHON) -m http.server --directory docs/build/html/

lint:
	flake8 docs/ examples/ pylops/ pytests/ tutorials/

typeannot:
	mypy pylops/

coverage:
	coverage run -m pytest && coverage xml && coverage html && $(PYTHON) -m http.server --directory htmlcov/
