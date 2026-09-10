PIP := $(shell command -v pip3 2> /dev/null || command which pip 2> /dev/null)
PYTHON := $(shell command -v python3 2> /dev/null || command which python 2> /dev/null)
UV := $(shell command -v uv 2> /dev/null || command which uv 2> /dev/null)
NOX := $(shell command -v nox 2> /dev/null || command which nox 2> /dev/null)

.PHONY: install_conda dev-install_conda dev-install_conda_intel_mkl dev-install_conda_arm dev-install_conda_gpu
.PHONY: dev-install_uv dev-install_uvcu126 dev-install_uvcu128 dev-install_uvcu13
.PHONY: tests tests_cpu_ongpu tests_gpu tests_uv tests_cpu_ongpu_uv tests_gpu_uv tests_nox
.PHONY: doc doc_uv docupdate docupdate_uv servedoc servedoc_uv lint lint_uv typeannot typeannot_uv
.PHONY: coverage coverage_uv

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

uvcheck:
ifndef UV
	$(error "Ensure uv is in your PATH")
endif
	@echo Using uv: $(UV)

noxcheck:
ifndef NOX
	$(error "Ensure nox is in your PATH")
endif
	@echo Using nox: $(NOX)

install_conda:
	# Create conda environment and install pylops in it
	conda env create -f environment.yml && source ${CONDA_PREFIX}/etc/profile.d/conda.sh && conda activate pylops && pip install .

dev-install_conda:
	# Create conda environment and install pylops in it (editable mode)
	conda env create -f environment-dev.yml && source ${CONDA_PREFIX}/etc/profile.d/conda.sh && conda activate pylops && pip install -e .

dev-install_conda_intel_mkl:
	# Create conda environment and install pylops in it (editable mode) with Intel MKL
	conda env create -f environment-dev-intel-mkl.yml && source ${CONDA_PREFIX}/etc/profile.d/conda.sh && conda activate pylops && pip install -e .

dev-install_conda_arm:
	# Create conda environment and install pylops in it (editable mode) for ARM architecture
	conda env create -f environment-dev-arm.yml && source ${CONDA_PREFIX}/etc/profile.d/conda.sh && conda activate pylops && pip install -e .

dev-install_conda_gpu:
	# Create conda environment and install pylops in it (editable mode) for GPU
	conda env create -f environment-dev-gpu.yml && source ${CONDA_PREFIX}/etc/profile.d/conda.sh && conda activate pylops_gpu && pip install -e .

dev-install_uv:
	# Create conda environment and install pylops in it (editable mode) using uv
	make uvcheck
	$(UV) sync --locked  --extra advanced  --extra stat --extra deep --all-groups

dev-install_uvcu126:
	# Create conda environment and install pylops in it (editable mode) using uv with CUDA 12.6
	make uvcheck
	$(UV) sync --locked  --extra advanced  --extra stat --extra gpu-cu12 --extra deep-cu126 --all-groups

dev-install_uvcu128:
	# Create conda environment and install pylops in it (editable mode) using uv with CUDA 12.8
	make uvcheck
	$(UV) sync --locked  --extra advanced  --extra stat --extra gpu-cu12 --extra deep-cu128 --all-groups

dev-install_uvcu13:
	# Create conda environment and install pylops in it (editable mode) using uv with CUDA 13
	make uvcheck
	$(UV) sync --locked  --extra advanced  --extra stat --extra gpu-cu13 --extra deep-cu13 --all-groups

tests:
	# Run tests with CPU
	make pythoncheck
	pytest

tests_uv:
	# Run tests with CPU using uv
	make uvcheck
	$(UV) run pytest

tests_nox:
	# Run tests with CPU using nox
	make noxcheck
	$(NOX) -s tests

tests_cpu_ongpu:
	# Run tests with CPU on a system with GPU (and CuPy installed)
	make pythoncheck
	export CUPY_PYLOPS=0 && export TEST_CUPY_PYLOPS=0 && pytest

tests_cpu_ongpu_uv:
	# Run tests with CPU on a system with GPU (and CuPy installed) using uv
	make pythoncheck
	export CUPY_PYLOPS=0 && export TEST_CUPY_PYLOPS=0 && $(UV) run pytest

tests_gpu:
	# Run tests with GPU (requires CuPy to be installed)
	make pythoncheck
	export TEST_CUPY_PYLOPS=1 && pytest

tests_gpu_uv:
	# Run tests with GPU (requires CuPy to be installed) using uv
	make pythoncheck
	export TEST_CUPY_PYLOPS=1 && $(UV) run pytest

doc:
	# Build the documentation (HTML) from the sources in ``docs/source``
	make pythoncheck
	cd docs && rm -rf source/api/generated && rm -rf source/gallery &&\
	rm -rf source/tutorials && rm -rf source/examples &&\
	rm -rf build && make html && cd ..

doc_uv:
	# Build the documentation (HTML) from the sources in ``docs/source`` using uv
	make uvcheck
	cd docs  && rm -rf source/api/generated && rm -rf source/gallery &&\
	rm -rf source/tutorials && rm -rf source/examples &&\
	rm -rf build && $(UV) run make html && cd ..

docupdate:
	# Update the documentation (HTML) from the sources in ``docs/source``
	make pythoncheck
	cd docs && make html && cd ..

docupdate_uv:
	# Update the documentation (HTML) from the sources in ``docs/source``
	make uvcheck
	cd docs && $(UV) run make html && cd ..

servedoc:
	# Serve the documentation (HTML) from the build directory
	make pythoncheck
	$(PYTHON) -m http.server --directory docs/build/html/

servedoc_uv:
	# Serve the documentation (HTML) from the build directory using uv
	make uvcheck
	$(UV) run python -m http.server --directory docs/build/html/

lint:
	# Run ruff linter on the source code and examples
	make pythoncheck
	ruff check docs/source examples/ pylops/ pytests/ tutorials/

lint_uv:
	# Run ruff linter on the source code and examples using uv
	make uvcheck
	$(UV) run ruff check docs/source examples/ pylops/ pytests/ tutorials/

typeannot:
	# Run mypy type checker on the source code
	make pythoncheck
	mypy pylops/

typeannot_uv:
	# Run mypy type checker on the source code using uv
	make uvcheck
	$(UV) run mypy pylops/

coverage:
	# Run tests with coverage and generate HTML report
	make pythoncheck
	coverage run --source=pylops -m pytest && \
	coverage xml && coverage html && $(PYTHON) -m http.server --directory htmlcov/

coverage_uv:
	# Run tests with coverage and generate HTML report using uv
	make uvcheck
	$(UV) run coverage run --source=pylops -m pytest  &&\
	$(UV) run coverage xml &&\
	$(UV) run coverage html  &&\
	$(UV) run python -m http.server --directory htmlcov/
