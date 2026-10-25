PRETTIER_VERSION := 3.0.3
PRETTIER := npm exec --yes --package=prettier@$(PRETTIER_VERSION) -- prettier

.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@uv sync
	@uv run pre-commit install

.PHONY: lint format-check typecheck deptry prettier-format check
lint:
	@echo "🚀 Linting code: Running ruff"
	uv run --locked ruff check --config ruff.toml .

format-check:
	@echo "🚀 Checking code formatting: Running ruff"
	uv run --locked ruff format --check --config ruff.toml .

typecheck:
	@echo "🚀 Static type checking: Running mypy"
	uv run --locked mypy

deptry:
	@echo "🚀 Checking for obsolete dependencies: Running deptry"
	uv run --locked deptry .

prettier-format:
	$(PRETTIER) --write .

check: lint format-check typecheck deptry prettier-format
	uv run --locked python scripts/check_agent_rules.py
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running pre-commit"
	@uv run pre-commit run -a

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest --cov --cov-config=pyproject.toml --cov-report=html

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "🚀 Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"


.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	@uv run mkdocs build -s

.PHONY: docs
docs: ## Build and serve the documentation
	@uv run mkdocs serve

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
