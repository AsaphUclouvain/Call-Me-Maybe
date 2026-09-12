install:
	uv sync

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports \
	--disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 src
	mypy src --strict

run: install
	uv run python -m src --trace


debug:
	uv run python -m pdb -m src


clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

fclean: clean
	rm -rf data/output

.PHONY: install run debug clean lint lint-strict