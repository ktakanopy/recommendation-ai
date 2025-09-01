## Recommendation API – Two-Towers

### Quickstart
- **Prereqs**: Python 3.13+, Poetry
- **Install deps**:
```bash
poetry install --with dev
```
- (Optional) **Activate venv**:
```bash
poetry shell
```

### Run the API (task runner)
- **Dev server with autoreload**:
```bash
poetry run task run
```
- The server starts via `fastapi dev` using `neural_recommendation/app.py`.
- Visit: `http://127.0.0.1:8000` → Docs: `http://127.0.0.1:8000/docs`
- Tweak log level with `LOG_LEVEL` (default set in the task).

### Lint and format
- **Lint** (Ruff):
```bash
poetry run task lint
```
- **Auto-fix common issues**:
```bash
poetry run task pre_format
```
- **Format** (Ruff formatter):
```bash
poetry run task format
```

### Tests and coverage
- **Run tests with coverage**:
```bash
poetry run task test
```
- This task will:
  - Run linting first
  - Execute tests (`pytest -s -x --cov=neural_recommendation -vv`)
  - Generate HTML coverage (`coverage html`)
- Open coverage report at `htmlcov/index.html`.

### Task reference (from `pyproject.toml`)
- **run**: `fastapi dev neural_recommendation/app.py`
- **lint**: `ruff check`
- **pre_format**: `ruff check --fix`
- **format**: `ruff format`
- **test**: `pytest -s -x --cov=neural_recommendation -vv` (with pre/post hooks)

### Poetry environment reset
- Verify Python version:
```bash
python3.13 --version
```
- Remove current Poetry virtualenv:
```bash
poetry env list
poetry env info --path
poetry env remove $(poetry env info --path)
```
- (Optional) clear Poetry caches:
```bash
poetry cache clear pypi --all -n
```
- Recreate environment and install deps:
```bash
poetry env use python3.13
poetry install
```
- Validate and run tasks:
```bash
poetry run python -V
poetry run task lint
poetry run task test
poetry run task run
```

### Docker (optional)
Use this minimal image for running the API without Poetry locally.
```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY neural_recommendation/pyproject.toml ./
RUN pip install --no-cache-dir poetry==1.8.3 \
    && poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi
COPY neural_recommendation/ ./neural_recommendation/
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "neural_recommendation.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t recommendation-api -f neural_recommendation/Dockerfile .
docker run --rm -p 8000:8000 recommendation-api
```

### Notes
- The application wiring uses `task run` for development. For production, prefer running Uvicorn or Gunicorn directly as shown in the Docker command.
- If you add environment-specific settings, export them before running tasks (e.g., `export LOG_LEVEL=DEBUG`).

