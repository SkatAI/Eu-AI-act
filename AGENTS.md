# Repository Guidelines

## Project Structure & Module Organization
- `streamlitapp/`: main application code. Key entry point is `streamlitapp/index.py`; retrieval and storage helpers live in `streamlitapp/retrieve.py`, `streamlitapp/streamlit_weaviate_utils.py`, and `streamlitapp/google_storage.py`.
- Root-level docs and notes: `readme.md`, `doing.md`, `next.md`, and `anomalies_numbering.md`.
- Data and dumps: `AIARAGtimedb.sql` and `aiact.pgsql` are large database artifacts; treat them as inputs, not everyday edit targets.
- Streamlit config: `.streamlit/`.
- Basic dev automation: `Makefile`.

## Build, Test, and Development Commands
- Create a virtual environment and install deps:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- Start local Weaviate via Docker:
  ```bash
  make run_weaviate
  ```
- Run the Streamlit app:
  ```bash
  streamlit run streamlitapp/index.py
  ```

## Coding Style & Naming Conventions
- Language: Python. Use 4-space indentation and keep lines readable over being overly compact.
- Naming: modules and functions use `snake_case`; constants use `UPPER_SNAKE_CASE`.
- Prefer small, focused functions in `streamlitapp/` and keep UI code in `index.py` with data/retrieval logic in helper modules.
- No formatter/linter is configured in-repo; match the surrounding style and avoid drive-by reformatting.

## Testing Guidelines
- There is no formal test suite yet. Use smoke tests:
  - Start Weaviate, then run `streamlit run streamlitapp/index.py`.
  - Exercise sidebar filters and at least one retrieval query.
- If adding tests, use `pytest` with files named `tests/test_*.py` and run:
  ```bash
  python -m pytest
  ```

## Commit & Pull Request Guidelines
- Git history favors short, imperative messages (e.g., `refactoring`, `cleanup imports`, `add amendments`).
- Aim for: `area: brief change` (example: `retrieve: tighten author filter`).
- PRs should include:
  - What changed and why.
  - How to run or verify locally.
  - Screenshots or short clips for UI changes.

## Security & Configuration Tips
- Keep secrets out of the repo. Use environment variables for API keys (for example, `OPENAI_API_KEY`) and Streamlit secrets in `.streamlit/secrets.toml`.
- Avoid committing large generated files or DB dumps unless they are intentional inputs.
