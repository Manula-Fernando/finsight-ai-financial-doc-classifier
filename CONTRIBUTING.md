# Contributing

Thanks for your interest in improving FinSight AI.

## Development Setup
1. Clone repo and create virtual env.
2. Install deps: `pip install -r requirements.txt`.
3. (Optional) Install Tesseract if using that backend.
4. Run ingestion (if needed): `python -m src.ingest_kagglehub`.
5. Train: `python -m src.train --classes auto`.
6. Launch app: `streamlit run app/streamlit_app.py`.

## Pull Requests
- Create a feature branch: `feat/<short-desc>`.
- Keep changes small & focused.
- Add/update documentation and tests where relevant.
- Ensure linting / type checks pass (if configured).

## Commit Messages
Format: `<type>: <short summary>`
Types: feat, fix, docs, refactor, perf, chore

## Issue Reporting
Include:
- Environment (OS, Python, relevant package versions)
- Steps to reproduce
- Expected vs actual behavior
- Screenshots / logs if helpful

## Code Style
- Favor readability over cleverness.
- Keep modules focused (classification vs OCR vs app UI).

## Licensing
By contributing you agree your contributions are licensed under the MIT License of this repository.

Thanks!
