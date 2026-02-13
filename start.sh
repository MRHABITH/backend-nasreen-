#!/bin/bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm || true
uvicorn app.main:app --host 0.0.0.0 --port 10000
