# DIGIT-OPS RAG v3

RAG Knowledge Base untuk Manual Book PLTU — Hybrid Text + Vision OCR.

## Deploy
1. Push repo ini ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io) → New app
3. Pilih repo → **Main file path**: `app.py`
4. Buka **Settings → Secrets**, paste:

```toml
[anthropic]
api_key = "sk-ant-XXXXXXXX"

[datalab]
api_key = "dl-XXXXXXXX"
```

5. Klik Deploy ✅
