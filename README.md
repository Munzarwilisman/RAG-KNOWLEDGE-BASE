# 📚 DIGIT-OPS Knowledge Base

> RAG Knowledge Base untuk Manual Book, Jurnal, dan Referensi Teknis PLTU  
> Bilingual ID + EN · Claude Vision OCR · Supabase PostgreSQL

## 🚀 Deploy ke Streamlit Cloud

### 1. Push ke GitHub
```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/USERNAME/REPO.git
git push -u origin main
```

### 2. Deploy
- Buka [share.streamlit.io](https://share.streamlit.io) → New app
- Pilih repo → **Main file path**: `app.py` → Deploy

### 3. Set Secrets
Streamlit Cloud → Settings → Secrets:
```toml
DATABASE_URL = "postgresql://postgres.xxx:PASSWORD@aws-xxx.pooler.supabase.com:6543/postgres"

[anthropic]
api_key = "sk-ant-..."
```

## 📁 Struktur File
```
├── app.py              # Aplikasi utama
├── requirements.txt    # Dependencies
├── .gitignore
├── .streamlit/
│   └── config.toml     # Tema & konfigurasi
└── rag_index/          # Index lokal (auto-generated)
```

## 🗄️ Setup Supabase
1. [supabase.com](https://supabase.com) → New project
2. Settings ⚙️ → Database → Connection string → URI → Copy
3. Paste ke Streamlit Secrets sebagai `DATABASE_URL`

Tabel dibuat otomatis saat pertama kali app dijalankan:
- `rag_files` — FAISS index binary
- `rag_documents` — metadata dokumen
- `rag_chunks` — referensi chunk per dokumen
