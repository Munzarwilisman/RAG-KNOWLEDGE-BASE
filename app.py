"""
╔══════════════════════════════════════════════════════════════╗
║         DIGIT-OPS RAG — Knowledge Base PLTU                 ║
║         Tim 2-10 orang | Supabase PostgreSQL                ║
║         Bilingual ID+EN | Claude Vision OCR                 ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st

# ── Page config HARUS paling atas ─────────────────────────────
st.set_page_config(
    page_title="DIGIT-OPS Knowledge Base",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os, json, hashlib, re, time, datetime, pickle, base64
import numpy as np
import pandas as pd
from pathlib import Path
from io import BytesIO
from typing import Optional

# ══════════════════════════════════════════════════════════════
# KONSTANTA
# ══════════════════════════════════════════════════════════════
APP_VERSION    = "2.0.0"
INDEX_DIR      = Path("rag_index")
INDEX_DIR.mkdir(exist_ok=True)

CHUNK_SIZE     = 1200
CHUNK_OVERLAP  = 200
TOP_K          = 8
EMBED_MODEL    = "paraphrase-multilingual-MiniLM-L12-v2"
IMG_THRESHOLD  = 120   # karakter min sebelum OCR dijalankan
OCR_DPI        = 150

DOC_CATEGORIES = [
    "Manual Book — Boiler CFB",
    "Manual Book — Steam Turbine",
    "Manual Book — Generator",
    "Manual Book — BOP & Auxiliaries",
    "Instrumen & Kontrol",
    "SOP & Prosedur Operasi",
    "Maintenance & Overhaul",
    "Jurnal & Referensi Teknis",
    "Standar (EPRI / ASME / IEC)",
    "Keselamatan (K3 & HSE)",
    "Data Sheet & Spesifikasi",
    "Laporan & Analisis",
    "Lain-lain",
]

# ══════════════════════════════════════════════════════════════
# DATABASE — SUPABASE POSTGRESQL
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_db():
    """Koneksi PostgreSQL via DATABASE_URL. Di-cache agar tidak reconnect tiap rerun."""
    try:
        import psycopg2
        import psycopg2.extras
        url  = st.secrets["DATABASE_URL"]
        conn = psycopg2.connect(url, sslmode="require")
        conn.autocommit = False
        _init_tables(conn)
        return conn
    except Exception as e:
        return None


def _init_tables(conn):
    """Buat tabel jika belum ada."""
    sql = """
    CREATE TABLE IF NOT EXISTS rag_files (
        filename    TEXT PRIMARY KEY,
        data        BYTEA NOT NULL,
        updated_at  TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS rag_documents (
        doc_id      TEXT PRIMARY KEY,
        name        TEXT NOT NULL,
        category    TEXT,
        description TEXT,
        pages       INT,
        n_chunks    INT,
        n_text      INT,
        n_ocr       INT,
        n_error     INT,
        file_hash   TEXT UNIQUE,
        ocr_engine  TEXT,
        uploaded_by TEXT,
        created_at  TIMESTAMPTZ DEFAULT NOW(),
        updated_at  TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS rag_chunks (
        id          SERIAL PRIMARY KEY,
        doc_id      TEXT NOT NULL REFERENCES rag_documents(doc_id) ON DELETE CASCADE,
        page_num    INT,
        page_type   TEXT,
        category    TEXT,
        chunk_text  TEXT NOT NULL,
        created_at  TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_chunks_doc ON rag_chunks(doc_id);
    CREATE INDEX IF NOT EXISTS idx_chunks_cat ON rag_chunks(category);
    """
    try:
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        cur.close()
    except Exception as e:
        conn.rollback()


def db_push_index(index_dir: Path) -> tuple[bool, str]:
    """Upload FAISS index files ke PostgreSQL."""
    conn = get_db()
    if not conn:
        return False, "Database tidak terhubung"
    try:
        import psycopg2
        cur = conn.cursor()
        pushed = []
        for fname in ["chunks.pkl", "faiss.index"]:
            fpath = index_dir / fname
            if fpath.exists():
                with open(fpath, "rb") as f:
                    data = f.read()
                cur.execute("""
                    INSERT INTO rag_files (filename, data, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (filename) DO UPDATE
                    SET data = EXCLUDED.data, updated_at = NOW()
                """, (fname, psycopg2.Binary(data)))
                pushed.append(fname)
        conn.commit()
        cur.close()
        return True, f"✅ Pushed: {', '.join(pushed)}"
    except Exception as e:
        conn.rollback()
        return False, f"❌ Push gagal: {e}"


def db_pull_index(index_dir: Path) -> tuple[bool, str]:
    """Download FAISS index files dari PostgreSQL."""
    conn = get_db()
    if not conn:
        return False, "Database tidak terhubung"
    try:
        cur = conn.cursor()
        pulled = []
        for fname in ["chunks.pkl", "faiss.index"]:
            cur.execute("SELECT data FROM rag_files WHERE filename = %s", (fname,))
            row = cur.fetchone()
            if row:
                index_dir.mkdir(exist_ok=True)
                with open(index_dir / fname, "wb") as f:
                    f.write(bytes(row[0]))
                pulled.append(fname)
        cur.close()
        if not pulled:
            return False, "Index belum ada (upload dokumen pertama kali)"
        return True, f"✅ Pulled: {', '.join(pulled)}"
    except Exception as e:
        return False, f"❌ Pull gagal: {e}"


def db_get_all_docs() -> list[dict]:
    """Ambil semua metadata dokumen dari PostgreSQL."""
    conn = get_db()
    if not conn:
        return []
    try:
        import psycopg2.extras
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM rag_documents ORDER BY created_at DESC")
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        return rows
    except Exception:
        return []


def db_save_doc_meta(doc_id: str, meta: dict) -> bool:
    """Simpan metadata dokumen ke PostgreSQL."""
    conn = get_db()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO rag_documents
              (doc_id, name, category, description, pages, n_chunks,
               n_text, n_ocr, n_error, file_hash, ocr_engine,
               uploaded_by, created_at, updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW(),NOW())
            ON CONFLICT (doc_id) DO UPDATE SET
              name=EXCLUDED.name, category=EXCLUDED.category,
              description=EXCLUDED.description, pages=EXCLUDED.pages,
              n_chunks=EXCLUDED.n_chunks, n_text=EXCLUDED.n_text,
              n_ocr=EXCLUDED.n_ocr, n_error=EXCLUDED.n_error,
              ocr_engine=EXCLUDED.ocr_engine,
              uploaded_by=EXCLUDED.uploaded_by,
              updated_at=NOW()
        """, (
            doc_id, meta["name"], meta["category"], meta.get("description",""),
            meta.get("pages",0), meta.get("n_chunks",0),
            meta.get("n_text",0), meta.get("n_ocr",0), meta.get("n_error",0),
            meta.get("hash",""), meta.get("ocr_engine","claude"),
            meta.get("uploaded_by",""),
        ))
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        conn.rollback()
        return False


def db_delete_doc(doc_id: str) -> bool:
    """Hapus dokumen dari PostgreSQL (CASCADE hapus chunks juga)."""
    conn = get_db()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM rag_documents WHERE doc_id = %s", (doc_id,))
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        conn.rollback()
        return False


def db_hash_exists(file_hash: str) -> bool:
    """Cek apakah file sudah pernah diupload (by hash)."""
    conn = get_db()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM rag_documents WHERE file_hash = %s", (file_hash,))
        exists = cur.fetchone() is not None
        cur.close()
        return exists
    except Exception:
        return False


def is_db_connected() -> bool:
    """Cek apakah koneksi database tersedia."""
    try:
        conn = get_db()
        if not conn:
            return False
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════
# OCR — CLAUDE VISION
# ══════════════════════════════════════════════════════════════

def ocr_claude(img_bytes: bytes, page_num: int, doc_name: str) -> str:
    """Claude Vision OCR — akurat untuk tabel teknis bilingual."""
    try:
        import anthropic
        client  = anthropic.Anthropic(api_key=st.secrets["anthropic"]["api_key"])
        img_b64 = base64.b64encode(img_bytes).decode()
        prompt  = f"""Kamu adalah OCR engine untuk dokumen teknis PLTU/Power Plant.
Dokumen: {doc_name} | Halaman: {page_num}

INSTRUKSI:
1. Ekstrak SEMUA teks dari halaman ini secara lengkap dan akurat.
2. Untuk TABEL PARAMETER — format setiap baris:
   [Nama Parameter] | [Unit] | [Nilai Normal] | [Alarm High] | [Alarm Low] | [Trip/Interlock]
   WAJIB: nama parameter HARUS ada di setiap baris, jangan pisahkan nama dari nilainya.
3. Untuk tabel clearance/alignment:
   [Komponen] | [Symbol] | [Nilai mm] | [Toleransi]
4. Teks bilingual Mandarin+Inggris → ekstrak KEDUANYA.
5. Angka teknis harus AKURAT: °C, MPa, kPa, bar, mm, rpm, MW.
6. Jika halaman kosong atau tidak ada teks, tulis: [Halaman kosong]

Ekstrak seluruh isi halaman:"""
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=3000,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": "image/jpeg", "data": img_b64}},
                {"type": "text", "text": prompt},
            ]}],
        )
        return resp.content[0].text
    except Exception as e:
        return f"[OCR Error hal {page_num}: {e}]"


# ══════════════════════════════════════════════════════════════
# PDF PROCESSING
# ══════════════════════════════════════════════════════════════

def pdf_page_count(pdf_bytes: bytes) -> int:
    for lib in ["fitz", "pypdf", "PyPDF2"]:
        try:
            if lib == "fitz":
                import fitz
                return len(fitz.open(stream=pdf_bytes, filetype="pdf"))
            elif lib == "pypdf":
                import pypdf
                return len(pypdf.PdfReader(BytesIO(pdf_bytes)).pages)
            elif lib == "PyPDF2":
                import PyPDF2
                return len(PyPDF2.PdfReader(BytesIO(pdf_bytes)).pages)
        except Exception:
            continue
    return 0


def pdf_extract_text(pdf_bytes: bytes, page_num: int) -> str:
    for lib in ["fitz", "pypdf", "PyPDF2"]:
        try:
            if lib == "fitz":
                import fitz
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                return doc[page_num - 1].get_text().strip()
            elif lib == "pypdf":
                import pypdf
                return (pypdf.PdfReader(BytesIO(pdf_bytes))
                        .pages[page_num - 1].extract_text() or "").strip()
            elif lib == "PyPDF2":
                import PyPDF2
                return (PyPDF2.PdfReader(BytesIO(pdf_bytes))
                        .pages[page_num - 1].extract_text() or "").strip()
        except Exception:
            continue
    return ""


def pdf_rasterize(pdf_bytes: bytes, page_num: int, dpi: int = OCR_DPI) -> Optional[bytes]:
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = doc[page_num - 1].get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("jpeg")
    except Exception:
        pass
    try:
        import pypdfium2 as pdfium
        doc  = pdfium.PdfDocument(pdf_bytes)
        page = doc[page_num - 1]
        bm   = page.render(scale=dpi / 72)
        buf  = BytesIO()
        bm.to_pil().save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception:
        pass
    return None


def process_pdf(pdf_bytes: bytes, filename: str, category: str,
                progress_cb=None) -> list[dict]:
    """Proses PDF dengan hybrid text + OCR."""
    pages    = []
    n_pages  = pdf_page_count(pdf_bytes)
    if n_pages == 0:
        return pages

    n_text = n_ocr = n_err = 0
    for pg in range(1, n_pages + 1):
        if progress_cb:
            progress_cb(pg, n_pages, n_text, n_ocr)

        text = pdf_extract_text(pdf_bytes, pg)

        if len(text) >= IMG_THRESHOLD:
            pages.append({"text": text, "page": pg,
                          "type": "text", "source": filename, "category": category})
            n_text += 1
        else:
            img = pdf_rasterize(pdf_bytes, pg)
            if img:
                ocr_text = ocr_claude(img, pg, filename)
                if "[OCR Error" not in ocr_text:
                    combined = (text + "\n\n" + ocr_text).strip() if text else ocr_text
                    pages.append({"text": combined, "page": pg,
                                  "type": "claude_ocr", "source": filename, "category": category})
                    n_ocr += 1
                else:
                    if text:
                        pages.append({"text": text, "page": pg,
                                      "type": "text", "source": filename, "category": category})
                        n_text += 1
                    else:
                        n_err += 1
            else:
                if text:
                    pages.append({"text": text, "page": pg,
                                  "type": "text", "source": filename, "category": category})
                    n_text += 1
                else:
                    n_err += 1
    return pages


def process_docx(file_bytes: bytes, filename: str, category: str) -> list[dict]:
    pages = []
    try:
        from docx import Document
        doc   = Document(BytesIO(file_bytes))
        paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        # Ambil tabel juga
        for tbl in doc.tables:
            for row in tbl.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    paras.append(" | ".join(cells))
        # Chunk per 500 words
        words = " ".join(paras).split()
        for i in range(0, len(words), 500):
            chunk = " ".join(words[i:i + 500])
            if chunk:
                pages.append({"text": chunk, "page": i // 500 + 1,
                               "type": "text", "source": filename, "category": category})
    except Exception as e:
        pages.append({"text": f"Error: {e}", "page": 1,
                      "type": "error", "source": filename, "category": category})
    return pages


def process_excel(file_bytes: bytes, filename: str, category: str) -> list[dict]:
    pages = []
    try:
        ext = Path(filename).suffix.lower()
        df  = (pd.read_csv(BytesIO(file_bytes)) if ext == ".csv"
               else pd.read_excel(BytesIO(file_bytes)))
        df  = df.fillna("")
        # Convert setiap 30 baris ke text
        for i in range(0, len(df), 30):
            chunk = df.iloc[i:i + 30].to_string(index=False)
            if chunk.strip():
                pages.append({"text": chunk, "page": i // 30 + 1,
                               "type": "text", "source": filename, "category": category})
    except Exception as e:
        pages.append({"text": f"Error: {e}", "page": 1,
                      "type": "error", "source": filename, "category": category})
    return pages


def process_image(file_bytes: bytes, filename: str, category: str) -> list[dict]:
    """Proses file gambar (JPG/PNG) dengan Claude OCR."""
    ocr_text = ocr_claude(file_bytes, 1, filename)
    if "[OCR Error" not in ocr_text and ocr_text.strip():
        return [{"text": ocr_text, "page": 1,
                 "type": "claude_ocr", "source": filename, "category": category}]
    return []


# ══════════════════════════════════════════════════════════════
# CHUNKING — CERDAS UNTUK TABEL TEKNIS
# ══════════════════════════════════════════════════════════════

def is_table_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if "|" in s:
        return True
    parts = [p.strip() for p in re.split(r"\s{2,}|\t", s) if p.strip()]
    return len(parts) >= 2


def smart_chunk(text: str, chunk_size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Chunking yang mempertahankan integritas tabel parameter."""
    text  = re.sub(r"\n{3,}", "\n\n", text).strip()
    lines = text.split("\n")

    # Deteksi konten tabel
    table_lines = sum(1 for l in lines if is_table_line(l))
    is_table    = table_lines >= 3 or text.count("|") > 5

    if is_table:
        # Pisahkan header dari data rows
        headers   = []
        data_rows = []
        found_data = False
        header_kws = {"parameter", "unit", "normal", "alarm", "interlock",
                       "satuan", "nilai", "gauging", "measuring", "symbol",
                       "clearance", "remark", "limit", "upper", "lower"}

        for ln in lines:
            ls = ln.strip().lower()
            if not found_data and any(kw in ls for kw in header_kws):
                headers.append(ln)
            else:
                found_data = True
                if ln.strip():
                    data_rows.append(ln)

        if not headers and data_rows:
            headers   = data_rows[:2]
            data_rows = data_rows[2:]

        header_str = "\n".join(headers)
        chunks, cur_rows, cur_len = [], [], len(header_str) + 1

        for row in data_rows:
            row_len = len(row) + 1
            if cur_len + row_len > chunk_size and cur_rows:
                content = (header_str + "\n" + "\n".join(cur_rows)).strip()
                if len(content) > 50:
                    chunks.append(content)
                cur_rows = cur_rows[-3:] + [row]
                cur_len  = len(header_str) + sum(len(r) + 1 for r in cur_rows)
            else:
                cur_rows.append(row)
                cur_len += row_len

        if cur_rows:
            content = (header_str + "\n" + "\n".join(cur_rows)).strip()
            if len(content) > 50:
                chunks.append(content)
        return chunks if chunks else [text]

    # Teks biasa — split per kalimat dengan overlap
    sents  = re.split(r"(?<=[.!?\n])\s+", text)
    chunks, cur = [], ""
    for sent in sents:
        if len(cur) + len(sent) + 1 > chunk_size and cur:
            chunks.append(cur.strip())
            words = cur.split()
            cur   = " ".join(words[max(0, len(words) - overlap // 5):]) + " " + sent
        else:
            cur = (cur + " " + sent).strip()
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in chunks if len(c) > 60]


# ══════════════════════════════════════════════════════════════
# RAG ENGINE
# ══════════════════════════════════════════════════════════════

class RAGEngine:
    def __init__(self, index_dir: Path = INDEX_DIR):
        self.index_dir    = index_dir
        self._embed       = None
        self._faiss       = None
        self._chunks: list[dict] = []
        self._load()

    # ── Embedding model ──────────────────────────────────────
    @property
    def embed(self):
        if self._embed is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embed = SentenceTransformer(EMBED_MODEL)
            except ImportError:
                pass
        return self._embed

    # ── Persistensi lokal ────────────────────────────────────
    def _save_local(self):
        import faiss
        with open(self.index_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        if self._faiss is not None:
            faiss.write_index(self._faiss, str(self.index_dir / "faiss.index"))

    def _load(self):
        """Load dari lokal, atau pull dari DB jika lokal kosong."""
        # Coba pull dari database jika lokal kosong
        if not (self.index_dir / "chunks.pkl").exists() and is_db_connected():
            db_pull_index(self.index_dir)
        self._load_local()

    def _load_local(self):
        try:
            import faiss
            cp = self.index_dir / "chunks.pkl"
            fp = self.index_dir / "faiss.index"
            if cp.exists():
                with open(cp, "rb") as f:
                    self._chunks = pickle.load(f)
            if fp.exists():
                self._faiss = faiss.read_index(str(fp))
        except Exception:
            self._chunks = []
            self._faiss  = None

    def _save(self):
        """Simpan lokal + push ke database."""
        self._save_local()
        if is_db_connected():
            db_push_index(self.index_dir)

    # ── Embed dan tambahkan ──────────────────────────────────
    def _embed_add(self, new_chunks: list[dict]):
        import faiss
        if not new_chunks or self.embed is None:
            return
        texts  = [c["text"] for c in new_chunks]
        vecs   = self.embed.encode(texts, batch_size=32, show_progress_bar=False)
        vecs   = np.array(vecs, dtype=np.float32)
        norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vecs  /= norms
        if self._faiss is None:
            self._faiss = faiss.IndexFlatIP(vecs.shape[1])
        self._faiss.add(vecs)
        self._chunks.extend(new_chunks)

    # ── Add dokumen ──────────────────────────────────────────
    def add_document(self, file_bytes: bytes, filename: str, category: str,
                     description: str = "", uploaded_by: str = "",
                     progress_cb=None) -> dict:
        if self.embed is None:
            return {"ok": False, "msg": "Embedding model tidak tersedia"}

        file_hash = hashlib.md5(file_bytes).hexdigest()
        if db_hash_exists(file_hash):
            return {"ok": False, "msg": f"'{filename}' sudah ada di knowledge base"}

        doc_id = "doc_" + file_hash[:12]
        ext    = Path(filename).suffix.lower()

        # Proses sesuai tipe file
        if ext == ".pdf":
            pages = process_pdf(file_bytes, filename, category, progress_cb)
        elif ext in (".docx", ".doc"):
            pages = process_docx(file_bytes, filename, category)
        elif ext in (".xlsx", ".xls", ".csv"):
            pages = process_excel(file_bytes, filename, category)
        elif ext in (".jpg", ".jpeg", ".png"):
            pages = process_image(file_bytes, filename, category)
        elif ext == ".txt":
            text  = file_bytes.decode("utf-8", errors="ignore")
            words = text.split()
            pages = [{"text": " ".join(words[i:i+500]), "page": i//500+1,
                      "type": "text", "source": filename, "category": category}
                     for i in range(0, len(words), 500) if words[i:i+500]]
        else:
            return {"ok": False, "msg": f"Format file tidak didukung: {ext}"}

        if not pages:
            return {"ok": False, "msg": "Tidak ada konten yang bisa diekstrak"}

        # Chunking
        new_chunks = []
        for pg in pages:
            if pg["type"] == "error":
                continue
            for ct in smart_chunk(pg["text"]):
                new_chunks.append({
                    "text":      ct,
                    "source":    filename,
                    "page":      pg["page"],
                    "page_type": pg["type"],
                    "category":  category,
                    "doc_id":    doc_id,
                })

        if not new_chunks:
            return {"ok": False, "msg": "Tidak ada chunk yang terbentuk"}

        self._embed_add(new_chunks)
        self._save()

        n_t = sum(1 for p in pages if p["type"] == "text")
        n_o = sum(1 for p in pages if p["type"] == "claude_ocr")
        n_e = sum(1 for p in pages if p["type"] == "error")

        meta = {
            "name": filename, "category": category, "description": description,
            "pages": len(pages), "n_chunks": len(new_chunks),
            "n_text": n_t, "n_ocr": n_o, "n_error": n_e,
            "hash": file_hash, "ocr_engine": "claude_vision",
            "uploaded_by": uploaded_by,
        }
        db_save_doc_meta(doc_id, meta)

        return {
            "ok": True, "doc_id": doc_id,
            "n_chunks": len(new_chunks), "n_text": n_t,
            "n_ocr": n_o, "n_error": n_e,
            "msg": (f"✅ {len(new_chunks)} chunks dari {len(pages)} halaman "
                    f"({n_t} teks + {n_o} OCR + {n_e} error)"),
        }

    # ── Delete dokumen ───────────────────────────────────────
    def delete_document(self, doc_id: str) -> bool:
        import faiss
        keep = [c for c in self._chunks if c.get("doc_id") != doc_id]
        db_delete_doc(doc_id)
        if not keep:
            self._chunks = []
            self._faiss  = None
            self._save()
            return True
        if self.embed is None:
            return False
        texts  = [c["text"] for c in keep]
        vecs   = self.embed.encode(texts, batch_size=32, show_progress_bar=False)
        vecs   = np.array(vecs, dtype=np.float32)
        norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vecs  /= norms
        new_idx = faiss.IndexFlatIP(vecs.shape[1])
        new_idx.add(vecs)
        self._chunks = keep
        self._faiss  = new_idx
        self._save()
        return True

    # ── Retrieval ────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = TOP_K,
                 category: Optional[str] = None) -> list[dict]:
        if not self._chunks or self._faiss is None or self.embed is None:
            return []

        # Query expansion dengan sinonim teknis
        queries = _expand_query(query)
        q_vecs  = self.embed.encode(queries, show_progress_bar=False)
        q_vecs  = np.array(q_vecs, dtype=np.float32)
        norms   = np.linalg.norm(q_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        q_vecs /= norms

        k          = min(top_k * 8, len(self._chunks))
        score_map: dict[int, float] = {}
        for qv in q_vecs:
            scores, idxs = self._faiss.search(qv.reshape(1, -1), k)
            for sc, idx in zip(scores[0], idxs[0]):
                if idx >= 0:
                    score_map[idx] = max(score_map.get(idx, 0.0), float(sc))

        results, seen = [], set()
        for idx, score in sorted(score_map.items(), key=lambda x: x[1], reverse=True):
            if idx >= len(self._chunks):
                continue
            c = self._chunks[idx]
            if category and category != "Semua" and c.get("category", "") != category:
                continue
            key = c["text"][:80]
            if key in seen:
                continue
            seen.add(key)
            results.append({**c, "score": score})
            if len(results) >= top_k:
                break
        return results

    def keyword_search(self, query: str, top_k: int = TOP_K,
                       category: Optional[str] = None) -> list[dict]:
        """Pencarian keyword eksak sebagai fallback."""
        keywords = [w.lower() for w in query.split() if len(w) > 2]
        if not keywords:
            return []
        results = []
        for c in self._chunks:
            if category and category != "Semua" and c.get("category", "") != category:
                continue
            tl    = c["text"].lower()
            score = sum(1 for kw in keywords if kw in tl) / len(keywords)
            if score > 0:
                results.append({**c, "score": score})
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    # ── Properties ───────────────────────────────────────────
    @property
    def n_chunks(self):
        return len(self._chunks)

    @property
    def n_ocr_chunks(self):
        return sum(1 for c in self._chunks if c.get("page_type") == "claude_ocr")

    @property
    def categories(self):
        return sorted(set(c.get("category", "") for c in self._chunks if c.get("category")))


# ══════════════════════════════════════════════════════════════
# QUERY EXPANSION — SINONIM TEKNIS
# ══════════════════════════════════════════════════════════════

_SYNONYMS = {
    "thrust pad":    ["推力瓦", "thrust bearing pad", "pad temperature", "瓦温"],
    "bearing":       ["bantalan", "瓦", "bearing temperature", "bearing pad"],
    "lube oil":      ["lubricating oil", "minyak pelumas", "润滑油", "oil pressure"],
    "clearance":     ["celah", "間隙", "tolerance", "fit", "gap"],
    "vibration":     ["vibrasi", "振动", "amplitude", "getaran"],
    "alignment":     ["senter", "对中", "centering", "kopling"],
    "alarm":         ["报警", "high alarm", "low alarm", "batas atas", "batas bawah"],
    "trip":          ["interlock", "跳机", "shutdown", "proteksi", "protection"],
    "temperature":   ["suhu", "temperatur", "temp", "°C", "温度"],
    "pressure":      ["tekanan", "压力", "MPa", "kPa", "bar"],
    "normal":        ["nilai normal", "normal value", "rated", "设计值"],
    "axial":         ["axial displacement", "轴向位移", "thrust position"],
}


def _expand_query(query: str) -> list[str]:
    queries = [query]
    ql = query.lower()
    for key, syns in _SYNONYMS.items():
        if key in ql:
            for syn in syns[:2]:
                exp = re.sub(re.escape(key), syn, ql, flags=re.IGNORECASE)
                if exp not in queries:
                    queries.append(exp)
    # Bersihkan kata tanya
    clean = re.sub(r"\b(berapa|apa|bagaimana|jelaskan|sebutkan|apakah)\b", "", ql).strip()
    if clean and clean not in queries:
        queries.append(clean)
    return queries[:4]


# ══════════════════════════════════════════════════════════════
# QUERY FUNCTION
# ══════════════════════════════════════════════════════════════

def ask_knowledge_base(question: str, engine: RAGEngine,
                       category: Optional[str] = None,
                       context: str = "") -> str:
    import anthropic

    # Semantic search
    chunks = engine.retrieve(question, top_k=TOP_K, category=category)

    # Keyword fallback jika skor rendah
    if not chunks or chunks[0]["score"] < 0.25:
        kw = engine.keyword_search(question, top_k=5, category=category)
        seen = {c["text"][:80] for c in chunks}
        for r in kw:
            if r["text"][:80] not in seen:
                chunks.append(r)
                seen.add(r["text"][:80])
        chunks = chunks[:TOP_K]

    if not chunks:
        return ("⚠️ Tidak ditemukan referensi relevan di knowledge base.\n\n"
                "Pastikan dokumen sudah diupload dan dicek di tab **📑 Dokumen**.")

    # Susun konteks
    ctx_parts = []
    for i, c in enumerate(chunks):
        tag = " [OCR]" if c.get("page_type") == "claude_ocr" else ""
        src = f"[{c['source']} | Hal.{c['page']}{tag} | {c['category']}]"
        ctx_parts.append(f"REFERENSI {i+1} {src}:\n{c['text']}")
    ctx_str = "\n\n---\n\n".join(ctx_parts)

    is_table_req = any(kw in question.lower() for kw in
        ["tunjukkan", "tampilkan", "semua", "seluruh", "daftar",
         "tabel", "list", "all parameter", "batas parameter"])

    system = """Kamu adalah AI Engineer senior spesialis PLTU (Pembangkit Listrik Tenaga Uap).
Keahlian: Boiler CFB, Steam Turbine, Generator, sistem kontrol, maintenance.

ATURAN MENJAWAB:
1. Jawab HANYA berdasarkan REFERENSI yang diberikan.
2. Jika tidak ada di referensi → tulis "Tidak ditemukan di dokumen yang tersedia."
3. Sebutkan sumber (nama dokumen + halaman) di akhir jawaban.
4. Satuan teknis WAJIB akurat: MPa, kPa, °C, bar, rpm, mm, kN·m, MW.

FORMAT TABEL PARAMETER (gunakan selalu untuk data parameter):
| No | Parameter | Satuan | Normal | Alarm High | Alarm Low | Trip | Keterangan |
|----|-----------|--------|--------|------------|-----------|------|------------|

ATURAN TABEL:
- Tampilkan SEMUA parameter dari referensi, jangan ada yang terlewat.
- Nilai "-" jika tidak disebutkan di referensi.
- Urutkan: Temperatur → Tekanan → Level → Flow → Lainnya."""

    user_msg = f"PERTANYAAN: {question}\n\n"
    if context:
        user_msg += f"DATA OPERASIONAL SAAT INI:\n{context}\n\n"
    if is_table_req:
        user_msg += "INSTRUKSI: Tampilkan dalam format tabel lengkap, kumpulkan dari SEMUA referensi.\n\n"
    user_msg += f"REFERENSI DOKUMEN:\n{ctx_str}"

    try:
        client   = anthropic.Anthropic(api_key=st.secrets["anthropic"]["api_key"])
        max_tok  = 4000 if is_table_req else 2000
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=max_tok,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        answer  = response.content[0].text
        sources = list(dict.fromkeys(f"{c['source']} (Hal.{c['page']})" for c in chunks))
        return answer + f"\n\n---\n📚 **Sumber:** {' · '.join(sources[:6])}"
    except Exception as e:
        return f"❌ Error API: {e}"


# ══════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════

# ── Load engine (cached) ───────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_engine() -> RAGEngine:
    return RAGEngine()


def main():
    engine = load_engine()
    db_ok  = is_db_connected()

    # ── SIDEBAR ───────────────────────────────────────────────
    with st.sidebar:
        # Logo
        try:
            st.image("https://gajiloker.com/wp-content/uploads/2024/02/"
                     "Gaji-PT-PLN-Nusantara-Power-Services.jpg", use_container_width=True)
        except Exception:
            pass

        st.markdown("## 📚 DIGIT-OPS")
        st.caption(f"Knowledge Base v{APP_VERSION}")
        st.divider()

        # DB status
        if db_ok:
            st.success("🟢 Database terhubung")
        else:
            st.error("🔴 Database tidak terhubung")
            st.caption("Set DATABASE_URL di Secrets")
        st.divider()

        # Stats dari DB
        docs = db_get_all_docs()
        col1, col2 = st.columns(2)
        col1.metric("Dokumen",  len(docs))
        col2.metric("Chunks",   engine.n_chunks)
        col3, col4 = st.columns(2)
        col3.metric("OCR",      engine.n_ocr_chunks)
        col4.metric("Teks",     engine.n_chunks - engine.n_ocr_chunks)
        st.divider()

        # Filter
        cat_opts = ["Semua"] + engine.categories
        sel_cat  = st.selectbox("🗂 Filter Kategori:", cat_opts, key="sel_cat")
        st.divider()

        # DB Sync
        st.markdown("**☁️ Sinkronisasi**")
        c1, c2 = st.columns(2)
        if c1.button("⬆ Push", key="btn_push", use_container_width=True,
                     help="Upload index ke database"):
            with st.spinner("Uploading..."):
                ok, msg = db_push_index(INDEX_DIR)
            st.toast(msg)
        if c2.button("⬇ Pull", key="btn_pull", use_container_width=True,
                     help="Download index dari database"):
            with st.spinner("Downloading..."):
                ok, msg = db_pull_index(INDEX_DIR)
            st.toast(msg)
            if ok:
                st.cache_resource.clear()
                st.rerun()
        st.divider()

        # User info
        username = st.text_input("👤 Nama Anda:", key="username",
                                  placeholder="contoh: Ahmad")

    # ── HEADER ────────────────────────────────────────────────
    st.title("📚 DIGIT-OPS Knowledge Base")
    st.caption("Manual Book · Jurnal · Referensi Teknis PLTU — Bilingual ID + EN")

    if not db_ok:
        st.warning("""⚠️ **Database belum terhubung.** Tambahkan di Streamlit Cloud → Settings → Secrets:
```toml
DATABASE_URL = "postgresql://postgres.xxx:PASSWORD@aws-xxx.pooler.supabase.com:6543/postgres"
[anthropic]
api_key = "sk-ant-..."
```""")

    tab_chat, tab_upload, tab_search, tab_docs, tab_setup = st.tabs([
        "💬 Tanya AI",
        "📤 Upload Dokumen",
        "🔍 Pencarian",
        "📑 Manajemen Dokumen",
        "⚙️ Setup & Info",
    ])

    # ══════════════════════════════════════════════════════════
    # TAB 1 — CHAT
    # ══════════════════════════════════════════════════════════
    with tab_chat:
        if engine.n_chunks == 0:
            st.info("📢 Belum ada dokumen. Upload manual book di tab **📤 Upload Dokumen**.")

        # Chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input
        if prompt := st.chat_input("Tanyakan seputar manual book, parameter, prosedur..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Mencari referensi..."):
                    cat = None if sel_cat == "Semua" else sel_cat
                    try:
                        answer = ask_knowledge_base(prompt, engine, category=cat)
                    except Exception as e:
                        answer = f"❌ Error: {e}"
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        if st.session_state.messages:
            if st.button("🗑 Hapus Riwayat", key="clr"):
                st.session_state.messages = []
                st.rerun()

        st.divider()

        # Quick buttons
        st.subheader("⚡ Shortcut Parameter")
        _BOILER_Q = ("Tampilkan SEMUA batas parameter operasi Boiler CFB dalam tabel lengkap: "
                     "nama parameter, nilai normal, alarm high, alarm low, trip/interlock, satuan. "
                     "Sertakan: bed temperature, furnace pressure, steam pressure, steam temperature, "
                     "drum level, ID/FD/PA fan, coal feeder, flue gas, oxygen.")
        _TURBIN_Q = ("Tampilkan SEMUA batas parameter operasi Steam Turbine dalam tabel lengkap: "
                     "nama parameter, nilai normal, alarm high, alarm low, trip, satuan. "
                     "Sertakan: thrust pad temp, bearing temp, lube oil pressure/temp, "
                     "axial displacement, vibration, steam pressure/temp, speed.")
        _GENSET_Q = ("Tampilkan SEMUA batas parameter operasi Generator dalam tabel lengkap: "
                     "nama parameter, nilai normal, alarm high, alarm low, trip, satuan.")

        gc1, gc2, gc3 = st.columns(3)
        if gc1.button("🔥 Parameter Boiler", use_container_width=True, key="q_boiler"):
            with st.spinner("Mengumpulkan..."):
                cat = None if sel_cat == "Semua" else sel_cat
                ans = ask_knowledge_base(_BOILER_Q, engine, category=cat)
            st.session_state.messages.extend([
                {"role": "user",      "content": "📊 Semua Parameter Boiler CFB"},
                {"role": "assistant", "content": ans},
            ])
            st.rerun()

        if gc2.button("⚙️ Parameter Turbin", use_container_width=True, key="q_turbin"):
            with st.spinner("Mengumpulkan..."):
                cat = None if sel_cat == "Semua" else sel_cat
                ans = ask_knowledge_base(_TURBIN_Q, engine, category=cat)
            st.session_state.messages.extend([
                {"role": "user",      "content": "📊 Semua Parameter Steam Turbine"},
                {"role": "assistant", "content": ans},
            ])
            st.rerun()

        if gc3.button("⚡ Parameter Generator", use_container_width=True, key="q_gen"):
            with st.spinner("Mengumpulkan..."):
                cat = None if sel_cat == "Semua" else sel_cat
                ans = ask_knowledge_base(_GENSET_Q, engine, category=cat)
            st.session_state.messages.extend([
                {"role": "user",      "content": "📊 Semua Parameter Generator"},
                {"role": "assistant", "content": ans},
            ])
            st.rerun()

        # Export chat
        if st.session_state.messages:
            st.divider()
            chat_export = "\n\n".join(
                f"{'USER' if m['role']=='user' else 'AI'}: {m['content']}"
                for m in st.session_state.messages
            )
            st.download_button("📥 Export Riwayat Chat (.txt)",
                                data=chat_export,
                                file_name=f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                mime="text/plain", key="dl_chat")

    # ══════════════════════════════════════════════════════════
    # TAB 2 — UPLOAD
    # ══════════════════════════════════════════════════════════
    with tab_upload:
        st.subheader("📤 Upload Dokumen ke Knowledge Base")

        # Cek API key
        try:
            _api_ok = bool(st.secrets["anthropic"]["api_key"])
        except Exception:
            _api_ok = False

        if not _api_ok:
            st.error("❌ Anthropic API key tidak ditemukan. Tambahkan di Secrets.")
            st.stop()

        st.info(
            "**Format didukung:** PDF · DOCX · XLSX · CSV · TXT · JPG · PNG\n\n"
            "Halaman teks → ekstraksi langsung (gratis). "
            "Halaman gambar/tabel → Claude Vision OCR (akurat untuk tabel bilingual)."
        )

        # Form
        uploaded_files = st.file_uploader(
            "Pilih satu atau beberapa file:",
            accept_multiple_files=True,
            type=["pdf", "docx", "doc", "xlsx", "xls", "csv", "txt", "jpg", "jpeg", "png"],
            key="file_uploader",
        )

        doc_category = st.selectbox("Kategori Dokumen:", DOC_CATEGORIES, key="doc_cat")
        doc_desc     = st.text_input("Deskripsi (opsional):", key="doc_desc",
                                      placeholder="contoh: Steam Turbine N27.5 OEM Manual 2019")

        if uploaded_files:
            st.caption(f"{len(uploaded_files)} file dipilih")
            if st.button("📥 Proses & Simpan ke Knowledge Base",
                          key="btn_upload", use_container_width=True, type="primary"):
                uname = st.session_state.get("username", "")
                for uf in uploaded_files:
                    with st.container(border=True):
                        st.markdown(f"**📄 {uf.name}**")
                        fb  = uf.read()
                        ext = Path(uf.name).suffix.lower()

                        if ext == ".pdf":
                            prog    = st.progress(0)
                            cap_ph  = st.empty()
                            log_ph  = st.empty()
                            log_buf = []

                            def _cb(pg, tot, nt, nv,
                                    _p=prog, _c=cap_ph, _l=log_ph, _b=log_buf):
                                _p.progress(pg / tot)
                                _c.caption(f"Halaman {pg}/{tot} — teks: {nt} | OCR: {nv}")
                                if nv > 0:
                                    _b.append(f"  Hal {pg}: Claude Vision OCR ✓")
                                    _l.code("\n".join(_b[-5:]), language=None)

                            with st.spinner("Memproses..."):
                                res = engine.add_document(
                                    fb, uf.name, doc_category,
                                    description=doc_desc, uploaded_by=uname,
                                    progress_cb=_cb,
                                )
                            prog.empty(); cap_ph.empty(); log_ph.empty()
                        else:
                            with st.spinner("Memproses..."):
                                res = engine.add_document(
                                    fb, uf.name, doc_category,
                                    description=doc_desc, uploaded_by=uname,
                                )

                        if res["ok"]:
                            st.success(res["msg"])
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Chunks",    res["n_chunks"])
                            m2.metric("Hal Teks",  res["n_text"])
                            m3.metric("Hal OCR",   res["n_ocr"])
                            m4.metric("Error",     res["n_error"])
                        else:
                            st.warning(res["msg"])

                st.success("✅ Selesai! Refresh halaman jika perlu.")
                st.cache_resource.clear()

    # ══════════════════════════════════════════════════════════
    # TAB 3 — PENCARIAN
    # ══════════════════════════════════════════════════════════
    with tab_search:
        st.subheader("🔍 Pencarian Dokumen")

        if engine.n_chunks == 0:
            st.info("Upload dokumen terlebih dahulu.")
        else:
            col_q, col_c = st.columns([3, 1])
            query    = col_q.text_input("Kata kunci / pertanyaan:", key="search_q",
                                         placeholder="contoh: thrust pad temperature alarm")
            cat_filt = col_c.selectbox("Kategori:", ["Semua"] + engine.categories,
                                        key="search_cat")

            col_m, col_k, col_n = st.columns([2, 1, 1])
            mode   = col_m.radio("Mode:", ["Semantik (AI)", "Keyword Eksak"],
                                  key="search_mode", horizontal=True)
            top_n  = col_k.slider("Jumlah hasil:", 3, 20, 8, key="top_n")
            do_src = col_n.checkbox("Tampilkan teks chunk", key="show_src")

            if st.button("🔍 Cari", key="btn_search",
                          use_container_width=True, type="primary") and query:
                cat = None if cat_filt == "Semua" else cat_filt
                with st.spinner("Mencari..."):
                    if mode == "Keyword Eksak":
                        results = engine.keyword_search(query, top_k=top_n, category=cat)
                    else:
                        results = engine.retrieve(query, top_k=top_n, category=cat)
                st.session_state["search_results"] = results
                st.session_state["search_query"]   = query

            results = st.session_state.get("search_results", [])
            if results:
                q_label = st.session_state.get("search_query", "")
                st.caption(f"**{len(results)} hasil** untuk: _{q_label}_")
                st.divider()

                for i, r in enumerate(results):
                    pct  = min(100, int(r["score"] * 100))
                    pt   = r.get("page_type", "text")
                    tag  = "🤖 OCR" if pt == "claude_ocr" else "📝 Teks"
                    col  = "🟢" if pct > 70 else "🟡" if pct > 45 else "🔴"

                    with st.container(border=True):
                        c1, c2, c3 = st.columns([3, 1, 1])
                        c1.markdown(f"**{r['source']}** — Hal. {r['page']}")
                        c2.caption(f"{tag} | {r['category'][:25]}")
                        c3.markdown(f"{col} **{pct}%**")
                        if do_src:
                            st.text(r["text"][:400] + ("…" if len(r["text"]) > 400 else ""))

                # Export hasil pencarian
                if st.button("📥 Export Hasil Pencarian (.csv)", key="exp_search"):
                    df_exp = pd.DataFrame([{
                        "Dokumen":   r["source"],
                        "Halaman":   r["page"],
                        "Kategori":  r["category"],
                        "Tipe":      r.get("page_type","text"),
                        "Relevansi": f"{min(100,int(r['score']*100))}%",
                        "Teks":      r["text"][:300],
                    } for r in results])
                    st.download_button("⬇️ Download CSV", data=df_exp.to_csv(index=False),
                                        file_name="hasil_pencarian.csv", mime="text/csv",
                                        key="dl_search")

    # ══════════════════════════════════════════════════════════
    # TAB 4 — MANAJEMEN DOKUMEN
    # ══════════════════════════════════════════════════════════
    with tab_docs:
        st.subheader("📑 Manajemen Dokumen")

        docs = db_get_all_docs()
        if not docs:
            st.info("Belum ada dokumen di knowledge base.")
        else:
            # Summary stats
            total_chunks = engine.n_chunks
            total_pages  = sum(d.get("pages", 0) for d in docs)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Dokumen", len(docs))
            c2.metric("Total Halaman", total_pages)
            c3.metric("Total Chunks",  total_chunks)
            c4.metric("Kategori",      len(set(d.get("category","") for d in docs)))
            st.divider()

            # Filter & search
            col_s, col_c = st.columns([2, 1])
            doc_search = col_s.text_input("Cari dokumen:", key="doc_search",
                                           placeholder="nama file atau deskripsi...")
            cat_filter = col_c.selectbox("Filter kategori:",
                                          ["Semua"] + sorted(set(d.get("category","") for d in docs)),
                                          key="doc_cat_filt")

            # Filter
            filtered = docs
            if doc_search:
                filtered = [d for d in filtered if
                            doc_search.lower() in d.get("name","").lower() or
                            doc_search.lower() in d.get("description","").lower()]
            if cat_filter != "Semua":
                filtered = [d for d in filtered if d.get("category") == cat_filter]

            st.caption(f"Menampilkan {len(filtered)} dari {len(docs)} dokumen")
            st.divider()

            # Daftar dokumen
            for doc in filtered:
                doc_id = doc["doc_id"]
                with st.container(border=True):
                    col_i, col_d = st.columns([5, 1])
                    with col_i:
                        st.markdown(f"**{doc['name']}**")
                        col_a, col_b, col_c2 = st.columns(3)
                        col_a.caption(f"📁 {doc.get('category','—')}")
                        col_b.caption(f"📃 {doc.get('pages',0)} hal | 🧩 {doc.get('n_chunks',0)} chunks")
                        col_c2.caption(f"👤 {doc.get('uploaded_by','—')} | {str(doc.get('created_at',''))[:10]}")
                        if doc.get("description"):
                            st.caption(f"_{doc['description']}_")
                        # Progress bar OCR ratio
                        n_ocr   = doc.get("n_ocr",  0)
                        n_text  = doc.get("n_text", 0)
                        total_p = n_ocr + n_text
                        if total_p > 0:
                            st.progress(n_ocr / total_p,
                                         text=f"OCR: {n_ocr} hal | Teks: {n_text} hal")
                    with col_d:
                        if st.button("🗑", key=f"del_{doc_id}",
                                      help="Hapus dokumen ini", use_container_width=True):
                            with st.spinner("Menghapus..."):
                                engine.delete_document(doc_id)
                            st.success("Dihapus")
                            st.cache_resource.clear()
                            st.rerun()

            st.divider()
            # Export metadata
            if st.button("📥 Export Daftar Dokumen (.xlsx)", key="exp_docs"):
                df_docs = pd.DataFrame([{
                    "Nama File":   d.get("name",""),
                    "Kategori":    d.get("category",""),
                    "Deskripsi":   d.get("description",""),
                    "Halaman":     d.get("pages",0),
                    "Chunks":      d.get("n_chunks",0),
                    "Hal Teks":    d.get("n_text",0),
                    "Hal OCR":     d.get("n_ocr",0),
                    "Diupload":    d.get("uploaded_by",""),
                    "Tanggal":     str(d.get("created_at",""))[:19],
                } for d in docs])
                buf = BytesIO()
                df_docs.to_excel(buf, index=False, engine="openpyxl")
                st.download_button("⬇️ Download Excel",
                                    data=buf.getvalue(),
                                    file_name="daftar_dokumen.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="dl_docs")

    # ══════════════════════════════════════════════════════════
    # TAB 5 — SETUP & INFO
    # ══════════════════════════════════════════════════════════
    with tab_setup:
        st.subheader("⚙️ Setup & Informasi")

        # Database status
        st.markdown("### 🗄️ Koneksi Database")
        if db_ok:
            st.success("✅ Supabase PostgreSQL terhubung dan berjalan normal")
        else:
            st.error("❌ Database tidak terhubung")
            st.markdown("""
**Cara setup Supabase:**

1. Buka [supabase.com](https://supabase.com) → login → **New project**
2. Tunggu project ready (~2 menit)
3. **Settings ⚙️ → Database → Connection string → URI** → Copy
4. Streamlit Cloud → app → **Settings → Secrets**:

```toml
DATABASE_URL = "postgresql://postgres.XXXX:PASSWORD@aws-X-ap-northeast-X.pooler.supabase.com:6543/postgres"

[anthropic]
api_key = "sk-ant-..."
```
5. Save → **Reboot app** ✅
""")

        st.divider()
        st.markdown("### 📖 Panduan Penggunaan")
        st.markdown("""
**Alur kerja yang direkomendasikan:**

1. **Upload dokumen** di tab 📤 — pilih kategori yang tepat agar pencarian lebih presisi
2. **Tanya AI** di tab 💬 — gunakan bahasa natural, Indonesia atau Inggris
3. **Pencarian** di tab 🔍 — untuk cari referensi spesifik dengan keyword
4. **Manajemen** di tab 📑 — lihat semua dokumen, hapus yang tidak perlu

**Tips untuk hasil terbaik:**
- Gunakan nama parameter teknis yang spesifik: "thrust pad temperature", "lube oil pressure"
- Pilih filter kategori untuk membatasi pencarian ke topik tertentu
- Gunakan shortcut "Parameter Boiler/Turbin/Generator" untuk tampilkan semua nilai sekaligus
""")

        st.divider()
        st.markdown("### 🔧 Status Sistem")
        status = {
            "Versi App":        APP_VERSION,
            "Database":         "✅ Terhubung" if db_ok else "❌ Tidak terhubung",
            "Total Dokumen":    str(len(db_get_all_docs())),
            "Total Chunks":     f"{engine.n_chunks:,}",
            "OCR Chunks":       f"{engine.n_ocr_chunks:,}",
            "Embed Model":      EMBED_MODEL,
            "LLM Model":        "claude-sonnet-4-5",
        }
        for k, v in status.items():
            col_k, col_v = st.columns([2, 3])
            col_k.caption(k)
            col_v.caption(v)


# ── Entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    main()
