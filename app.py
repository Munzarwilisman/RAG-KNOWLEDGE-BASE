# ══════════════════════════════════════════════════════════════════════════════
# DIGIT-OPS RAG v3 — Manual Book Knowledge Base
# Hybrid Text + Vision OCR | Windows-safe (tanpa Poppler)
# Akurat untuk tabel parameter teknis (thrust pad temp, clearance, dll)
#
# OCR Engine yang didukung:
#   1. Claude Vision  — via Anthropic API (claude-haiku)
#   2. Chandra OCR HF — lokal via HuggingFace (pip install chandra-ocr)
#   3. Chandra OCR API— via Datalab hosted API (https://www.datalab.to/)
#
# Install:
#   pip install streamlit anthropic faiss-cpu sentence-transformers
#               pymupdf PyPDF2 python-docx pandas numpy plotly openpyxl
#
#   Untuk Chandra OCR (opsional):
#   pip install chandra-ocr          # HuggingFace local
#   pip install chandra-ocr requests  # Datalab API
#
# Jalankan STANDALONE:
#   streamlit run rag_manualbook_v3.py
#
# Import ke DIGIT-OPS (TIDAK membuka RAG app):
#   from rag_manualbook_v3 import RAGEngine, query_manual
# ══════════════════════════════════════════════════════════════════════════════

import os, json, hashlib, re, time, datetime, pickle, base64, tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from io import BytesIO
from typing import Optional, Literal


# ══════════════════════════════════════════════════════════════════════════════
# SUPABASE STORAGE — Persistensi index di Streamlit Cloud
# ══════════════════════════════════════════════════════════════════════════════

_SUPABASE_BUCKET   = "rag-index"
_SUPABASE_INDEX_FILES = ["chunks.pkl", "faiss.index", "doc_meta.json"]


def _get_supabase_client():
    """Buat Supabase client dari secrets."""
    try:
        import streamlit as _st
        from supabase import create_client
        url = _st.secrets["supabase"]["url"]
        key = _st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception:
        return None


def _ensure_bucket(client) -> bool:
    """Pastikan bucket ada, buat jika belum."""
    try:
        buckets = [b.name for b in client.storage.list_buckets()]
        if _SUPABASE_BUCKET not in buckets:
            client.storage.create_bucket(
                _SUPABASE_BUCKET,
                options={"public": False}
            )
        return True
    except Exception:
        return False


def supabase_pull_index(index_dir: Path) -> tuple[bool, str]:
    """
    Download semua file index dari Supabase Storage ke lokal.
    Return: (success, message)
    """
    client = _get_supabase_client()
    if not client:
        return False, "Supabase tidak dikonfigurasi"

    _ensure_bucket(client)
    downloaded = []

    for fname in _SUPABASE_INDEX_FILES:
        try:
            data = client.storage.from_(_SUPABASE_BUCKET).download(fname)
            if data:
                local_path = index_dir / fname
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, "wb") as f:
                    f.write(data)
                downloaded.append(fname)
        except Exception:
            continue  # file belum ada di bucket (pertama kali)

    if not downloaded:
        return False, "Index belum ada di Supabase (pertama kali pakai)"
    return True, f"✅ Berhasil pull: {', '.join(downloaded)}"


def supabase_push_index(index_dir: Path) -> tuple[bool, str]:
    """
    Upload semua file index dari lokal ke Supabase Storage.
    Return: (success, message)
    """
    client = _get_supabase_client()
    if not client:
        return False, "Supabase tidak dikonfigurasi"

    _ensure_bucket(client)
    uploaded = []

    for fname in _SUPABASE_INDEX_FILES:
        fpath = index_dir / fname
        if not fpath.exists():
            continue
        try:
            with open(fpath, "rb") as f:
                file_bytes = f.read()
            # upsert = update jika sudah ada, insert jika belum
            client.storage.from_(_SUPABASE_BUCKET).upload(
                path=fname,
                file=file_bytes,
                file_options={"upsert": "true"},
            )
            uploaded.append(fname)
        except Exception as e:
            continue

    if not uploaded:
        return False, "Tidak ada file index untuk di-upload"
    return True, f"✅ Berhasil push: {', '.join(uploaded)}"


def is_supabase_configured() -> bool:
    """Cek apakah Supabase sudah dikonfigurasi di secrets."""
    try:
        import streamlit as _st
        return ("supabase" in _st.secrets and
                "url" in _st.secrets["supabase"] and
                "key" in _st.secrets["supabase"])
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# OCR ENGINE 1: CLAUDE VISION
# ══════════════════════════════════════════════════════════════════════════════

def _ocr_with_vision(img_bytes: bytes, page_num: int,
                     api_key: str, doc_context: str = "") -> str:
    """Kirim gambar halaman ke Claude Vision → return teks terstruktur."""
    import anthropic as _anthropic
    img_b64 = base64.b64encode(img_bytes).decode()

    prompt = f"""Kamu adalah OCR engine khusus dokumen teknis PLTU/Power Plant.
Dokumen: {doc_context} | Halaman: {page_num}

═══ ATURAN UTAMA — WAJIB DIIKUTI ═══

1. TABEL PARAMETER (contoh: "Parameters at Gauging Points", "Parameters at Measuring Points"):
   - Format SETIAP BARIS: [Nama Parameter Lengkap] | [Unit] | [Normal] | [Alarm Upper] | [Alarm Lower] | [Interlock] | [Remarks]
   - Nama parameter HARUS ada di setiap baris data — JANGAN pisahkan nama dari nilainya
   - Jika sel kosong gunakan "-"

   Contoh OUTPUT BENAR:
   Judul: 5. Parameters at Gauging Points
   Header: Gauging point | Unit | Normal value | Alarm Upper | Alarm Lower | Interlock | Remarks
   Thrust pad temp. | °C | <80 | 85 | - | 100 | -
   Liner temp of turbine front bearing | °C | <80 | 85 | - | 100 | -
   Scavenge oil temp. of thrust bearing | °C | <60 | 65 | - | 70 | -
   Lube oil pressure | MPa(g) | 0.08~0.12 | - | 0.055 | 0.02 | -

   Contoh OUTPUT SALAH (jangan lakukan):
   <80 | 85 | 100    ← nama parameter hilang!
   Thrust pad temp.  ← nilai tidak ada!

2. TABEL CLEARANCE / ALIGNMENT:
   Format: [Nama Komponen] | [Symbol] | [Nilai mm] | [Keterangan]

3. DIAGRAM / P&ID:
   Format: [Tag/Nama Komponen]: [nilai/spesifikasi]

4. TEKS BIASA: salin apa adanya

5. ATURAN TAMBAHAN:
   - Teks bilingual Mandarin+Inggris → ekstrak KEDUANYA
   - Angka harus AKURAT: °C, MPa, kPa, mm, rpm, kN·m, bar
   - "Continued Table N" → tulis ulang header di awal
   - Jangan lewatkan satu pun baris data

Ekstrak SELURUH isi halaman:"""

    try:
        client = _anthropic.Anthropic(api_key=api_key)
        resp   = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2500,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": img_b64
                }},
                {"type": "text", "text": prompt}
            ]}]
        )
        return resp.content[0].text
    except Exception as e:
        return f"[OCR Error hal {page_num}: {e}]"


# ══════════════════════════════════════════════════════════════════════════════
# OCR ENGINE 2: CHANDRA OCR (HuggingFace lokal)
# ══════════════════════════════════════════════════════════════════════════════

def _ocr_with_chandra_hf(img_bytes: bytes, page_num: int,
                          doc_context: str = "") -> str:
    """
    OCR menggunakan Chandra model via HuggingFace Transformers (lokal, tanpa API key).
    Butuh: pip install chandra-ocr
    Rekomendasi: GPU NVIDIA, atau CPU (lambat)
    """
    try:
        from chandra.model import InferenceManager
        from PIL import Image

        # Inisialisasi manager (model di-load sekali, di-cache)
        # Gunakan singleton agar tidak reload setiap halaman
        if not hasattr(_ocr_with_chandra_hf, "_manager"):
            _ocr_with_chandra_hf._manager = InferenceManager(method="hf")

        manager = _ocr_with_chandra_hf._manager
        img     = Image.open(BytesIO(img_bytes)).convert("RGB")
        results = manager.generate([img])

        if results and results[0].markdown:
            md = results[0].markdown.strip()
            return md if md else f"[Chandra HF: halaman {page_num} kosong]"
        return f"[Chandra HF: tidak ada output hal {page_num}]"

    except ImportError:
        return "[Chandra OCR tidak terinstall. Jalankan: pip install chandra-ocr]"
    except Exception as e:
        return f"[Chandra HF Error hal {page_num}: {e}]"


# ══════════════════════════════════════════════════════════════════════════════
# OCR ENGINE 3: CHANDRA OCR (Datalab Hosted API)
# ══════════════════════════════════════════════════════════════════════════════

def _ocr_with_chandra_api(img_bytes: bytes, page_num: int,
                           api_key: str, doc_context: str = "") -> str:
    """
    OCR menggunakan Chandra via Datalab hosted API.
    Daftar & dapatkan API key di: https://www.datalab.to/
    Butuh: pip install requests
    """
    try:
        import requests

        # Datalab Marker/OCR API endpoint
        # Ref: https://www.datalab.to/
        url     = "https://www.datalab.to/api/v1/marker"
        headers = {"X-Api-Key": api_key}

        # Kirim gambar sebagai file (multipart/form-data)
        files = {
            "file": (f"page_{page_num}.jpg", BytesIO(img_bytes), "image/jpeg"),
        }
        data = {
            "output_format": "markdown",
            "force_ocr": "true",           # paksa OCR meski ada teks embed
            "use_llm": "false",            # gunakan model OCR saja (lebih cepat)
            "page_range": "1",             # 1 halaman per request
        }

        response = requests.post(url, headers=headers, files=files,
                                  data=data, timeout=120)

        if response.status_code == 200:
            result = response.json()
            # Polling jika response async
            if result.get("status") == "pending" and result.get("request_check_url"):
                check_url = result["request_check_url"]
                for _ in range(60):  # max 60 detik
                    time.sleep(2)
                    poll = requests.get(check_url, headers=headers, timeout=30)
                    if poll.status_code == 200:
                        poll_data = poll.json()
                        if poll_data.get("status") == "complete":
                            md = poll_data.get("markdown", "").strip()
                            return md if md else f"[Chandra API: halaman {page_num} kosong]"
                        elif poll_data.get("status") == "error":
                            return f"[Chandra API Error: {poll_data.get('error', 'unknown')}]"
                return f"[Chandra API Timeout: halaman {page_num}]"

            # Response langsung (synchronous)
            md = result.get("markdown", "").strip()
            if md:
                return md
            return f"[Chandra API: tidak ada output hal {page_num}]"

        elif response.status_code == 401:
            return "[Chandra API: API key tidak valid. Periksa di https://www.datalab.to/]"
        elif response.status_code == 429:
            return "[Chandra API: Rate limit. Coba beberapa saat lagi.]"
        else:
            return f"[Chandra API HTTP {response.status_code}: {response.text[:200]}]"

    except ImportError:
        return "[requests tidak terinstall. Jalankan: pip install requests]"
    except Exception as e:
        return f"[Chandra API Error hal {page_num}: {e}]"


# ══════════════════════════════════════════════════════════════════════════════
# DISPATCHER OCR — pilih engine berdasarkan setting
# ══════════════════════════════════════════════════════════════════════════════

def _run_ocr(img_bytes: bytes, page_num: int, doc_context: str,
             engine: str,
             claude_api_key: str = "",
             chandra_api_key: str = "") -> tuple[str, str]:
    """
    Jalankan OCR sesuai engine yang dipilih.
    Return: (teks_hasil_ocr, engine_label)
    """
    if engine == OCR_ENGINE_CLAUDE:
        text = _ocr_with_vision(img_bytes, page_num, claude_api_key, doc_context)
        return text, "claude_vision"

    elif engine == OCR_ENGINE_CHANDRA_HF:
        text = _ocr_with_chandra_hf(img_bytes, page_num, doc_context)
        return text, "chandra_hf"

    elif engine == OCR_ENGINE_CHANDRA_API:
        text = _ocr_with_chandra_api(img_bytes, page_num, chandra_api_key, doc_context)
        return text, "chandra_api"

    else:
        # Fallback ke Claude Vision
        text = _ocr_with_vision(img_bytes, page_num, claude_api_key, doc_context)
        return text, "claude_vision"


# ══════════════════════════════════════════════════════════════════════════════
# PDF UTILITIES (pure Python, Windows-safe, tanpa Poppler)
# ══════════════════════════════════════════════════════════════════════════════

def _get_page_count(pdf_bytes: bytes) -> int:
    for lib in ["pypdf", "PyPDF2", "fitz"]:
        try:
            if lib == "pypdf":
                import pypdf
                return len(pypdf.PdfReader(BytesIO(pdf_bytes)).pages)
            elif lib == "PyPDF2":
                import PyPDF2
                return len(PyPDF2.PdfReader(BytesIO(pdf_bytes)).pages)
            elif lib == "fitz":
                import fitz
                return len(fitz.open(stream=pdf_bytes, filetype="pdf"))
        except Exception:
            continue
    return 0


def _extract_text(pdf_bytes: bytes, page_num: int) -> str:
    """Ekstrak teks 1 halaman PDF — pure Python, tanpa CLI."""
    for lib in ["fitz", "pypdf", "PyPDF2", "pdfplumber"]:
        try:
            if lib == "fitz":
                import fitz
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                return doc[page_num - 1].get_text().strip()
            elif lib == "pypdf":
                import pypdf
                return (pypdf.PdfReader(BytesIO(pdf_bytes)).pages[page_num-1]
                        .extract_text() or "").strip()
            elif lib == "PyPDF2":
                import PyPDF2
                return (PyPDF2.PdfReader(BytesIO(pdf_bytes)).pages[page_num-1]
                        .extract_text() or "").strip()
            elif lib == "pdfplumber":
                import pdfplumber
                with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                    return (pdf.pages[page_num-1].extract_text() or "").strip()
        except Exception:
            continue
    return ""


def _rasterize(pdf_bytes: bytes, page_num: int, dpi: int = _OCR_DPI) -> Optional[bytes]:
    """Rasterize 1 halaman PDF → JPEG bytes. Tanpa Poppler."""
    # 1. PyMuPDF
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = doc[page_num - 1].get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("jpeg")
    except Exception:
        pass
    # 2. pdf2image
    try:
        from pdf2image import convert_from_bytes
        imgs = convert_from_bytes(pdf_bytes, dpi=dpi,
                                  first_page=page_num, last_page=page_num)
        if imgs:
            buf = BytesIO()
            imgs[0].save(buf, format="JPEG", quality=85)
            return buf.getvalue()
    except Exception:
        pass
    # 3. pypdfium2
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


def _extract_pdf_hybrid(pdf_bytes: bytes, filename: str, category: str,
                        claude_api_key: str,
                        chandra_api_key: str = "",
                        ocr_engine: str = OCR_ENGINE_CLAUDE,
                        progress_cb=None) -> list[dict]:
    """
    Scan semua halaman PDF:
    - teks >= _IMG_THRESHOLD  → ekstraksi teks biasa (gratis)
    - teks <  _IMG_THRESHOLD  → rasterize + OCR (engine sesuai pilihan)
    """
    results = []
    n_pages = _get_page_count(pdf_bytes)
    if n_pages == 0:
        return results

    doc_ctx = f"{filename} | {category}"
    n_txt = n_vis = n_err = 0

    for pg in range(1, n_pages + 1):
        if progress_cb:
            progress_cb(pg, n_pages, n_txt, n_vis)

        raw = _extract_text(pdf_bytes, pg)

        if len(raw) >= _IMG_THRESHOLD:
            results.append({"text": raw, "page": pg, "type": "text",
                             "source": filename, "category": category})
            n_txt += 1
        else:
            img = _rasterize(pdf_bytes, pg)
            if img:
                ocr_text, engine_label = _run_ocr(
                    img, pg, doc_ctx,
                    engine=ocr_engine,
                    claude_api_key=claude_api_key,
                    chandra_api_key=chandra_api_key,
                )
                is_error = (ocr_text.startswith("[OCR Error") or
                            ocr_text.startswith("[Chandra") and "Error" in ocr_text)
                if ocr_text and not is_error:
                    combined = (raw + "\n\n" + ocr_text).strip() if raw else ocr_text
                    results.append({"text": combined, "page": pg,
                                    "type": engine_label,
                                    "source": filename, "category": category})
                    n_vis += 1
                else:
                    results.append({"text": f"[Hal {pg} OCR gagal: {ocr_text}]", "page": pg,
                                    "type": "error", "source": filename, "category": category})
                    n_err += 1
            else:
                if raw:
                    results.append({"text": raw, "page": pg, "type": "text",
                                    "source": filename, "category": category})
                    n_txt += 1
                else:
                    n_err += 1
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CHUNKING — TABEL-AWARE
# ══════════════════════════════════════════════════════════════════════════════

def _is_table_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if "|" in s:
        return True
    parts = [p.strip() for p in re.split(r'  +|\t', s) if p.strip()]
    return len(parts) >= 2


def _is_header_line(line: str) -> bool:
    s = line.strip().lower()
    keywords = ["gauging point", "parameter", "unit", "normal value", "alarm",
                 "interlock", "symbol", "clearance", "fit", "remark", "upper",
                 "lower", "limit", "nilai", "satuan", "measuring point"]
    return any(kw in s for kw in keywords)


def chunk_text(text: str, chunk_size: int = _CHUNK_SIZE,
               overlap: int = _CHUNK_OVERLAP) -> list[str]:
    text  = re.sub(r'\n{3,}', '\n\n', text).strip()
    lines = text.split('\n')

    table_line_count = sum(1 for l in lines if _is_table_line(l))
    is_table_content = table_line_count >= 3 or text.count('|') > 6

    if is_table_content:
        headers    = []
        data_rows  = []
        found_data = False

        for ln in lines:
            if not ln.strip():
                continue
            if not found_data and (_is_header_line(ln) or ln.strip().startswith("Judul")
                                   or ln.strip().startswith("Header")
                                   or ln.strip().startswith("5.")
                                   or ln.strip().startswith("Table")):
                headers.append(ln)
            else:
                found_data = True
                data_rows.append(ln)

        if not headers and data_rows:
            headers   = data_rows[:2]
            data_rows = data_rows[2:]

        header_str = "\n".join(headers)
        chunks     = []
        cur_rows   = []
        cur_len    = len(header_str) + 1

        for row in data_rows:
            row_len = len(row) + 1
            if cur_len + row_len > chunk_size and cur_rows:
                content = (header_str + "\n" + "\n".join(cur_rows)).strip()
                if len(content) > 30:
                    chunks.append(content)
                cur_rows = cur_rows[-3:] + [row]
                cur_len  = len(header_str) + sum(len(r)+1 for r in cur_rows)
            else:
                cur_rows.append(row)
                cur_len += row_len

        if cur_rows:
            content = (header_str + "\n" + "\n".join(cur_rows)).strip()
            if len(content) > 30:
                chunks.append(content)

        return chunks if chunks else [text]

    sents  = re.split(r'(?<=[.!?\n])\s+', text)
    chunks = []
    cur    = ""
    for sent in sents:
        if len(cur) + len(sent) + 1 > chunk_size and cur:
            chunks.append(cur.strip())
            words = cur.split()
            cur   = " ".join(words[max(0, len(words) - overlap // 6):]) + " " + sent
        else:
            cur = (cur + " " + sent).strip()
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in chunks if len(c) > 60]


# ══════════════════════════════════════════════════════════════════════════════
# SINONIM TEKNIS
# ══════════════════════════════════════════════════════════════════════════════

_SYNONYMS: dict[str, list[str]] = {
    "thrust pad":         ["thrust pad temp", "推力瓦", "thrust bearing pad temperature",
                            "瓦温", "pad metal temperature", "bearing pad"],
    "pad temp":           ["thrust pad temp", "bearing pad temperature", "瓦温",
                            "metal temperature", "推力瓦块温度"],
    "bearing temp":       ["bearing temperature", "瓦温", "轴承温度", "liner temp",
                            "bearing pad temp", "pad metal temp", "scavenge oil temp"],
    "liner temp":         ["bearing liner temperature", "瓦温", "轴承温度",
                            "bearing temperature"],
    "lube oil":           ["lubricating oil", "润滑油", "lube oil pressure",
                            "oil temperature", "oil pressure", "minyak pelumas"],
    "scavenge oil":       ["return oil", "回油", "scavenge oil temperature",
                            "bearing return oil"],
    "axial displacement": ["axial shift", "轴向位移", "rotor axial", "thrust position"],
    "clearance":          ["fit", "gap", "间隙", "toleransi", "tolerance", "celah"],
    "vibration":          ["vibrasi", "振动", "bearing vibration", "amplitude"],
    "alignment":          ["centering", "对中", "coupling alignment", "senter"],
    "alarm":              ["upper limit", "报警", "alarm value", "upper alarm",
                            "high alarm", "HH", "HA"],
    "trip":               ["interlock", "跳机", "shutdown", "emergency stop",
                            "protection", "proteksi"],
    "normal":             ["normal value", "normal operating", "rated", "额定",
                            "design value", "setpoint"],
    "temperature":        ["temp", "temperatur", "suhu", "℃", "°C", "温度"],
    "pressure":           ["press", "tekanan", "压力", "MPa", "kPa", "bar"],
}


def _expand_query(query: str) -> list[str]:
    queries = [query]
    q_lower = query.lower()
    for key, syns in _SYNONYMS.items():
        if key in q_lower:
            for syn in syns[:2]:
                expanded = re.sub(re.escape(key), syn, q_lower, flags=re.IGNORECASE)
                if expanded not in queries:
                    queries.append(expanded)
    clean = re.sub(r'\b(berapa|apa|bagaimana|apakah|jelaskan|sebutkan)\b',
                   '', q_lower).strip()
    if clean and clean not in queries:
        queries.append(clean)
    return queries[:5]


# ══════════════════════════════════════════════════════════════════════════════
# RAG ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class RAGEngine:
    """
    Engine RAG untuk DIGIT-OPS Manual Book.
    Import ke DigitOPS.py:
        from rag_manualbook_v3 import RAGEngine, query_manual
    """

    def __init__(self, index_dir: str = "rag_index"):
        self.index_dir    = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        self._embed_model = None
        self._faiss_index = None
        self._chunks: list[dict] = []
        self._doc_meta: dict     = {}
        # Pull index dari Supabase jika dikonfigurasi dan index lokal kosong
        if is_supabase_configured():
            local_ok = (self.index_dir / "chunks.pkl").exists()
            if not local_ok:
                supabase_pull_index(self.index_dir)
        self._load_index()

    @property
    def embed_model(self):
        if self._embed_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embed_model = SentenceTransformer(_EMBED_MODEL)
            except ImportError:
                return None
        return self._embed_model

    def _save(self, push_to_supabase: bool = True):
        import faiss
        with open(self.index_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        with open(self.index_dir / "doc_meta.json", "w", encoding="utf-8") as f:
            json.dump(self._doc_meta, f, ensure_ascii=False, indent=2)
        if self._faiss_index is not None:
            faiss.write_index(self._faiss_index, str(self.index_dir / "faiss.index"))
        # Auto-push ke Supabase jika dikonfigurasi
        if push_to_supabase and is_supabase_configured():
            supabase_push_index(self.index_dir)

    def _load_index(self):
        try:
            import faiss
            cp = self.index_dir / "chunks.pkl"
            mp = self.index_dir / "doc_meta.json"
            fp = self.index_dir / "faiss.index"

            if cp.exists():
                with open(cp, "rb") as f:
                    self._chunks = pickle.load(f)
                changed = False
                for c in self._chunks:
                    if "page_type" not in c:
                        c["page_type"] = "text"; changed = True
                    if "doc_id" not in c:
                        c["doc_id"] = "legacy_" + hashlib.md5(
                            c.get("source", "").encode()).hexdigest()[:8]
                        changed = True
                if changed:
                    with open(cp, "wb") as f:
                        pickle.dump(self._chunks, f)

            if mp.exists():
                with open(mp, "r", encoding="utf-8") as f:
                    self._doc_meta = json.load(f)
                changed = False
                for meta in self._doc_meta.values():
                    if "n_text_pages" not in meta:
                        meta["n_text_pages"]   = meta.get("pages", 0)
                        meta["n_vision_pages"] = 0
                        meta["n_error_pages"]  = 0
                        changed = True
                    # Migrasi: tambah field ocr_engine jika belum ada
                    if "ocr_engine" not in meta:
                        meta["ocr_engine"] = OCR_ENGINE_CLAUDE
                        changed = True
                if changed:
                    with open(mp, "w", encoding="utf-8") as f:
                        json.dump(self._doc_meta, f, ensure_ascii=False, indent=2)

            if fp.exists():
                self._faiss_index = faiss.read_index(str(fp))
        except Exception:
            self._chunks = []; self._doc_meta = {}; self._faiss_index = None

    def _embed_add(self, new_chunks: list[dict]):
        import faiss
        if not new_chunks or self.embed_model is None:
            return
        texts  = [c["text"] for c in new_chunks]
        embeds = self.embed_model.encode(texts, batch_size=32, show_progress_bar=False)
        embeds = np.array(embeds, dtype=np.float32)
        norms  = np.linalg.norm(embeds, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeds /= norms
        if self._faiss_index is None:
            self._faiss_index = faiss.IndexFlatIP(embeds.shape[1])
        self._faiss_index.add(embeds)
        self._chunks.extend(new_chunks)

    def add_pdf_hybrid(self, file_bytes: bytes, filename: str, category: str,
                       claude_api_key: str = "",
                       chandra_api_key: str = "",
                       ocr_engine: str = OCR_ENGINE_CLAUDE,
                       doc_description: str = "",
                       progress_cb=None) -> dict:
        doc_hash = hashlib.md5(file_bytes).hexdigest()
        if doc_hash in [m.get("hash") for m in self._doc_meta.values()]:
            return {"ok": False, "message": f"'{filename}' sudah ada.",
                    "n_chunks": 0, "n_text": 0, "n_vision": 0, "n_err": 0}

        doc_id = f"doc_{doc_hash[:12]}"
        pages  = _extract_pdf_hybrid(
            file_bytes, filename, category,
            claude_api_key=claude_api_key,
            chandra_api_key=chandra_api_key,
            ocr_engine=ocr_engine,
            progress_cb=progress_cb,
        )
        if not pages:
            return {"ok": False, "message": "Gagal ekstrak halaman.",
                    "n_chunks": 0, "n_text": 0, "n_vision": 0, "n_err": 0}

        new_chunks = []
        for pg in pages:
            if pg["type"] == "error":
                continue
            for ct in chunk_text(pg["text"]):
                new_chunks.append({
                    "text": ct, "source": filename, "page": pg["page"],
                    "page_type": pg["type"], "category": category,
                    "doc_id": doc_id, "description": doc_description,
                })

        if not new_chunks:
            return {"ok": False, "message": "Tidak ada chunk terbentuk.",
                    "n_chunks": 0, "n_text": 0, "n_vision": 0, "n_err": 0}
        if self.embed_model is None:
            return {"ok": False, "message": "Embedding model tidak tersedia.",
                    "n_chunks": 0, "n_text": 0, "n_vision": 0, "n_err": 0}

        self._embed_add(new_chunks)

        n_t = sum(1 for p in pages if p["type"] == "text")
        n_v = sum(1 for p in pages if p["type"] != "text" and p["type"] != "error")
        n_e = sum(1 for p in pages if p["type"] == "error")

        self._doc_meta[doc_id] = {
            "name": filename, "category": category, "description": doc_description,
            "pages": len(pages), "n_text_pages": n_t, "n_vision_pages": n_v,
            "n_error_pages": n_e, "n_chunks": len(new_chunks), "hash": doc_hash,
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "n_chars": sum(len(c["text"]) for c in new_chunks),
            "ocr_engine": ocr_engine,
        }
        self._save()
        engine_short = {
            OCR_ENGINE_CLAUDE:       "Claude Vision",
            OCR_ENGINE_CHANDRA_HF:   "Chandra HF",
            OCR_ENGINE_CHANDRA_API:  "Chandra API",
        }.get(ocr_engine, ocr_engine)
        return {"ok": True, "doc_id": doc_id, "n_chunks": len(new_chunks),
                "n_text": n_t, "n_vision": n_v, "n_err": n_e,
                "message": (f"Berhasil [{engine_short}]: {len(new_chunks)} chunks "
                             f"dari {len(pages)} halaman "
                             f"({n_t} teks + {n_v} OCR + {n_e} error)")}

    def add_document(self, file_bytes: bytes, filename: str, category: str,
                     doc_description: str = "") -> dict:
        doc_hash = hashlib.md5(file_bytes).hexdigest()
        if doc_hash in [m.get("hash") for m in self._doc_meta.values()]:
            return {"ok": False, "doc_id": None, "n_chunks": 0,
                    "message": f"'{filename}' sudah ada."}

        doc_id = f"doc_{doc_hash[:12]}"
        ext    = Path(filename).suffix.lower()
        pages  = []

        try:
            if ext in (".docx", ".doc"):
                from docx import Document
                doc  = Document(BytesIO(file_bytes))
                full = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
                words = full.split()
                for i in range(0, len(words), 500):
                    pages.append({"text": " ".join(words[i:i+500]),
                                  "page": i//500+1, "type": "text"})
            elif ext == ".txt":
                words = file_bytes.decode("utf-8", errors="ignore").split()
                for i in range(0, len(words), 500):
                    pages.append({"text": " ".join(words[i:i+500]),
                                  "page": i//500+1, "type": "text"})
            elif ext in (".xlsx", ".xls", ".csv"):
                df = (pd.read_csv(BytesIO(file_bytes)) if ext == ".csv"
                      else pd.read_excel(BytesIO(file_bytes)))
                for i in range(0, len(df), 20):
                    pages.append({"text": df.iloc[i:i+20].to_string(index=False),
                                  "page": i//20+1, "type": "text"})
        except Exception as e:
            return {"ok": False, "doc_id": None, "n_chunks": 0, "message": str(e)}

        if not pages:
            return {"ok": False, "doc_id": None, "n_chunks": 0,
                    "message": "Tidak ada halaman berhasil diekstrak."}

        new_chunks = []
        for pg in pages:
            for ct in chunk_text(pg["text"]):
                new_chunks.append({
                    "text": ct, "source": filename, "page": pg["page"],
                    "page_type": "text", "category": category,
                    "doc_id": doc_id, "description": doc_description,
                })

        if not new_chunks:
            return {"ok": False, "doc_id": None, "n_chunks": 0, "message": "Tidak ada chunk."}
        if self.embed_model is None:
            return {"ok": False, "doc_id": None, "n_chunks": 0,
                    "message": "Embedding model tidak tersedia."}

        self._embed_add(new_chunks)
        self._doc_meta[doc_id] = {
            "name": filename, "category": category, "description": doc_description,
            "pages": len(pages), "n_text_pages": len(pages), "n_vision_pages": 0,
            "n_error_pages": 0, "n_chunks": len(new_chunks), "hash": doc_hash,
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "n_chars": sum(len(c["text"]) for c in new_chunks),
            "ocr_engine": "text_only",
        }
        self._save()
        return {"ok": True, "doc_id": doc_id, "n_chunks": len(new_chunks),
                "message": f"Berhasil: {len(new_chunks)} chunks dari {len(pages)} halaman."}

    def retrieve(self, query: str, top_k: int = _TOP_K,
                 filter_category: Optional[str] = None,
                 filter_doc_id: Optional[str]   = None) -> list[dict]:
        if self._faiss_index is None or not self._chunks:
            return []
        if self.embed_model is None:
            return []

        queries = _expand_query(query)
        q_embs  = self.embed_model.encode(queries, show_progress_bar=False)
        q_embs  = np.array(q_embs, dtype=np.float32)
        norms   = np.linalg.norm(q_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        q_embs /= norms

        k         = min(top_k * 10, len(self._chunks))
        score_map: dict[int, float] = {}
        for q_emb in q_embs:
            scores, idxs = self._faiss_index.search(q_emb.reshape(1, -1), k)
            for sc, idx in zip(scores[0], idxs[0]):
                if idx >= 0:
                    score_map[idx] = max(score_map.get(idx, 0.0), float(sc))

        results    = []
        seen_texts = set()
        for idx, score in sorted(score_map.items(), key=lambda x: x[1], reverse=True):
            if idx >= len(self._chunks):
                continue
            c = self._chunks[idx]
            if (filter_category and filter_category != "Semua"
                    and c["category"] != filter_category):
                continue
            if filter_doc_id and c["doc_id"] != filter_doc_id:
                continue
            key = c["text"][:100]
            if key in seen_texts:
                continue
            seen_texts.add(key)
            results.append({**c, "score": score})
            if len(results) >= top_k:
                break
        return results

    def keyword_search(self, query: str, top_k: int = 5,
                       filter_category: Optional[str] = None) -> list[dict]:
        keywords = [w.lower() for w in re.split(r'\s+', query.strip()) if len(w) > 2]
        results  = []
        for c in self._chunks:
            if filter_category and filter_category != "Semua" \
                    and c["category"] != filter_category:
                continue
            text_lower = c["text"].lower()
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                results.append({**c, "score": score / len(keywords),
                                "match_type": "keyword"})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def delete_document(self, doc_id: str) -> bool:
        import faiss
        if doc_id not in self._doc_meta:
            return False
        keep = [c for c in self._chunks if c["doc_id"] != doc_id]
        if not keep:
            self._chunks = []; self._faiss_index = None
            del self._doc_meta[doc_id]; self._save()
            return True
        if self.embed_model is None:
            return False
        texts  = [c["text"] for c in keep]
        embeds = self.embed_model.encode(texts, batch_size=32, show_progress_bar=False)
        embeds = np.array(embeds, dtype=np.float32)
        norms  = np.linalg.norm(embeds, axis=1, keepdims=True)
        norms[norms == 0] = 1; embeds /= norms
        ni = faiss.IndexFlatIP(embeds.shape[1]); ni.add(embeds)
        self._chunks = keep; self._faiss_index = ni
        del self._doc_meta[doc_id]; self._save()
        return True

    @property
    def n_docs(self): return len(self._doc_meta)
    @property
    def n_chunks(self): return len(self._chunks)
    @property
    def n_vision_chunks(self):
        try: return sum(1 for c in self._chunks if c.get("page_type") != "text")
        except: return 0
    @property
    def doc_list(self): return list(self._doc_meta.items())
    @property
    def categories_available(self):
        return list(set(m["category"] for m in self._doc_meta.values()))


# ══════════════════════════════════════════════════════════════════════════════
# QUERY FUNCTION (untuk integrasi DIGIT-OPS)
# ══════════════════════════════════════════════════════════════════════════════

def query_manual(question: str, context_data: Optional[str] = None,
                 filter_category: Optional[str] = None, top_k: int = _TOP_K,
                 api_key: Optional[str] = None,
                 engine: Optional[RAGEngine] = None) -> str:
    import anthropic as _anthropic

    if engine is None:
        engine = RAGEngine()
    if api_key is None:
        try:
            import streamlit as _st
            api_key = _st.secrets["anthropic"]["api_key"]
        except Exception:
            return "❌ API key tidak ditemukan."

    chunks = engine.retrieve(question, top_k=top_k, filter_category=filter_category)

    if not chunks or (chunks and chunks[0]["score"] < 0.3):
        kw_results = engine.keyword_search(question, top_k=5,
                                           filter_category=filter_category)
        existing_keys = {c["text"][:100] for c in chunks}
        for r in kw_results:
            if r["text"][:100] not in existing_keys:
                chunks.append(r)
                existing_keys.add(r["text"][:100])
        chunks = chunks[:top_k]

    if not chunks:
        return ("⚠️ Tidak ditemukan referensi yang relevan di knowledge base.\n"
                "Pastikan sudah upload manual book dan lakukan Vision OCR pada "
                "halaman tabel/gambar.")

    ctx_parts = []
    for i, ch in enumerate(chunks):
        pt   = ch.get("page_type", "text")
        note = ""
        if pt == "claude_vision": note = " [Claude Vision OCR]"
        elif "chandra" in pt:     note = " [Chandra OCR]"
        src  = f"[{ch['source']} | Hal.{ch['page']}{note} | {ch['category']}]"
        ctx_parts.append(f"SUMBER {i+1} {src}:\n{ch['text']}")
    context_str = "\n\n---\n\n".join(ctx_parts)

    _is_full_table_query = any(kw in question.lower() for kw in
        ["tunjukkan", "tampilkan", "semua", "seluruh", "all parameter",
         "batas parameter", "daftar parameter", "tabel parameter"])

    sys_prompt = """Kamu adalah AI Engineer PLTU ahli operasional boiler CFB, steam turbine, generator.

ATURAN WAJIB:
1. Jawab HANYA berdasarkan SUMBER REFERENSI yang diberikan
2. Jika tidak ada di sumber → katakan "Tidak ditemukan di manual book yang tersedia"
3. Sebutkan sumber referensi (nama dokumen + halaman) di akhir
4. Satuan teknis HARUS benar: MPa, kPa, °C, bar, MW, rpm, mm, kN·m
5. "thrust pad" = "推力瓦" = "bearing pad" = "pad temperature" → sama

ATURAN FORMAT TABEL PARAMETER:
- SELALU tampilkan parameter dalam tabel markdown:
  | No | Parameter | Satuan | Normal | Alarm Upper | Alarm Lower | Trip/Interlock | Keterangan |
  |----|-----------|--------|--------|-------------|-------------|----------------|------------|
  | 1  | nama param| unit   | nilai  | nilai       | nilai       | nilai          | catatan    |
- Gunakan "-" jika nilai tidak ada di sumber
- JANGAN hilangkan parameter apapun dari sumber
- URUTKAN per kelompok: Temperatur → Tekanan → Level → Flow → Lainnya
- Jika ada tabel "Parameters at Gauging Points" di sumber → ekstrak SEMUA barisnya
- Untuk query "tampilkan semua" → kumpulkan dari SEMUA sumber lalu gabungkan dalam 1 tabel besar"""

    user_msg = f"PERTANYAAN: {question}\n\n"
    if context_data:
        user_msg += f"DATA OPERASIONAL:\n{context_data}\n\n"

    if _is_full_table_query:
        user_msg += (
            "INSTRUKSI KHUSUS: Ini adalah permintaan untuk menampilkan SEMUA parameter.\n"
            "Kumpulkan SETIAP baris parameter dari semua sumber di bawah.\n"
            "Gabungkan dalam SATU tabel besar yang terurut.\n"
            "Jangan ringkas, jangan skip parameter apapun.\n\n"
        )

    user_msg += f"REFERENSI MANUAL BOOK:\n{context_str}\n\nBuat tabel parameter yang LENGKAP dan AKURAT."

    try:
        client   = _anthropic.Anthropic(api_key=api_key)
        _max_tok = 4000 if _is_full_table_query else 1800
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=_max_tok,
            system=sys_prompt,
            messages=[{"role": "user", "content": user_msg}]
        )
        answer  = response.content[0].text
        sources = list(dict.fromkeys(
            [f"{c['source']} (Hal.{c['page']})" for c in chunks]))
        return answer + "\n\n---\n📚 **Sumber:** " + " · ".join(sources[:8])
    except Exception as e:
        return f"❌ Error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.stApp { background: linear-gradient(135deg,#0a0f1e,#111827,#0d1117); color: #e2e8f0; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0f172a,#1e293b) !important; border-right: 1px solid rgba(99,102,241,.2); }
h1 { font-size:1.8rem !important; font-weight:800 !important; background:linear-gradient(90deg,#6366f1,#a78bfa,#38bdf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
h2 { font-size:1.3rem !important; font-weight:700 !important; color:#c7d2fe !important; }
h3 { font-size:1.1rem !important; font-weight:600 !important; color:#a5b4fc !important; }
.stButton > button { background:linear-gradient(135deg,#4f46e5,#7c3aed) !important; color:#fff !important; font-weight:600 !important; border:none !important; border-radius:10px !important; padding:10px 22px !important; }
.stTextInput > div > div > input, .stTextArea > div > div > textarea { background:rgba(30,41,59,.8) !important; color:#e2e8f0 !important; border:1px solid rgba(99,102,241,.3) !important; border-radius:10px !important; }
.stSelectbox > div > div { background:rgba(30,41,59,.8) !important; color:#e2e8f0 !important; border:1px solid rgba(99,102,241,.3) !important; border-radius:10px !important; }
.stTabs [data-baseweb="tab-list"] { background:rgba(15,23,42,.8); border-radius:12px; padding:4px; border:1px solid rgba(99,102,241,.2); }
.stTabs [data-baseweb="tab"] { border-radius:9px !important; font-weight:600 !important; color:#94a3b8 !important; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,#4f46e5,#7c3aed) !important; color:#fff !important; }
.stat-card { background:linear-gradient(135deg,rgba(30,41,59,.9),rgba(15,23,42,.9)); border:1px solid rgba(99,102,241,.25); border-radius:12px; padding:14px; text-align:center; margin:4px 0; }
.stat-val { font-size:1.6em; font-weight:800; color:#818cf8; }
.stat-lbl { font-size:.72rem; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:.8px; }
.answer-box { background:linear-gradient(135deg,rgba(30,41,59,.95),rgba(15,23,42,.95)); border:1px solid rgba(99,102,241,.3); border-left:4px solid #6366f1; border-radius:14px; padding:20px 24px; margin:12px 0; line-height:1.75; }
.chunk-card { background:rgba(30,41,59,.6); border:1px solid rgba(99,102,241,.2); border-left:3px solid #6366f1; border-radius:8px; padding:12px 16px; margin:8px 0; font-size:.88rem; }
.source-badge { display:inline-block; background:rgba(99,102,241,.15); border:1px solid rgba(99,102,241,.35); border-radius:20px; padding:3px 12px; font-size:.76rem; color:#a5b4fc; margin:3px 4px 3px 0; }
.doc-card { background:linear-gradient(135deg,rgba(30,41,59,.9),rgba(15,23,42,.9)); border:1px solid rgba(99,102,241,.25); border-radius:12px; padding:14px 18px; margin:6px 0; }
.progress-log { background:rgba(15,23,42,.9); border:1px solid rgba(99,102,241,.2); border-radius:10px; padding:12px 16px; font-size:.82rem; color:#94a3b8; font-family:monospace; max-height:180px; overflow-y:auto; }
::-webkit-scrollbar { width:6px; } ::-webkit-scrollbar-track { background:#0f172a; } ::-webkit-scrollbar-thumb { background:rgba(99,102,241,.4); border-radius:3px; }
</style>
"""


def run_rag_app():
    import streamlit as st

    st.set_page_config(page_title="DIGIT-OPS RAG v3", page_icon="📚",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Session state ──────────────────────────────────────────────────────────
    if "rag_engine" in st.session_state:
        if not hasattr(st.session_state.rag_engine, "n_vision_chunks"):
            del st.session_state["rag_engine"]
    if "rag_engine" not in st.session_state:
        with st.spinner("Memuat knowledge base..."):
            st.session_state.rag_engine = RAGEngine()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "ocr_engine" not in st.session_state:
        st.session_state.ocr_engine = OCR_ENGINE_CLAUDE

    rag: RAGEngine = st.session_state.rag_engine

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        try:
            st.image("https://gajiloker.com/wp-content/uploads/2024/02/Gaji-PT-PLN-Nusantara-Power-Services.jpg", width=250)
        except Exception:
            pass
        st.markdown("## 📚 DIGIT-OPS RAG v3")
        st.markdown("<div style='color:#64748b;font-size:.82rem;'>Hybrid Text + Vision OCR | Keyword Fallback</div>", unsafe_allow_html=True)
        st.markdown("---")

        # Stats
        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="stat-card"><div class="stat-val">{rag.n_docs}</div><div class="stat-lbl">Dokumen</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-card"><div class="stat-val">{rag.n_chunks:,}</div><div class="stat-lbl">Chunks</div></div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        c3.markdown(f'<div class="stat-card"><div class="stat-val">{rag.n_vision_chunks:,}</div><div class="stat-lbl">OCR</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="stat-card"><div class="stat-val">{rag.n_chunks - rag.n_vision_chunks:,}</div><div class="stat-lbl">Teks</div></div>', unsafe_allow_html=True)
        st.markdown("---")

        # OCR Engine
        st.markdown("#### 🔬 OCR Engine")
        selected_engine = st.selectbox(
            "Engine untuk halaman gambar/tabel:",
            _OCR_ENGINE_OPTIONS,
            index=_OCR_ENGINE_OPTIONS.index(st.session_state.ocr_engine),
            key="sb_ocr_engine",
        )
        st.session_state.ocr_engine = selected_engine

        if selected_engine == OCR_ENGINE_CHANDRA_API:
            st.text_input("Datalab API Key:", type="password", key="datalab_api_key", placeholder="dl-...")
        st.markdown("---")

        # Supabase status
        st.markdown("#### ☁️ Supabase Storage")
        supa_ok = is_supabase_configured()
        if supa_ok:
            st.success("✅ Terhubung — index tersimpan otomatis")
            sc1, sc2 = st.columns(2)
            if sc1.button("⬆️ Push", key="supa_push", use_container_width=True):
                with st.spinner("Uploading..."):
                    ok, msg = supabase_push_index(rag.index_dir)
                st.toast(msg)
            if sc2.button("⬇️ Pull", key="supa_pull", use_container_width=True):
                with st.spinner("Downloading..."):
                    ok, msg = supabase_pull_index(rag.index_dir)
                if ok:
                    del st.session_state["rag_engine"]
                    st.rerun()
                else:
                    st.toast(msg)
        else:
            st.warning("⚠️ Belum dikonfigurasi\nLihat tab 🔗 Integrasi")
        st.markdown("---")

        # Filter & mode
        all_cats = ["Semua"] + sorted(rag.categories_available)
        selected_cat = st.selectbox("Filter Kategori:", all_cats, key="sb_cat")
        st.markdown("---")
        int_mode = st.checkbox("Mode Integrasi DIGIT-OPS", key="int_mode")
        ctx_input = ""
        if int_mode:
            ctx_input = st.text_area("Data Operasional:", height=90, key="ctx_in",
                placeholder="Beban: 22 MW\nSteam P: 9.2 MPa\nBed Temp: 870°C")

    # ── Main ───────────────────────────────────────────────────────────────────
    st.title("📚 DIGIT-OPS RAG v3 — Manual Book Knowledge Base")
    st.markdown("<div style='color:#64748b;margin-bottom:16px;'>Hybrid Text + Vision OCR + Keyword Fallback — parameter teknis PLTU</div>", unsafe_allow_html=True)

    tab_chat, tab_upload, tab_search, tab_docs, tab_integrate = st.tabs([
        "💬 Tanya AI", "📤 Upload Manual", "🔍 Pencarian", "📑 Dokumen", "🔗 Integrasi"
    ])

    # ── TAB 1: CHAT ────────────────────────────────────────────────────────────
    with tab_chat:
        if rag.n_docs == 0:
            st.info("📢 Upload manual book di tab **📤 Upload Manual** terlebih dahulu.")

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div style="background:rgba(99,102,241,.12);border:1px solid rgba(99,102,241,.25);border-radius:12px;padding:12px 16px;margin:8px 0 4px 40px;">🧑‍💼 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="answer-box">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

        question = st.text_area("Tanyakan:", height=90, key="cin",
            placeholder="Contoh: Berapa thrust pad temperature normal dan alarm?")

        col_send, col_clr = st.columns([1, 1])
        send = col_send.button("➤ Kirim", key="send", use_container_width=True)
        clr  = col_clr.button("🗑 Hapus Chat", key="clr", use_container_width=True)

        if clr:
            st.session_state.chat_history = []
            st.rerun()

        if send and question.strip():
            if rag.n_docs == 0:
                st.warning("Upload dokumen dulu.")
            else:
                st.session_state.chat_history.append({"role": "user", "content": question.strip()})
                with st.spinner("🔍 Mencari referensi..."):
                    try: ak = st.secrets["anthropic"]["api_key"]
                    except: ak = None
                    cf  = None if selected_cat == "Semua" else selected_cat
                    ctx = ctx_input if int_mode else None
                    ans = query_manual(question.strip(), context_data=ctx, filter_category=cf, api_key=ak, engine=rag)
                st.session_state.chat_history.append({"role": "assistant", "content": ans})
                st.rerun()

        st.markdown("---")
        st.markdown("#### 📊 Tampilkan Semua Batas Parameter")

        _QUERY_BOILER = ("Tunjukkan SEMUA batas parameter operasi Boiler CFB dari manual book. "
            "Format tabel: nama parameter, nilai normal, alarm upper, alarm lower, trip/interlock, satuan. "
            "Cari: bed temperature, furnace pressure, steam pressure, steam temperature, feed water, "
            "ID fan, FD fan, PA fan, coal feeder, drum level, flue gas, oxygen, NOx, dan semua lainnya.")
        _QUERY_TURBIN = ("Tunjukkan SEMUA batas parameter operasi Steam Turbine dari manual book. "
            "Format tabel: nama parameter, nilai normal, alarm upper, alarm lower, trip/interlock, satuan. "
            "Cari: thrust pad temperature, bearing temperature, lube oil pressure, lube oil temperature, "
            "axial displacement, vibration, steam pressure, exhaust pressure, speed, dan semua lainnya.")

        bcol1, bcol2 = st.columns(2)
        with bcol1:
            st.markdown("<div style='background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);border-radius:12px;padding:12px 16px;margin-bottom:8px;'><div style='color:#fca5a5;font-weight:600;font-size:.85rem;'>🔥 BOILER CFB</div><div style='color:#94a3b8;font-size:.78rem;'>Bed temp · Pressure · Steam · Fan · Feed water</div></div>", unsafe_allow_html=True)
            if st.button("📊 Parameter Boiler", key="btn_boiler", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": "📊 Semua Parameter Boiler CFB"})
                with st.spinner("Mengumpulkan..."):
                    try: ak2 = st.secrets["anthropic"]["api_key"]
                    except: ak2 = None
                    ans = query_manual(_QUERY_BOILER, filter_category="Boiler CFB", top_k=15, api_key=ak2, engine=rag)
                st.session_state.chat_history.append({"role": "assistant", "content": ans})
                st.rerun()

        with bcol2:
            st.markdown("<div style='background:rgba(99,102,241,.1);border:1px solid rgba(99,102,241,.3);border-radius:12px;padding:12px 16px;margin-bottom:8px;'><div style='color:#a5b4fc;font-weight:600;font-size:.85rem;'>⚙️ STEAM TURBINE</div><div style='color:#94a3b8;font-size:.78rem;'>Thrust pad · Bearing · Lube oil · Vibration</div></div>", unsafe_allow_html=True)
            if st.button("📊 Parameter Turbin", key="btn_turbin", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": "📊 Semua Parameter Steam Turbine"})
                with st.spinner("Mengumpulkan..."):
                    try: ak2 = st.secrets["anthropic"]["api_key"]
                    except: ak2 = None
                    ans = query_manual(_QUERY_TURBIN, filter_category="Steam Turbine", top_k=15, api_key=ak2, engine=rag)
                st.session_state.chat_history.append({"role": "assistant", "content": ans})
                st.rerun()

        st.markdown("---")
        st.markdown("#### 💡 Pertanyaan Cepat")
        qq = [
            "Berapa thrust pad temperature: normal, alarm, dan trip?",
            "Berapa clearance thrust bearing dan journal bearing?",
            "Prosedur alignment coupling steam turbine?",
            "Parameter normal lube oil: tekanan, suhu, flow?",
            "Batas vibrasi dan axial displacement turbine?",
            "Trip speed overspeed turbine dan prosedur test?",
        ]
        qcols = st.columns(3)
        for i, q in enumerate(qq):
            if qcols[i % 3].button(q[:40] + "…", key=f"qq{i}", use_container_width=True, help=q):
                st.session_state.chat_history.append({"role": "user", "content": q})
                with st.spinner("Mencari..."):
                    try: ak2 = st.secrets["anthropic"]["api_key"]
                    except: ak2 = None
                    st.session_state.chat_history.append({"role": "assistant", "content": query_manual(q, api_key=ak2, engine=rag)})
                st.rerun()

    # ── TAB 2: UPLOAD ──────────────────────────────────────────────────────────
    with tab_upload:
        st.markdown("### 📤 Upload Manual Book")
        current_engine = st.session_state.ocr_engine

        # Status OCR engine
        engine_colors = {
            OCR_ENGINE_CLAUDE:      ("rgba(99,102,241,.1)",  "rgba(99,102,241,.3)",  "#a5b4fc", "🤖 Claude Vision OCR — Aktif"),
            OCR_ENGINE_CHANDRA_HF:  ("rgba(236,72,153,.08)", "rgba(236,72,153,.3)",  "#f9a8d4", "🔮 Chandra OCR Lokal — Aktif"),
            OCR_ENGINE_CHANDRA_API: ("rgba(236,72,153,.08)", "rgba(236,72,153,.3)",  "#f9a8d4", "🌐 Chandra OCR API — Aktif"),
        }
        bg, border, color, label = engine_colors.get(current_engine, engine_colors[OCR_ENGINE_CLAUDE])
        st.markdown(f"<div style='background:{bg};border:1px solid {border};border-radius:12px;padding:14px 18px;margin-bottom:16px;'><b style='color:{color};'>{label}</b><br><span style='color:#64748b;font-size:.82rem;'>Halaman teks → PyMuPDF (gratis) &nbsp;|&nbsp; Halaman gambar/tabel → {current_engine}</span></div>", unsafe_allow_html=True)

        # Cek API key
        ak_ok = False
        chandra_api_key_val = ""
        if current_engine == OCR_ENGINE_CLAUDE:
            try:
                _ak = st.secrets["anthropic"]["api_key"]
                ak_ok = bool(_ak)
                st.success("✅ Anthropic API key tersedia")
            except Exception:
                st.error("❌ Anthropic API key tidak ada di Secrets")

        elif current_engine == OCR_ENGINE_CHANDRA_HF:
            try:
                from chandra.model import InferenceManager
                ak_ok = True
                st.success("✅ Chandra OCR terinstall")
            except ImportError:
                st.warning("⚠️ Jalankan: pip install chandra-ocr")
                ak_ok = True

        elif current_engine == OCR_ENGINE_CHANDRA_API:
            try:
                chandra_api_key_val = st.session_state.get("datalab_api_key", "") or st.secrets.get("datalab", {}).get("api_key", "")
            except Exception:
                chandra_api_key_val = st.session_state.get("datalab_api_key", "")
            if chandra_api_key_val:
                ak_ok = True
                st.success("✅ Datalab API key tersedia")
            else:
                st.error("❌ Datalab API key belum diisi di sidebar")

        # Upload form
        uc1, uc2 = st.columns([2, 1])
        with uc1:
            uploaded = st.file_uploader("Pilih file:", accept_multiple_files=True,
                type=["pdf","docx","doc","txt","xlsx","xls","csv"], key="uploader")
        with uc2:
            doc_cat  = st.selectbox("Kategori:", _DOC_CATEGORIES, key="up_cat")
            doc_desc = st.text_input("Deskripsi:", key="up_desc", placeholder="contoh: Steam Turbine N27.5")

        if uploaded and ak_ok:
            try: claude_api_key = st.secrets["anthropic"]["api_key"]
            except: claude_api_key = ""

            if st.button(f"📥 Proses {len(uploaded)} File", key="proc", use_container_width=True):
                for uf in uploaded:
                    fb  = uf.read()
                    ext = Path(uf.name).suffix.lower()
                    st.markdown(f"**📄 {uf.name}**")

                    if ext == ".pdf":
                        prog = st.progress(0)
                        info_txt = st.empty()
                        log_box  = st.empty()
                        logs = []

                        def _cb(pg, tot, nt, nv, _e=current_engine, _p=prog, _i=info_txt, _l=log_box, _logs=logs):
                            _p.progress(int(pg/tot*100))
                            _i.caption(f"Halaman {pg}/{tot} — teks: {nt} | OCR: {nv}")
                            if nv > 0:
                                _logs.append(f"✓ Hal {pg}: {_e}")
                                if len(_logs) > 6: _logs.pop(0)
                                _l.markdown('<div class="progress-log">' + "<br>".join(_logs) + "</div>", unsafe_allow_html=True)

                        with st.spinner("Processing..."):
                            res = rag.add_pdf_hybrid(fb, uf.name, doc_cat,
                                claude_api_key=claude_api_key,
                                chandra_api_key=chandra_api_key_val,
                                ocr_engine=current_engine,
                                doc_description=doc_desc or "",
                                progress_cb=_cb)

                        prog.empty(); info_txt.empty(); log_box.empty()
                        if res["ok"]:
                            st.success(res["message"])
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Chunks", res["n_chunks"])
                            m2.metric("Teks", res["n_text"])
                            m3.metric("OCR", res["n_vision"])
                            m4.metric("Error", res["n_err"])
                        else:
                            st.error(res["message"])
                    else:
                        with st.spinner(f"Processing {uf.name}..."):
                            res = rag.add_document(fb, uf.name, doc_cat, doc_desc or "")
                        if res["ok"]: st.success(res["message"])
                        else:         st.error(res["message"])
                st.rerun()

        with st.expander("📊 Perbandingan OCR Engine"):
            st.markdown("""
| | Claude Vision | Chandra HF | Chandra API |
|---|---|---|---|
| **Biaya** | ~$0.01/100hal | Gratis | Freemium |
| **Kebutuhan** | Anthropic key | GPU + install | Datalab key |
| **Bilingual CN+EN** | ✅ Excellent | ✅ Baik | ✅ Baik |
| **Offline** | ❌ | ✅ | ❌ |
""")

    # ── TAB 3: PENCARIAN ───────────────────────────────────────────────────────
    with tab_search:
        st.markdown("### 🔍 Pencarian")
        if rag.n_docs == 0:
            st.info("Upload dokumen terlebih dahulu.")
        else:
            sc1, sc2, sc3 = st.columns([3, 1, 1])
            sq   = sc1.text_input("Query:", key="sq", placeholder="thrust pad temperature")
            scat = sc2.selectbox("Kategori:", ["Semua"] + sorted(rag.categories_available), key="scat")
            sk   = sc3.slider("Hasil:", 3, 20, 8, key="sk")

            rc1, rc2 = st.columns(2)
            do_sem = rc1.button("🔍 Semantik", key="dos", use_container_width=True)
            do_kw  = rc2.button("🔎 Keyword Eksak", key="dok", use_container_width=True)

            if (do_sem or do_kw) and sq:
                cf = None if scat == "Semua" else scat
                with st.spinner("Mencari..."):
                    if do_kw:
                        st.session_state.search_results = rag.keyword_search(sq, top_k=sk, filter_category=cf)
                    else:
                        st.session_state.search_results = rag.retrieve(sq, top_k=sk, filter_category=cf)

            for r in st.session_state.search_results:
                pct = min(100, int(r["score"] * 100))
                col = "#10b981" if pct > 70 else "#f59e0b" if pct > 45 else "#ef4444"
                pt  = r.get("page_type", "text")
                badge = "🤖 Claude OCR" if pt == "claude_vision" else ("🔮 Chandra OCR" if "chandra" in str(pt) else "📝 Teks")
                st.markdown(f"""<div class="chunk-card">
                    <div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
                        <div><span class="source-badge">📄 {r['source']}</span><span class="source-badge">Hal.{r['page']}</span><span class="source-badge">{r['category']}</span><span class="source-badge">{badge}</span></div>
                        <div style='color:{col};font-weight:700;font-size:.78rem;'>{pct}%</div>
                    </div>
                    <div style='font-size:.85rem;color:#cbd5e1;white-space:pre-wrap;'>{r['text'][:500]}{'...' if len(r['text'])>500 else ''}</div>
                </div>""", unsafe_allow_html=True)

    # ── TAB 4: DOKUMEN ─────────────────────────────────────────────────────────
    with tab_docs:
        st.markdown("### 📑 Daftar Dokumen")
        if rag.n_docs == 0:
            st.info("Belum ada dokumen.")
        else:
            d1, d2, d3, d4 = st.columns(4)
            tc = sum(m.get("n_chars", 0) for m in rag._doc_meta.values())
            d1.markdown(f'<div class="stat-card"><div class="stat-val">{rag.n_docs}</div><div class="stat-lbl">Dokumen</div></div>', unsafe_allow_html=True)
            d2.markdown(f'<div class="stat-card"><div class="stat-val">{rag.n_chunks:,}</div><div class="stat-lbl">Chunks</div></div>', unsafe_allow_html=True)
            d3.markdown(f'<div class="stat-card"><div class="stat-val">{rag.n_vision_chunks:,}</div><div class="stat-lbl">OCR</div></div>', unsafe_allow_html=True)
            d4.markdown(f'<div class="stat-card"><div class="stat-val">{tc//1000}K</div><div class="stat-lbl">Chars</div></div>', unsafe_allow_html=True)
            st.markdown("")

            for doc_id, meta in rag._doc_meta.items():
                nv  = meta.get("n_vision_pages", 0)
                nt  = meta.get("n_text_pages", 0)
                eng = meta.get("ocr_engine", OCR_ENGINE_CLAUDE)
                with st.expander(f"📄 {meta['name']} — {meta['category']}"):
                    dc, db = st.columns([4, 1])
                    dc.markdown(f"""<div class="doc-card">
                        <div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;'>
                            <span class="source-badge">{meta['category']}</span>
                            <span class="source-badge">🗓 {meta.get('date','—')}</span>
                            <span class="source-badge">📃 {meta.get('pages','?')} hal</span>
                            <span class="source-badge">🧩 {meta['n_chunks']} chunks</span>
                            <span class="source-badge">📝 {nt} teks / 🔮 {nv} OCR</span>
                        </div>
                        <div style='font-size:.82rem;color:#64748b;'>{meta.get('description','—') or '—'}</div>
                    </div>""", unsafe_allow_html=True)
                    if db.button("🗑 Hapus", key=f"del_{doc_id}"):
                        with st.spinner("Menghapus..."):
                            rag.delete_document(doc_id)
                        st.rerun()

            st.markdown("---")
            if st.button("📥 Export CSV", key="exp"):
                rows = [{"nama": v["name"], "kategori": v["category"],
                         "halaman": v.get("pages","?"), "chunks": v["n_chunks"],
                         "ocr_engine": v.get("ocr_engine","—"),
                         "tanggal": v.get("date","—")} for v in rag._doc_meta.values()]
                st.download_button("⬇️ Download CSV",
                    data=pd.DataFrame(rows).to_csv(index=False),
                    file_name="rag_docs.csv", mime="text/csv")

    # ── TAB 5: INTEGRASI ───────────────────────────────────────────────────────
    with tab_integrate:
        st.markdown("### 🔗 Integrasi & Setup")

        # Supabase setup guide
        st.markdown("#### ☁️ Setup Supabase (Persistensi Database)")
        supa_status = is_supabase_configured()
        if supa_status:
            st.success("✅ Supabase sudah terhubung!")
        else:
            st.warning("⚠️ Supabase belum dikonfigurasi — index hilang saat app restart")

        with st.expander("📋 Panduan Setup Supabase", expanded=not supa_status):
            st.markdown("""
**Langkah 1** — Buka [supabase.com](https://supabase.com) → login GitHub → **New project** → isi nama & password → Create

**Langkah 2** — Di dashboard project → **Settings ⚙️ → API Keys** → tab **Legacy anon, service_role API keys**
- Copy **anon public** key

**Langkah 3** — Streamlit Cloud → app → **Settings → Secrets**, tambahkan:
```toml
[supabase]
url = "https://PROJECT_ID.supabase.co"
key = "eyJhbGci..."

[anthropic]
api_key = "sk-ant-..."
```

**Langkah 4** — Save → Reboot app ✅
""")

        st.markdown("---")
        st.markdown("#### 🔌 Integrasi ke DIGIT-OPS")
        with st.expander("📝 Snippet kode"):
            st.code("""
from app import RAGEngine, query_manual

@st.cache_resource
def load_rag():
    return RAGEngine("rag_index")

rag = load_rag()

answer = query_manual(
    "Berapa thrust pad temperature normal?",
    engine=rag,
    api_key=st.secrets["anthropic"]["api_key"]
)
""", language="python")

        st.markdown("---")
        st.markdown("#### ⚙️ Info Knowledge Base")
        st.markdown(f"""<div class="doc-card">
            <div style='font-size:.88rem;color:#94a3b8;line-height:2;'>
            🧩 Chunks: <b style='color:#818cf8;'>{rag.n_chunks:,}</b> (📝 {rag.n_chunks-rag.n_vision_chunks:,} teks + 🔮 {rag.n_vision_chunks:,} OCR)<br>
            📚 Dokumen: <b style='color:#818cf8;'>{rag.n_docs}</b><br>
            🔬 OCR Engine: <b style='color:#f9a8d4;'>{st.session_state.ocr_engine}</b><br>
            ☁️ Supabase: <b style='color:{"#6ee7b7" if supa_status else "#f87171"};'>{"Terhubung ✅" if supa_status else "Belum dikonfigurasi ⚠️"}</b>
            </div></div>""", unsafe_allow_html=True)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_rag_app()
