"""
ingest.py  --  step 1 of the pipeline.

Job: take whatever files the dataset gives us (CSV, JSON, JSONL, Excel, TXT, audio)
and turn every usable report into a clean `Message` (see models.py).
Anything unusable (empty, silent, broken, unreadable) is NOT hidden and does NOT crash
the run: it becomes a `RejectedInput` with a plain-language reason.

    dataset folder --> Ingestor.run() --> IngestResult(messages, rejected, warnings)

Usage from other files:
    from ingest import Ingestor
    result = Ingestor().run("data/")
    for message in result.messages: ...

Command line:
    python ingest.py data/ --out messages.json      # process a folder
    python ingest.py --selftest                      # run the built-in checks

Audio needs:  pip install faster-whisper     (runs locally, no API key)
Excel needs:  pip install openpyxl           (only if the dataset has .xlsx)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

from models import Message, RejectedInput, Source

log = logging.getLogger("ingest")

# Reports with a timezone are converted to this one (India Standard Time).
LOCAL_TZ = timezone(timedelta(hours=5, minutes=30))

TABLE_EXTS = {".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".xlsx", ".xlsm", ".txt"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".amr", ".wma", ".webm"}
IGNORED_EXTS = {".md", ".pdf", ".docx", ".png", ".jpg", ".jpeg", ".gif", ".gitkeep", ".log", ".py"}

# Different datasets name their columns differently. We accept all of these (case-insensitive).
TEXT_KEYS = ("text", "message", "body", "content", "msg", "report", "description",
             "transcript", "tweet", "post", "sms", "details")
TIME_KEYS = ("timestamp", "datetime", "date_time", "created_at", "received_at", "sent_at",
             "reported_at", "posted_at", "call_time", "time", "date")
SOURCE_KEYS = ("source", "channel", "platform", "medium", "via", "mode")
SENDER_KEYS = ("sender", "from", "phone", "phone_number", "number", "mobile", "user",
               "username", "handle", "author", "caller", "contact", "user_id")
ID_KEYS = ("id", "message_id", "msg_id", "uuid", "report_id", "ticket_id")
WAVE_KEYS = ("wave", "stage", "phase")
AUDIO_KEYS = ("audio", "audio_file", "audio_path", "recording", "recording_file",
              "call_audio", "file", "filename", "file_name", "path")

_SOURCE_HINTS: list[tuple[Source, tuple[str, ...]]] = [
    (Source.WHATSAPP, ("whatsapp",)),
    (Source.SMS, ("sms",)),
    (Source.SOCIAL, ("twitter", "tweet", "facebook", "instagram", "social", "reddit", "youtube")),
    (Source.CALL, ("call", "voice", "audio", "phone", "helpline", "ivr", "recording")),
]

_TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M",
    "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M",
    "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M",
    "%d/%m/%y %H:%M:%S", "%d/%m/%y %H:%M", "%d/%m/%y, %I:%M %p", "%d/%m/%Y, %I:%M %p",
    "%d/%m/%Y %I:%M %p", "%d-%m-%Y %I:%M %p",
    "%d %b %Y %H:%M:%S", "%d %b %Y %H:%M", "%d %B %Y %H:%M:%S", "%d %B %Y %H:%M",
    "%b %d, %Y %H:%M:%S", "%b %d, %Y %H:%M", "%b %d, %Y %I:%M %p",
)
_DATE_ONLY_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d %b %Y", "%d %B %Y")
_CLOCK_FORMATS = ("%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M:%S %p")

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufeff]")
_WAVE_IN_PATH = re.compile(r"(?:wave|stage|phase)[\s_-]*([1-9])", re.IGNORECASE)
_STAMP_IN_NAME = re.compile(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})[T_\- ]?(\d{2})[-_:]?(\d{2})[-_:]?(\d{2})")

csv.field_size_limit(10_000_000)


# =====================================================================
# Result types
# =====================================================================

class IngestError(Exception):
    """A single input could not be used. The message is shown to the user as the reason."""


@dataclass
class IngestResult:
    messages: list[Message] = field(default_factory=list)
    rejected: list[RejectedInput] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    files_seen: int = 0

    def summary(self) -> str:
        by_source = Counter(m.source.value for m in self.messages)
        parts = ", ".join(f"{k}: {v}" for k, v in sorted(by_source.items())) or "none"
        return (f"{len(self.messages)} messages ready ({parts}); "
                f"{len(self.rejected)} rejected; {len(self.warnings)} warnings; "
                f"{self.files_seen} files read")


class Transcriber(Protocol):
    def transcribe(self, path: Path) -> str: ...


# =====================================================================
# Small helpers
# =====================================================================

def clean_text(value: Any) -> str:
    """Normalise a raw cell into tidy text. Keeps Tamil/Hindi/etc. untouched."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    text = _CONTROL_CHARS.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def has_content(text: str) -> bool:
    """True if there is at least one letter or digit (rejects things like '...' or '???')."""
    return any(ch.isalnum() for ch in text)


def guess_source(hint: Any) -> Source | None:
    """Work out SMS / WhatsApp / social / call from a column value or a file name."""
    if hint is None:
        return None
    s = str(hint).strip().lower()
    if not s:
        return None
    for source, words in _SOURCE_HINTS:
        if any(w in s for w in words):
            return source
    if s in {"x", "text", "txt", "message"}:
        return Source.SOCIAL if s == "x" else None
    return None


def _norm_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


def _pick(row: dict[str, Any], aliases: Iterable[str]) -> Any:
    """First non-empty value among the aliases (aliases are in priority order)."""
    for key in aliases:
        value = row.get(key)
        if value is not None and clean_text(value) != "":
            return value
    return None


def _is_clock_only(value: Any) -> bool:
    s = clean_text(value)
    return bool(s) and ":" in s and not re.search(r"[/\-.]\d|\d[/\-.]|[a-z]{3,}\s+\d", s, re.IGNORECASE)


def parse_time(value: Any, default_date: date) -> datetime | None:
    """Understand many time styles. Returns a timezone-naive datetime in local time, or None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime.combine(value, datetime.min.time())
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        seconds = value / 1000 if value > 1e11 else value        # milliseconds vs seconds
        if seconds < 1e8:                                        # too small to be a real timestamp
            return None
        try:
            dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    else:
        s = clean_text(value)
        if not s:
            return None
        if re.fullmatch(r"\d{9,13}(\.\d+)?", s):
            return parse_time(float(s), default_date)
        dt = None
        iso = s[:-1] + "+00:00" if s.endswith(("Z", "z")) else s
        try:
            dt = datetime.fromisoformat(iso)
        except ValueError:
            for fmt in _TIME_FORMATS:
                try:
                    dt = datetime.strptime(s, fmt)
                    break
                except ValueError:
                    continue
        if dt is None:
            for fmt in _CLOCK_FORMATS:                           # "08:10" -> use the default date
                try:
                    clock = datetime.strptime(s.upper(), fmt).time()
                    return datetime.combine(default_date, clock)
                except ValueError:
                    continue
            for fmt in _DATE_ONLY_FORMATS:
                try:
                    return datetime.strptime(s, fmt)
                except ValueError:
                    continue
            return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(LOCAL_TZ).replace(tzinfo=None)
    return dt


def time_from_filename(name: str) -> datetime | None:
    """Audio files are often named like call_20260301_081500.wav."""
    m = _STAMP_IN_NAME.search(name)
    if not m:
        return None
    try:
        return datetime(*(int(g) for g in m.groups()))
    except ValueError:
        return None


def read_text_file(path: Path) -> str:
    """Read a text file whatever its encoding (UTF-8, UTF-16, Windows-1252)."""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def wave_from_path(path: Path) -> int | None:
    m = _WAVE_IN_PATH.search(str(path))
    return int(m.group(1)) if m else None


def source_from_path(path: Path, include_dirs: bool = True) -> Source | None:
    stem_guess = guess_source(path.stem)
    if stem_guess or not include_dirs:
        return stem_guess
    for parent in path.parents:
        g = guess_source(parent.name)
        if g:
            return g
    return None


# =====================================================================
# Audio -> text  (local Whisper, with a disk cache so reruns are instant)
# =====================================================================

class WhisperTranscriber:
    """Speech-to-text using faster-whisper. The model loads only when first needed."""

    def __init__(self, model_size: str = "small", device: str = "auto",
                 compute_type: str = "int8", cache_dir: str | Path | None = "cache/transcripts"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model = None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise IngestError("Audio support is not installed. Run: pip install faster-whisper") from exc
            log.info("Loading Whisper model '%s' (first time can take a minute)...", self.model_size)
            self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        return self._model

    def _cache_file(self, path: Path) -> Path | None:
        if not self.cache_dir:
            return None
        stat = path.stat()
        key = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{self.model_size}"
        return self.cache_dir / (hashlib.sha1(key.encode()).hexdigest() + ".json")

    def transcribe(self, path: Path) -> str:
        cache_file = self._cache_file(path)
        if cache_file and cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))["text"]
            except (ValueError, KeyError, OSError):
                pass  # damaged cache entry: just transcribe again
        model = self._load()
        try:
            segments, info = model.transcribe(str(path), vad_filter=True, beam_size=5)
            text = clean_text(" ".join(seg.text for seg in segments))
        except Exception as exc:  # corrupt / unsupported audio
            raise IngestError(f"Audio could not be decoded ({type(exc).__name__})") from exc
        if not has_content(text):
            raise IngestError("No speech detected (silent or unreadable audio)")
        if cache_file:
            try:
                cache_file.write_text(json.dumps({"text": text, "language": getattr(info, 'language', None)},
                                                 ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass
        return text


# =====================================================================
# The main class
# =====================================================================

@dataclass
class _FileContext:
    path: Path
    root: Path
    default_source: Source | None
    wave: int | None
    fallback_base: datetime          # used when a row has no readable time
    time_guessed: int = 0

    @property
    def rel(self) -> str:
        try:
            return str(self.path.relative_to(self.root))
        except ValueError:
            return self.path.name


class Ingestor:
    def __init__(self, transcriber: Transcriber | None = None, default_date: date | None = None,
                 max_text_chars: int = 5000):
        self._transcriber = transcriber
        self.default_date = default_date or date.today()
        self.max_text_chars = max_text_chars
        self._result = IngestResult()
        self._claimed_audio: set[Path] = set()

    # ---- public entry point -------------------------------------------------

    def run(self, path: str | Path) -> IngestResult:
        root = Path(path)
        self._result = IngestResult()
        self._claimed_audio = set()

        if not root.exists():
            self._reject(str(root), "Path does not exist")
            return self._result

        files = [root] if root.is_file() else sorted(
            p for p in root.rglob("*") if p.is_file() and not any(part.startswith(".") for part in p.parts[len(root.parts):]))
        base = root.parent if root.is_file() else root

        # Tables first, so audio files they point to are not transcribed twice.
        for f in (f for f in files if f.suffix.lower() in TABLE_EXTS):
            self._safe(self._read_table, f, base)
        for f in (f for f in files if f.suffix.lower() in AUDIO_EXTS and f.resolve() not in self._claimed_audio):
            self._safe(self._read_audio_file, f, base)
        for f in files:
            ext = f.suffix.lower()
            if ext not in TABLE_EXTS | AUDIO_EXTS | IGNORED_EXTS:
                self._reject(self._rel(f, base), f"Unsupported file type '{ext or 'none'}'")
                self._result.files_seen += 1

        self._finish()
        return self._result

    # ---- bookkeeping --------------------------------------------------------

    def _rel(self, path: Path, base: Path) -> str:
        try:
            return str(path.relative_to(base))
        except ValueError:
            return path.name

    def _reject(self, origin: str, reason: str) -> None:
        self._result.rejected.append(RejectedInput(origin=origin, reason=reason))
        log.warning("Rejected %s: %s", origin, reason)

    def _safe(self, func, path: Path, base: Path) -> None:
        """Run one file's reader; a failure in one file must never stop the others."""
        self._result.files_seen += 1
        try:
            func(path, base)
        except IngestError as exc:
            self._reject(self._rel(path, base), str(exc))
        except Exception as exc:  # last safety net
            log.exception("Unexpected problem in %s", path)
            self._reject(self._rel(path, base), f"Unexpected error while reading file ({type(exc).__name__}: {exc})")

    def _finish(self) -> None:
        """Make ids unique, sort by time, report timing guesses."""
        seen: Counter[str] = Counter()
        for m in self._result.messages:
            seen[m.id] += 1
            if seen[m.id] > 1:
                m.id = f"{m.id}-dup{seen[m.id]}"
        self._result.messages.sort(key=lambda m: (m.timestamp, m.id))

    # ---- table files --------------------------------------------------------

    def _read_table(self, path: Path, base: Path) -> None:
        if path.stat().st_size == 0:
            raise IngestError("File is empty (0 bytes)")
        ext = path.suffix.lower()
        stat_time = datetime.fromtimestamp(path.stat().st_mtime)
        ctx = _FileContext(
            path=path, root=base,
            default_source=source_from_path(path) or None,
            wave=wave_from_path(path.relative_to(base) if path.is_relative_to(base) else path),
            fallback_base=stat_time,
        )
        if ext in {".csv", ".tsv"}:
            rows = self._rows_from_csv(path, "\t" if ext == ".tsv" else None)
        elif ext == ".json":
            rows = self._rows_from_json(path)
        elif ext in {".jsonl", ".ndjson"}:
            rows = self._rows_from_jsonl(path, ctx)
        elif ext in {".xlsx", ".xlsm"}:
            rows = self._rows_from_excel(path)
        else:
            rows = self._rows_from_txt(path)

        count = 0
        for number, row in rows:
            count += 1
            origin = f"{ctx.rel}:row {number}"
            if isinstance(row, _BadRow):
                self._reject(origin, row.reason)
                continue
            try:
                self._result.messages.append(self._row_to_message(row, ctx, number))
            except IngestError as exc:
                self._reject(origin, str(exc))
        if count == 0:
            raise IngestError("File has no data rows")
        if ctx.time_guessed:
            self._result.warnings.append(
                f"{ctx.rel}: {ctx.time_guessed} rows had no readable time; file order was used instead")

    def _rows_from_csv(self, path: Path, forced_delimiter: str | None):
        text = read_text_file(path)
        if not text.strip():
            raise IngestError("File contains only blank space")
        delimiter = forced_delimiter
        if delimiter is None:
            try:
                delimiter = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|").delimiter
            except csv.Error:
                delimiter = ","
        reader = csv.reader(text.splitlines(), delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            raise IngestError("File has no header row")
        keys = [_norm_key(h) for h in header]
        if not any(keys):
            raise IngestError("Header row is blank")
        for number, values in enumerate(reader, start=2):
            row = {k: v for k, v in zip(keys, values) if k}
            yield number, row

    def _rows_from_json(self, path: Path):
        try:
            data = json.loads(read_text_file(path))
        except json.JSONDecodeError as exc:
            raise IngestError(f"Malformed JSON (line {exc.lineno}, column {exc.colno}): {exc.msg}")
        if isinstance(data, dict):
            lists = [v for v in data.values() if isinstance(v, list)]
            data = max(lists, key=len) if lists else [data]      # {"data":[...]} or a single record
        if not isinstance(data, list):
            raise IngestError("JSON does not contain a list of reports")
        for number, item in enumerate(data, start=1):
            yield number, self._json_item_to_row(item)

    def _rows_from_jsonl(self, path: Path, ctx: _FileContext):
        for number, line in enumerate(read_text_file(path).splitlines(), start=1):
            if not line.strip():
                continue
            try:
                yield number, self._json_item_to_row(json.loads(line))
            except json.JSONDecodeError as exc:
                yield number, _BadRow(f"Malformed JSON line: {exc.msg}")

    @staticmethod
    def _json_item_to_row(item: Any):
        if isinstance(item, dict):
            return {_norm_key(k): v for k, v in item.items()}
        if isinstance(item, str):
            return {"text": item}
        return _BadRow(f"Report is not an object or text (found {type(item).__name__})")

    def _rows_from_excel(self, path: Path):
        try:
            from openpyxl import load_workbook
        except ImportError as exc:
            raise IngestError("Excel support is not installed. Run: pip install openpyxl") from exc
        try:
            workbook = load_workbook(path, read_only=True, data_only=True)
        except Exception as exc:
            raise IngestError(f"Excel file could not be opened ({type(exc).__name__})") from exc
        number = 0
        for sheet in workbook.worksheets:
            header = None
            for index, values in enumerate(sheet.iter_rows(values_only=True), start=1):
                if header is None:
                    if any(v is not None and str(v).strip() for v in values):
                        header = [_norm_key(v) if v is not None else "" for v in values]
                    continue
                number += 1
                yield number, {k: v for k, v in zip(header, values) if k}

    def _rows_from_txt(self, path: Path):
        for number, line in enumerate(read_text_file(path).splitlines(), start=1):
            if line.strip():
                yield number, {"text": line}

    # ---- one table row -> one Message --------------------------------------

    def _row_to_message(self, row: dict[str, Any], ctx: _FileContext, number: int) -> Message:
        if not any(clean_text(v) for v in row.values()):
            raise IngestError("Row is empty")

        text = clean_text(_pick(row, TEXT_KEYS))
        audio_ref = self._find_audio_ref(row, ctx)
        from_audio = False

        if audio_ref is not None:
            self._claimed_audio.add(audio_ref.resolve())
            if not text:
                text = self._transcribe(audio_ref)
                from_audio = True
        elif not text:
            raise IngestError("Row has no message text")

        if not has_content(text):
            raise IngestError("Message has no readable content")
        if len(text) > self.max_text_chars:
            text = text[: self.max_text_chars]
            self._result.warnings.append(f"{ctx.rel}:row {number}: very long text was cut to {self.max_text_chars} characters")

        timestamp = self._row_time(row, ctx, number)

        source = (guess_source(_pick(row, SOURCE_KEYS)) or ctx.default_source
                  or (Source.CALL if from_audio else Source.UNKNOWN))
        sender = clean_text(_pick(row, SENDER_KEYS)) or "unknown"
        raw_id = clean_text(_pick(row, ID_KEYS)) or f"{ctx.path.stem}-{number:05d}"

        wave = ctx.wave
        wave_raw = _pick(row, WAVE_KEYS)
        if wave_raw is not None:
            m = re.search(r"\d+", str(wave_raw))
            wave = int(m.group()) if m else wave

        try:
            return Message(id=raw_id, timestamp=timestamp, source=source, text=text,
                           sender=sender, wave=wave, origin=f"{ctx.rel}:row {number}")
        except ValueError as exc:
            raise IngestError(str(exc)) from exc

    def _row_time(self, row: dict[str, Any], ctx: _FileContext, number: int) -> datetime:
        date_val, time_val = row.get("date"), row.get("time")
        if clean_text(date_val) and clean_text(time_val) and _is_clock_only(time_val):
            combined = parse_time(f"{clean_text(date_val)} {clean_text(time_val)}", self.default_date)
            if combined:
                return combined
        for key in TIME_KEYS:
            value = row.get(key)
            if value is None or clean_text(value) == "":
                continue
            parsed = parse_time(value, self.default_date)
            if parsed:
                return parsed
        ctx.time_guessed += 1
        return ctx.fallback_base + timedelta(seconds=number)

    def _find_audio_ref(self, row: dict[str, Any], ctx: _FileContext) -> Path | None:
        """If the row names an audio file, find it on disk (or reject with a clear reason)."""
        for key in AUDIO_KEYS:
            value = clean_text(row.get(key))
            if not value or Path(value).suffix.lower() not in AUDIO_EXTS:
                continue
            name = Path(value.replace("\\", "/"))
            for candidate in (ctx.path.parent / name, ctx.root / name, *ctx.root.rglob(name.name)):
                if candidate.is_file():
                    return candidate
            raise IngestError(f"Audio file '{name.name}' was named in the table but not found")
        return None

    # ---- audio files --------------------------------------------------------

    def _read_audio_file(self, path: Path, base: Path) -> None:
        rel = self._rel(path, base)
        text = self._transcribe(path)
        timestamp = (time_from_filename(path.name)
                     or datetime.fromtimestamp(path.stat().st_mtime))
        if time_from_filename(path.name) is None:
            self._result.warnings.append(f"{rel}: no time in file name; file modified time was used")
        # A voice note from WhatsApp stays WhatsApp; everything else is treated as a phone call.
        audio_source = Source.WHATSAPP if source_from_path(path) is Source.WHATSAPP else Source.CALL
        try:
            self._result.messages.append(Message(
                id=path.stem, timestamp=timestamp,
                source=audio_source,
                text=text, sender="unknown",
                wave=wave_from_path(Path(rel)), origin=rel))
        except ValueError as exc:
            raise IngestError(str(exc)) from exc

    def _transcribe(self, path: Path) -> str:
        if not path.exists():
            raise IngestError("Audio file not found")
        if path.stat().st_size == 0:
            raise IngestError("Audio file is empty (0 bytes)")
        if self._transcriber is None:
            self._transcriber = WhisperTranscriber()
        text = clean_text(self._transcriber.transcribe(path))
        if not has_content(text):
            raise IngestError("No speech detected (silent or unreadable audio)")
        return text


@dataclass
class _BadRow:
    reason: str


# =====================================================================
# Command line + self-test
# =====================================================================

def write_outputs(result: IngestResult, out_path: Path) -> None:
    payload = {"messages": [m.to_dict() for m in result.messages],
               "rejected": [r.to_dict() for r in result.rejected],
               "warnings": result.warnings}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _selftest() -> None:
    import tempfile
    import wave as wavelib

    class FakeTranscriber:
        def transcribe(self, path: Path) -> str:
            if "silent" in path.name:
                raise IngestError("No speech detected (silent or unreadable audio)")
            return f"transcript of {path.stem}"

    def write_wav(path: Path) -> None:
        with wavelib.open(str(path), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
            w.writeframes(b"\x00\x00" * 800)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "stage_1").mkdir(); (root / "audio").mkdir()

        # CSV with unusual column names, an empty message, a "no content" message, a bad time
        (root / "stage_1" / "whatsapp_export.csv").write_text(
            "Time,From,Message\n"
            "26/09/2025 08:10,+911111111111,first report text\n"
            "26/09/2025 08:12,+912222222222,\n"
            "26/09/2025 08:13,+913333333333,...\n"
            "not a time,+914444444444,report with a bad time\n", encoding="utf-8")
        # JSON wrapped in an object, with epoch + ISO times and a Twitter channel
        (root / "reports.json").write_text(json.dumps({"data": [
            {"id": "t1", "body": "social post text", "created_at": "2025-09-26T02:45:00Z", "channel": "Twitter"},
            {"id": "t1", "body": "same id different text", "created_at": 1758854700},
            "plain string report",
            12345]}), encoding="utf-8")
        (root / "broken.json").write_text("{not valid json", encoding="utf-8")
        (root / "sms.jsonl").write_text('{"text": "sms line one", "time": "2025-09-26 09:00:00"}\n{oops\n', encoding="utf-8")
        (root / "zero.csv").write_bytes(b"")
        (root / "notes.txt").write_text("line one\n\nline two\n", encoding="utf-8")
        (root / "mystery.xyz").write_text("???", encoding="utf-8")
        # table that points at an audio file, plus date/time in separate columns
        write_wav(root / "audio" / "call_a.wav")
        (root / "calls.csv").write_text(
            "date,time,caller,audio_file\n"
            "26/09/2025,08:30,+915555555555,call_a.wav\n"
            "26/09/2025,08:31,+916666666666,missing.wav\n", encoding="utf-8")
        # stand-alone audio files
        write_wav(root / "audio" / "call_20250926_084500.wav")
        write_wav(root / "audio" / "silent_call.wav")
        (root / "audio" / "empty.wav").write_bytes(b"")

        result = Ingestor(transcriber=FakeTranscriber(), default_date=date(2025, 9, 26)).run(root)
        texts = {m.text for m in result.messages}
        reasons = " | ".join(r.reason for r in result.rejected)
        print(result.summary())
        for r in result.rejected:
            print("  rejected:", r.origin, "->", r.reason)

        assert "first report text" in texts and "transcript of call_a" in texts
        assert "transcript of call_20250926_084500" in texts and "plain string report" in texts
        assert "line one" in texts and "line two" in texts and "sms line one" in texts
        assert "report with a bad time" in texts                       # kept, with guessed time
        assert not any(m.text == "..." for m in result.messages)       # no-content row rejected
        assert "Malformed JSON" in reasons and "0 bytes" in reasons and "Unsupported file type" in reasons
        assert "No speech detected" in reasons and "not found" in reasons and "Row has no message text" in reasons
        assert len({m.id for m in result.messages}) == len(result.messages)   # ids unique
        assert [m.timestamp for m in result.messages] == sorted(m.timestamp for m in result.messages)
        by_text = {m.text: m for m in result.messages}
        assert by_text["first report text"].wave == 1 and by_text["first report text"].source is Source.WHATSAPP
        assert by_text["first report text"].timestamp == datetime(2025, 9, 26, 8, 10)
        assert by_text["social post text"].source is Source.SOCIAL
        assert by_text["social post text"].timestamp == datetime(2025, 9, 26, 8, 15)      # 02:45Z -> IST
        assert by_text["transcript of call_a"].timestamp == datetime(2025, 9, 26, 8, 30)  # date + time columns
        assert by_text["transcript of call_a"].source is Source.CALL
        assert sum(1 for m in result.messages if m.text == "transcript of call_a") == 1   # not transcribed twice
        try:
            import openpyxl  # noqa: F401
            from openpyxl import Workbook
            wb = Workbook(); ws = wb.active
            ws.append(["Timestamp", "Text"]); ws.append([datetime(2025, 9, 26, 10, 0), "excel report"])
            wb.save(root / "sheet.xlsx")
            r2 = Ingestor(transcriber=FakeTranscriber()).run(root / "sheet.xlsx")
            assert [m.text for m in r2.messages] == ["excel report"]
            print("excel check passed")
        except ImportError:
            print("excel check skipped (openpyxl not installed)")
    print("ingest.py: all checks passed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read dataset files into clean messages.")
    parser.add_argument("path", nargs="?", help="folder or file to read")
    parser.add_argument("--out", default="messages.json", help="where to save the result (JSON)")
    parser.add_argument("--whisper-size", default="small", help="tiny, base, small, medium, large-v3")
    parser.add_argument("--selftest", action="store_true", help="run built-in checks and exit")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.selftest:
        _selftest()
        return 0
    if not args.path:
        parser.error("give a folder or file to read, or use --selftest")
    result = Ingestor(transcriber=WhisperTranscriber(model_size=args.whisper_size)).run(args.path)
    write_outputs(result, Path(args.out))
    print(result.summary())
    print(f"Saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())