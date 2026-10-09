"""db.py - the filing cabinet of the assistant.

WHAT IT DOES
    Saves everything the pipeline produced into ONE SQLite file (a normal file on your
    disk, no server needed) and reads it back for the dashboard.

WHY IT EXISTS
    - The dashboard (app.py) must be fast and must NEVER call the AI. It only reads this file.
    - The AI and the grouping take a long time. They run once; the dashboard reads the answer.
    - It remembers the state of the incidents after EACH wave, so the dashboard can show
      "the situation as of wave 2" (the replay).

WHAT GOES IN / OUT
    In : Message, Extraction, Incident, Decision, HeldReport, RejectedInput (from models.py / cluster.py)
    Out: the same objects again, plus counts for the headline numbers.

DOES IT USE AI?  No. Plain storage code.

SAFE TO RE-RUN
    Saving the same data twice gives the same result (no duplicates).
    Results are replaced in ONE step ("all or nothing"), so the dashboard never sees a half-written state.

RUN
    python db.py --selftest            check that it works
    python db.py --info emergency.db   print what is inside a database file
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from models import Extraction, Incident, Message, RejectedInput

SCHEMA_VERSION = "1"
DEFAULT_DB = "emergency.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id        TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    source    TEXT NOT NULL,
    sender    TEXT NOT NULL,
    text      TEXT NOT NULL,
    wave      INTEGER,
    origin    TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_messages_time ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_wave ON messages(wave);

CREATE TABLE IF NOT EXISTS extractions (
    message_id TEXT PRIMARY KEY,
    data       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    message_id            TEXT PRIMARY KEY,
    action                TEXT NOT NULL,
    incident_id           TEXT,
    score                 REAL NOT NULL DEFAULT 0,
    possible_duplicate_of TEXT,
    data                  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_action   ON decisions(action);
CREATE INDEX IF NOT EXISTS idx_decisions_incident ON decisions(incident_id);

CREATE TABLE IF NOT EXISTS incidents (
    id           TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    last_update  TEXT NOT NULL,
    status       TEXT NOT NULL,
    urgency      TEXT NOT NULL,
    urgency_rank INTEGER NOT NULL,
    people_count INTEGER,
    resource     TEXT NOT NULL,
    title        TEXT NOT NULL,
    lat          REAL,
    lon          REAL,
    loc_conf     REAL,
    report_count INTEGER NOT NULL,
    search_text  TEXT NOT NULL,
    data         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS snapshots (
    wave         INTEGER NOT NULL,
    id           TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    last_update  TEXT NOT NULL,
    status       TEXT NOT NULL,
    urgency      TEXT NOT NULL,
    urgency_rank INTEGER NOT NULL,
    people_count INTEGER,
    resource     TEXT NOT NULL,
    title        TEXT NOT NULL,
    lat          REAL,
    lon          REAL,
    loc_conf     REAL,
    report_count INTEGER NOT NULL,
    search_text  TEXT NOT NULL,
    data         TEXT NOT NULL,
    PRIMARY KEY (wave, id)
);

CREATE TABLE IF NOT EXISTS held (
    message_id          TEXT PRIMARY KEY,
    reason              TEXT NOT NULL,
    related_incident_id TEXT
);

CREATE TABLE IF NOT EXISTS rejected (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    origin    TEXT NOT NULL,
    reason    TEXT NOT NULL,
    timestamp TEXT
);
"""

# Tables that hold the pipeline's RESULTS (replaced as one block).
_RESULT_TABLES = ("decisions", "incidents", "snapshots", "held")
_INCIDENT_COLUMNS = ("id", "created_at", "last_update", "status", "urgency", "urgency_rank", "people_count",
                     "resource", "title", "lat", "lon", "loc_conf", "report_count", "search_text", "data")

# Order used on the dashboard: open incidents first, most urgent first, then newest news first.
_ATTENTION_ORDER = ("(status = 'resolved') ASC, urgency_rank DESC, "
                    "(status = 'rescue_in_progress') ASC, last_update DESC, id ASC")
_RECENT_ORDER = "last_update DESC, id ASC"
_SQL_CHUNK = 400   # SQLite limits how many "?" one query may have; we ask for ids in small groups.


class DatabaseError(Exception):
    """A problem with the database file, with a plain-language message."""


# =====================================================================
# Small result shapes (what the dashboard receives)
# =====================================================================

@dataclass
class Report:
    """One message together with what the AI read from it and what the grouping step decided."""
    message: Message
    extraction: Extraction | None
    decision: dict          # Decision.to_dict(): action, incident_id, score, reasons, note, ...


@dataclass
class HeldRow:
    """A report kept for a human to look at (rumor or unclear). It never changed any incident."""
    message: Message
    extraction: Extraction | None
    reason: str
    related_incident_id: str | None
    kind: str               # "rumor" or "unclear"


# =====================================================================
# Helpers
# =====================================================================

def _ts(value: datetime | str | None) -> str | None:
    """Times are saved in ONE fixed text format so sorting by text = sorting by time."""
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.strftime("%Y-%m-%dT%H:%M:%S")


def _parse(value: str | None) -> datetime:
    return datetime.fromisoformat(value) if value else datetime.min


def _as_dict(obj: Any) -> dict:
    return obj if isinstance(obj, dict) else obj.to_dict()


def _chunks(items: Sequence, size: int = _SQL_CHUNK) -> Iterable[Sequence]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _like(word: str) -> str:
    word = word.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{word}%"


def _incident_row(d: dict) -> dict:
    """Pull the searchable columns out of an Incident dict. The full JSON is kept too."""
    loc = d.get("location") or {}
    title = loc.get("name") or d.get("location_text") or "Location unknown"
    pieces = [d.get("id", ""), title, d.get("location_text", ""), loc.get("matched_text", ""),
              loc.get("qualifier", ""), d.get("summary", ""), d.get("resource_detail", ""),
              d.get("resource", ""), d.get("status", ""), d.get("urgency", ""),
              " ".join(d.get("vulnerable", [])), " ".join(d.get("needs", []))]
    inc = Incident.from_dict(d)          # also cleans odd values (e.g. messy urgency text)
    return {
        "id": inc.id, "created_at": _ts(inc.created_at), "last_update": _ts(inc.last_update),
        "status": inc.status.value, "urgency": inc.urgency.value, "urgency_rank": inc.urgency.rank,
        "people_count": inc.people_count, "resource": inc.resource.value, "title": title,
        "lat": loc.get("lat"), "lon": loc.get("lon"), "loc_conf": loc.get("confidence"),
        "report_count": len(inc.message_ids),
        "search_text": " ".join(str(p) for p in pieces if p).lower(),
        "data": json.dumps(inc.to_dict(), ensure_ascii=False),
    }


# =====================================================================
# The database
# =====================================================================

class Database:
    """Open with Database("emergency.db"). The dashboard opens it with readonly=True,
    which makes writing impossible (so the dashboard cannot damage the data)."""

    def __init__(self, path: str | Path = DEFAULT_DB, readonly: bool = False):
        self.path = Path(path)
        self.readonly = readonly
        if readonly and not self.path.is_file():
            raise DatabaseError(f"Database file '{self.path}' not found. Run pipeline.py first to create it.")
        try:
            self.conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not open '{self.path}': {exc}") from exc
        self.conn.row_factory = sqlite3.Row
        try:
            if not readonly:
                self.conn.execute("PRAGMA journal_mode=WAL")      # lets the dashboard read while the pipeline writes
                self.conn.executescript(SCHEMA)
                self._check_version(create=True)
            else:
                self._check_version(create=False)
                self.conn.execute("PRAGMA query_only=ON")
        except sqlite3.DatabaseError as exc:
            self.conn.close()
            raise DatabaseError(f"'{self.path}' is not a usable database ({exc}). "
                                "Delete it and run pipeline.py again.") from exc

    def _check_version(self, create: bool) -> None:
        try:
            row = self.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        except sqlite3.OperationalError:
            raise DatabaseError(f"'{self.path}' is not an assistant database. Run pipeline.py first.")
        if row is None:
            if create:
                with self.conn:
                    self.conn.execute("INSERT INTO meta(key, value) VALUES('schema_version', ?)", (SCHEMA_VERSION,))
            return
        if row["value"] != SCHEMA_VERSION:
            raise DatabaseError(f"'{self.path}' was made by a different version of the program "
                                f"(file {row['value']}, program {SCHEMA_VERSION}). "
                                "Delete the file and run pipeline.py again.")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # WRITING
    # ------------------------------------------------------------------

    def clear_all(self) -> None:
        """Empty everything except the schema. The database is a product of the pipeline,
        so wiping and rebuilding is safe (AI answers are cached on disk elsewhere)."""
        with self.conn:
            for table in ("messages", "extractions", "rejected", *_RESULT_TABLES):
                self.conn.execute(f"DELETE FROM {table}")
            self.conn.execute("DELETE FROM meta WHERE key != 'schema_version'")

    def save_messages(self, messages: Iterable[Message]) -> int:
        rows = [(m.id, _ts(m.timestamp), m.source.value, m.sender or "unknown", m.text, m.wave, m.origin or "")
                for m in messages]
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO messages(id, timestamp, source, sender, text, wave, origin) "
                "VALUES (?,?,?,?,?,?,?)", rows)
        return len(rows)

    def save_extractions(self, extractions: dict[str, Extraction]) -> int:
        rows = [(mid, json.dumps(_as_dict(e), ensure_ascii=False)) for mid, e in extractions.items()]
        with self.conn:
            self.conn.executemany("INSERT OR REPLACE INTO extractions(message_id, data) VALUES (?,?)", rows)
        return len(rows)

    def save_rejected(self, rejected: Iterable[RejectedInput | dict]) -> int:
        """Replaces the whole rejected list (it comes from one ingest run)."""
        rows = []
        for r in rejected:
            d = _as_dict(r)
            rows.append((str(d.get("origin", "")), str(d.get("reason", "")), _ts(d.get("timestamp"))))
        with self.conn:
            self.conn.execute("DELETE FROM rejected")
            self.conn.executemany("INSERT INTO rejected(origin, reason, timestamp) VALUES (?,?,?)", rows)
        return len(rows)

    def save_results(self, incidents: Iterable[Incident | dict], decisions: Iterable[Any],
                     held: Iterable[Any], snapshots: dict[int, list[dict]] | None = None,
                     stats: dict | None = None) -> None:
        """Replace ALL results in one all-or-nothing step.

        incidents : final incident list         decisions : Decision objects (or dicts)
        held      : HeldReport objects (or dicts)
        snapshots : {wave number: list of incident dicts as they were after that wave}
        stats     : any dict of counters (kept for reference)
        """
        incident_rows = [_incident_row(_as_dict(i)) for i in incidents]
        decision_rows = []
        for d in (_as_dict(x) for x in decisions):
            decision_rows.append((d["message_id"], d["action"], d.get("incident_id"), float(d.get("score") or 0),
                                  d.get("possible_duplicate_of"), json.dumps(d, ensure_ascii=False)))
        held_rows = []
        for h in (_as_dict(x) for x in held):
            message_id = (h.get("message") or {}).get("id") or h.get("message_id")
            held_rows.append((message_id, h.get("reason", ""), h.get("related_incident_id")))
        snapshot_rows = []
        for wave, items in (snapshots or {}).items():
            for d in items:
                row = _incident_row(_as_dict(d))
                snapshot_rows.append((int(wave), *[row[c] for c in _INCIDENT_COLUMNS]))

        columns = ", ".join(_INCIDENT_COLUMNS)
        marks = ", ".join("?" for _ in _INCIDENT_COLUMNS)
        with self.conn:      # one transaction: everything below is saved together or not at all
            for table in _RESULT_TABLES:
                self.conn.execute(f"DELETE FROM {table}")
            self.conn.executemany(f"INSERT INTO incidents({columns}) VALUES ({marks})",
                                  [tuple(r[c] for c in _INCIDENT_COLUMNS) for r in incident_rows])
            self.conn.executemany(f"INSERT INTO snapshots(wave, {columns}) VALUES (?, {marks})", snapshot_rows)
            self.conn.executemany(
                "INSERT INTO decisions(message_id, action, incident_id, score, possible_duplicate_of, data) "
                "VALUES (?,?,?,?,?,?)", decision_rows)
            self.conn.executemany("INSERT OR REPLACE INTO held(message_id, reason, related_incident_id) VALUES (?,?,?)",
                                  held_rows)
            if stats is not None:
                self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('cluster_stats', ?)",
                                  (json.dumps(stats, ensure_ascii=False),))

    def set_meta(self, key: str, value: Any) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                              (key, json.dumps(value, ensure_ascii=False, default=str)))

    # ------------------------------------------------------------------
    # READING: small things
    # ------------------------------------------------------------------

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return default

    def waves(self) -> list[int]:
        """Waves that have a saved snapshot (used for the wave tabs / replay)."""
        return [r[0] for r in self.conn.execute("SELECT DISTINCT wave FROM snapshots ORDER BY wave")]

    def get_rejected(self) -> list[RejectedInput]:
        rows = self.conn.execute("SELECT origin, reason, timestamp FROM rejected ORDER BY id").fetchall()
        return [RejectedInput(r["origin"], r["reason"], _parse(r["timestamp"])) for r in rows]

    @staticmethod
    def _message(row: sqlite3.Row) -> Message:
        return Message(id=row["id"], timestamp=_parse(row["timestamp"]), source=row["source"], text=row["text"],
                       sender=row["sender"], wave=row["wave"], origin=row["origin"])

    def get_message(self, message_id: str) -> Message | None:
        row = self.conn.execute("SELECT * FROM messages WHERE id=?", (message_id,)).fetchone()
        return self._message(row) if row else None

    def _messages_by_ids(self, ids: Sequence[str]) -> dict[str, Message]:
        found: dict[str, Message] = {}
        for part in _chunks(list(ids)):
            q = ",".join("?" for _ in part)
            for row in self.conn.execute(f"SELECT * FROM messages WHERE id IN ({q})", list(part)):
                found[row["id"]] = self._message(row)
        return found

    def _extractions_by_ids(self, ids: Sequence[str]) -> dict[str, Extraction]:
        found: dict[str, Extraction] = {}
        for part in _chunks(list(ids)):
            q = ",".join("?" for _ in part)
            for row in self.conn.execute(f"SELECT message_id, data FROM extractions WHERE message_id IN ({q})",
                                         list(part)):
                found[row["message_id"]] = Extraction.from_dict(json.loads(row["data"]))
        return found

    def _decisions_by_ids(self, ids: Sequence[str]) -> dict[str, dict]:
        found: dict[str, dict] = {}
        for part in _chunks(list(ids)):
            q = ",".join("?" for _ in part)
            for row in self.conn.execute(f"SELECT message_id, data FROM decisions WHERE message_id IN ({q})",
                                         list(part)):
                found[row["message_id"]] = json.loads(row["data"])
        return found

    # ------------------------------------------------------------------
    # READING: incidents
    # ------------------------------------------------------------------

    def _table_for(self, as_of_wave: int | None) -> tuple[str, int | None, bool]:
        """Which table to read. Returns (table, wave to use, has_data).
        None = latest state. A number = the saved state after that wave (or the last wave before it)."""
        if as_of_wave is None:
            return "incidents", None, True
        row = self.conn.execute("SELECT MAX(wave) AS w FROM snapshots WHERE wave <= ?", (int(as_of_wave),)).fetchone()
        if row is None or row["w"] is None:
            return "snapshots", None, False
        return "snapshots", int(row["w"]), True

    def list_incidents(self, as_of_wave: int | None = None, statuses: Sequence[str] | None = None,
                       urgencies: Sequence[str] | None = None, search: str = "",
                       order: str = "attention", limit: int | None = None) -> list[Incident]:
        """Incidents for the list and the map.
        statuses / urgencies: e.g. ["active", "rescue_in_progress"], ["critical", "high"]
        search: words that must ALL appear somewhere in the incident (place, summary, needs, id...)
        order : "attention" (default: most urgent open first) or "recent" (newest news first)"""
        table, wave, ok = self._table_for(as_of_wave)
        if not ok:
            return []
        where, args = [], []
        if wave is not None:
            where.append("wave = ?")
            args.append(wave)
        if statuses:
            where.append(f"status IN ({','.join('?' for _ in statuses)})")
            args += [str(s) for s in statuses]
        if urgencies:
            where.append(f"urgency IN ({','.join('?' for _ in urgencies)})")
            args += [str(u) for u in urgencies]
        for word in (search or "").lower().split():
            where.append("search_text LIKE ? ESCAPE '\\'")
            args.append(_like(word))
        sql = f"SELECT data FROM {table}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY " + (_RECENT_ORDER if order == "recent" else _ATTENTION_ORDER)
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [Incident.from_dict(json.loads(r["data"])) for r in self.conn.execute(sql, args)]

    def get_incident(self, incident_id: str, as_of_wave: int | None = None) -> Incident | None:
        table, wave, ok = self._table_for(as_of_wave)
        if not ok:
            return None
        sql, args = f"SELECT data FROM {table} WHERE id = ?", [incident_id]
        if wave is not None:
            sql += " AND wave = ?"
            args.append(wave)
        row = self.conn.execute(sql, args).fetchone()
        return Incident.from_dict(json.loads(row["data"])) if row else None

    def get_reports(self, incident_id: str, as_of_wave: int | None = None) -> list[Report]:
        """All messages grouped under one incident, newest first (the timeline of evidence)."""
        incident = self.get_incident(incident_id, as_of_wave)
        if incident is None:
            return []
        ids = list(dict.fromkeys(incident.message_ids))
        messages = self._messages_by_ids(ids)
        extractions = self._extractions_by_ids(ids)
        decisions = self._decisions_by_ids(ids)
        reports = [Report(messages[i], extractions.get(i), decisions.get(i, {})) for i in ids if i in messages]
        reports.sort(key=lambda r: (r.message.timestamp, r.message.id), reverse=True)
        return reports

    # ------------------------------------------------------------------
    # READING: set-aside messages (noise, held, late updates)
    # ------------------------------------------------------------------

    def get_reports_by_action(self, action: str, as_of_wave: int | None = None,
                              limit: int | None = None) -> list[Report]:
        """Messages the grouping step handled in one way: "noise", "ignored_resolved", "ignored_unlinked"..."""
        sql = ("SELECT m.id FROM decisions d JOIN messages m ON m.id = d.message_id "
               "WHERE d.action = ? AND COALESCE(m.wave, 0) <= ? ORDER BY m.timestamp DESC, m.id DESC")
        args: list[Any] = [action, 10 ** 9 if as_of_wave is None else int(as_of_wave)]
        if limit:
            sql += f" LIMIT {int(limit)}"
        ids = [r[0] for r in self.conn.execute(sql, args)]
        messages = self._messages_by_ids(ids)
        extractions = self._extractions_by_ids(ids)
        decisions = self._decisions_by_ids(ids)
        return [Report(messages[i], extractions.get(i), decisions.get(i, {})) for i in ids if i in messages]

    def get_held(self, as_of_wave: int | None = None, limit: int | None = None) -> list[HeldRow]:
        """Rumors and unclear reports kept for a human to review, newest first."""
        sql = ("SELECT h.message_id, h.reason, h.related_incident_id, d.action FROM held h "
               "JOIN messages m ON m.id = h.message_id LEFT JOIN decisions d ON d.message_id = h.message_id "
               "WHERE COALESCE(m.wave, 0) <= ? ORDER BY m.timestamp DESC, m.id DESC")
        args: list[Any] = [10 ** 9 if as_of_wave is None else int(as_of_wave)]
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = self.conn.execute(sql, args).fetchall()
        ids = [r["message_id"] for r in rows]
        messages = self._messages_by_ids(ids)
        extractions = self._extractions_by_ids(ids)
        out = []
        for r in rows:
            if r["message_id"] not in messages:
                continue
            kind = "rumor" if r["action"] == "held_rumor" else "unclear"
            out.append(HeldRow(messages[r["message_id"]], extractions.get(r["message_id"]),
                               r["reason"], r["related_incident_id"], kind))
        return out

    # ------------------------------------------------------------------
    # READING: headline numbers
    # ------------------------------------------------------------------

    def headline(self, as_of_wave: int | None = None) -> dict[str, int]:
        """Numbers for the top of the dashboard ("N messages became M incidents").
        as_of_wave=None means the latest state."""
        limit = 10 ** 9 if as_of_wave is None else int(as_of_wave)
        counts = {r["action"]: r["n"] for r in self.conn.execute(
            "SELECT d.action AS action, COUNT(*) AS n FROM decisions d JOIN messages m ON m.id = d.message_id "
            "WHERE COALESCE(m.wave, 0) <= ? GROUP BY d.action", (limit,))}
        possible = self.conn.execute(
            "SELECT COUNT(*) FROM decisions d JOIN messages m ON m.id = d.message_id "
            "WHERE d.possible_duplicate_of IS NOT NULL AND COALESCE(m.wave, 0) <= ?", (limit,)).fetchone()[0]
        loaded = self.conn.execute("SELECT COUNT(*) FROM messages WHERE COALESCE(wave, 0) <= ?", (limit,)).fetchone()[0]

        table, wave, ok = self._table_for(as_of_wave)
        by_status: dict[str, int] = {}
        people = 0
        if ok:
            cond, args = ("WHERE wave = ?", [wave]) if wave is not None else ("", [])
            for r in self.conn.execute(f"SELECT status, COUNT(*) AS n, COALESCE(SUM(people_count), 0) AS p "
                                       f"FROM {table} {cond} GROUP BY status", args):
                by_status[r["status"]] = r["n"]
                if r["status"] != "resolved":
                    people += r["p"]
        held_total = counts.get("held_rumor", 0) + counts.get("held_unclear", 0)
        return {
            "messages": sum(counts.values()),            # messages that went through the pipeline
            "messages_loaded": loaded,
            "incidents": sum(by_status.values()),
            "active": by_status.get("active", 0),
            "rescue_in_progress": by_status.get("rescue_in_progress", 0),
            "resolved": by_status.get("resolved", 0),
            "open": by_status.get("active", 0) + by_status.get("rescue_in_progress", 0),
            "people_in_open_incidents": people,
            "created": counts.get("created", 0),
            "merged": counts.get("merged", 0),
            "noise": counts.get("noise", 0),
            "rumors_held": counts.get("held_rumor", 0),
            "unclear_held": counts.get("held_unclear", 0),
            "held": held_total,
            "ignored_resolved": counts.get("ignored_resolved", 0),
            "ignored_unlinked": counts.get("ignored_unlinked", 0),
            "possible_duplicates": possible,
            "rejected": self.conn.execute("SELECT COUNT(*) FROM rejected").fetchone()[0],
        }


# =====================================================================
# SELF-TEST  (run:  python db.py --selftest)
# =====================================================================

def _selftest() -> None:
    import tempfile
    from cluster import Clusterer
    from models import ResolvedLocation

    # Invented places only (no real names in code).
    places = {
        "gate zone": ResolvedLocation(name="Testville Gate", landmark_id="t1", lat=13.000, lon=80.000,
                                      confidence=0.9, matched_text="gate zone", kind="landmark"),
        "river side": ResolvedLocation(name="Riverside Test Area", landmark_id="t2", lat=13.100, lon=80.100,
                                       confidence=0.9, matched_text="river side", kind="landmark"),
    }

    def locate(text):
        return places.get((text or "").strip().lower())

    def msg(i, hh, mm, text, wave, sender="s1"):
        return Message(id=i, timestamp=datetime(2026, 3, 1, hh, mm), source="sms", text=text,
                       sender=sender, wave=wave, origin="test")

    def ex(**kw):
        return Extraction(**kw)

    pairs_by_wave = {
        1: [
            (msg("m1", 8, 0, "6 trapped at gate zone, elderly diabetic needs insulin", 1),
             ex(intent="emergency", location_text="gate zone", people_count=6, resource="medical",
                resource_detail="insulin", vulnerable=["elderly"], confidence=0.9)),
            (msg("m2", 8, 10, "10 people now at gate zone, water rising", 1, "s2"),
             ex(intent="emergency", location_text="gate zone", people_count=10, resource="rescue",
                worsening=True, confidence=0.9)),
            (msg("m3", 8, 12, "Is there a power cut in my street?", 1, "s3"),
             ex(intent="noise", confidence=0.9)),
            (msg("m4", 8, 15, "Forwarded: bridge collapsed near river side!!", 1, "s4"),
             ex(intent="rumor", location_text="river side", confidence=0.6)),
        ],
        2: [
            (msg("m5", 9, 0, "Family stuck at river side, 4 people, need a boat", 2, "s5"),
             ex(intent="emergency", location_text="river side", people_count=4, resource="rescue",
                resource_detail="boat", confidence=0.9)),
            (msg("m6", 9, 30, "gate zone everyone rescued", 2, "s1"),
             ex(intent="emergency", location_text="gate zone", status_hint="resolved", confidence=0.9)),
            (msg("m7", 9, 40, "மழை நீர் அதிகரிக்கிறது river side 6 பேர்", 2, "s6"),
             ex(intent="emergency", location_text="river side", people_count=6, count_is_additional=False,
                resource="rescue", confidence=0.8)),
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.db"
        clusterer = Clusterer(locate)
        snapshots: dict[int, list[dict]] = {}
        all_messages, all_extractions = [], {}
        for wave, pairs in pairs_by_wave.items():
            clusterer.run(pairs)
            snapshots[wave] = clusterer.snapshot()
            for m, e in pairs:
                all_messages.append(m)
                all_extractions[m.id] = e
        result = clusterer.result()

        def write(db: Database) -> None:
            db.save_messages(all_messages)
            db.save_extractions(all_extractions)
            db.save_rejected([RejectedInput("broken.csv", "Malformed file", datetime(2026, 3, 1, 7, 0))])
            db.save_results(result.incidents, result.decisions, result.held, snapshots, result.stats.__dict__)

        with Database(path) as db:
            write(db)
            write(db)          # SAFE TO RE-RUN: second save must not duplicate anything
            assert db.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 7
            assert db.conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0] == len(result.incidents)
            assert db.conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 7
            assert len(db.get_rejected()) == 1

            # Incidents round-trip exactly (nothing lost or changed by saving)
            saved = {i.id: i.to_dict() for i in db.list_incidents()}
            assert saved == {i.id: i.to_dict() for i in result.incidents}
            assert db.get_incident("INC-001").to_dict() == result.incidents[0].to_dict()
            assert db.get_incident("INC-999") is None

            # Messages and extractions round-trip, including non-English text
            assert db.get_message("m7").text == pairs_by_wave[2][2][0].text
            assert db.get_message("m7").timestamp == datetime(2026, 3, 1, 9, 40)
            assert db.get_message("nope") is None

            # Ordering: open incidents come before resolved ones
            order = db.list_incidents()
            statuses = [i.status.value == "resolved" for i in order]
            assert statuses == sorted(statuses)
            assert all(a.urgency.rank >= b.urgency.rank for a, b in zip(order, order[1:])
                       if a.is_open and b.is_open and a.status == b.status)

            # Filters and search
            assert {i.id for i in db.list_incidents(statuses=["resolved"])} == \
                   {i.id for i in result.incidents if not i.is_open}
            assert all(i.is_open for i in db.list_incidents(statuses=["active", "rescue_in_progress"]))
            assert [i.title for i in db.list_incidents(search="riverside")] == \
                   [i.title for i in result.incidents if "Riverside" in i.title]
            assert db.list_incidents(search="RIVERSIDE boat")            # all words must match, any case
            assert db.list_incidents(search="riverside insulin") == []   # words from two different incidents
            assert db.list_incidents(search="100%_") == []               # odd characters are safe
            assert len(db.list_incidents(limit=1)) == 1

            # Replay: state as of wave 1 is different from the final state
            assert db.waves() == [1, 2]
            w1 = db.list_incidents(as_of_wave=1)
            assert {i.id for i in w1} == {i["id"] for i in snapshots[1]}
            assert all(i.is_open for i in w1)                            # nothing was resolved yet in wave 1
            assert any(not i.is_open for i in db.list_incidents(as_of_wave=2))
            assert db.list_incidents(as_of_wave=0) == []                 # before the first wave: nothing
            assert db.get_incident("INC-001", as_of_wave=1).is_open
            assert not db.get_incident("INC-001", as_of_wave=2).is_open
            assert db.get_incident("INC-002", as_of_wave=1) is None      # created in wave 2, not visible in wave 1

            # Headline numbers add up and respect the wave
            h = db.headline()
            assert h["messages"] == 7 == h["messages_loaded"]
            assert h["messages"] == (h["created"] + h["merged"] + h["noise"] + h["held"]
                                     + h["ignored_resolved"] + h["ignored_unlinked"])
            assert h["incidents"] == len(result.incidents) == h["open"] + h["resolved"]
            assert h["noise"] == 1 and h["rumors_held"] == 1 and h["rejected"] == 1
            h1 = db.headline(as_of_wave=1)
            assert h1["messages"] == 4 and h1["noise"] == 1 and h1["resolved"] == 0
            assert h1["incidents"] == len(w1)
            assert db.headline(as_of_wave=0)["messages"] == 0

            # Evidence lists
            reports = db.get_reports("INC-001")
            assert reports and [r.message.timestamp for r in reports] == sorted(
                (r.message.timestamp for r in reports), reverse=True)           # newest first
            assert all(r.extraction is not None and r.decision.get("incident_id") == "INC-001" for r in reports)
            assert {r.message.id for r in db.get_reports("INC-001", as_of_wave=1)} <= {"m1", "m2"}
            assert [r.message.id for r in db.get_reports_by_action("noise")] == ["m3"]
            assert db.get_reports_by_action("noise", as_of_wave=0) == []
            held = db.get_held()
            assert len(held) == 1 and held[0].message.id == "m4" and held[0].kind == "rumor"
            assert db.get_held(as_of_wave=0) == []
            assert db.get_reports("INC-999") == []

            # Meta values
            db.set_meta("run", {"note": "தமிழ்", "n": 3})
            assert db.get_meta("run") == {"note": "தமிழ்", "n": 3}
            assert db.get_meta("missing", "dflt") == "dflt"
            assert db.get_meta("cluster_stats")["messages"] == 7

        # Read-only mode (what the dashboard uses): reading works, writing is refused
        with Database(path, readonly=True) as ro:
            assert len(ro.list_incidents()) == len(result.incidents)
            try:
                ro.save_messages([all_messages[0]])
                raise AssertionError("read-only database accepted a write")
            except sqlite3.Error:
                pass
            try:
                ro.clear_all()
                raise AssertionError("read-only database accepted clear_all")
            except sqlite3.Error:
                pass

        # clear_all empties the data; the result stays usable
        with Database(path) as db:
            db.clear_all()
            assert db.headline()["messages"] == 0 and db.list_incidents() == [] and db.get_rejected() == []
            assert db.waves() == []
            write(db)
            assert db.headline()["messages"] == 7

        # A failed save changes nothing (all or nothing)
        with Database(path) as db:
            before = [i.id for i in db.list_incidents()]
            try:
                db.save_results(result.incidents, [{"action": "created"}], [], snapshots)   # decision without message_id
                raise AssertionError("bad data was accepted")
            except KeyError:
                pass
            assert [i.id for i in db.list_incidents()] == before

        # Problems are reported in plain words
        try:
            Database(Path(tmp) / "missing.db", readonly=True)
            raise AssertionError("missing file accepted")
        except DatabaseError as exc:
            assert "pipeline.py" in str(exc)
        junk = Path(tmp) / "junk.db"
        junk.write_text("this is not a database", encoding="utf-8")
        try:
            Database(junk)
            raise AssertionError("junk file accepted")
        except DatabaseError:
            pass
        old = Path(tmp) / "old.db"
        with Database(old) as d:
            d.conn.execute("UPDATE meta SET value='0' WHERE key='schema_version'")
            d.conn.commit()
        try:
            Database(old)
            raise AssertionError("wrong version accepted")
        except DatabaseError as exc:
            assert "different version" in str(exc)

        # Empty database: dashboard gets empty answers, not errors
        empty = Path(tmp) / "empty.db"
        with Database(empty) as d:
            assert d.list_incidents() == [] and d.headline()["messages"] == 0 and d.get_held() == []
    print("db.py: all checks passed")


def _info(path: str) -> int:
    try:
        with Database(path, readonly=True) as db:
            h = db.headline()
            print(f"Database: {path}")
            print(f"  {h['messages']} messages became {h['incidents']} incidents "
                  f"({h['open']} open, {h['resolved']} resolved)")
            print(f"  merged {h['merged']}, noise removed {h['noise']}, held for review {h['held']}, "
                  f"late/unlinked set aside {h['ignored_resolved'] + h['ignored_unlinked']}, "
                  f"possible duplicates flagged {h['possible_duplicates']}")
            print(f"  rejected inputs: {h['rejected']}")
            print(f"  waves saved: {db.waves() or 'none'}")
            run = db.get_meta("run")
            if run:
                print(f"  last run: {run.get('finished_at', '?')}")
    except DatabaseError as exc:
        print(exc)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The assistant's database (storage only).")
    parser.add_argument("--info", metavar="DB_FILE", help="show what is inside a database file")
    parser.add_argument("--selftest", action="store_true", help="run built-in checks and exit")
    args = parser.parse_args(argv)
    if args.selftest:
        _selftest()
        return 0
    if args.info:
        return _info(args.info)
    parser.error("use --selftest or --info DB_FILE")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())