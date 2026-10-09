"""pipeline.py - the conductor. It runs every step in order, wave by wave.

WHAT IT DOES
    data folder -> ingest -> extract (AI reads each message) -> locations -> cluster (+rules) -> database
    After EACH wave it saves the database, so the dashboard can already show progress,
    and it keeps a snapshot of the incidents after every wave (for the replay).

WHY IT EXISTS
    The other files each do one job. This one connects them, prints clear progress,
    and prints the headline numbers ("N messages became M incidents").

WHAT GOES IN / OUT
    In : a folder or file with the dataset (or a messages.json you made earlier with ingest.py)
    Out: emergency.db  (the dashboard reads only this file)

DOES IT USE AI?
    Only through extract.py (free Gemini / Groq / Ollama). Everything else is plain code.
    Without a key, it still runs using the basic keyword reader (weaker results, and it says so).

SAFE TO RE-RUN
    Run it again any time. The database is rebuilt from scratch, but AI answers are cached on
    disk, so a re-run is fast and a stopped run continues where it stopped.

RUN
    python pipeline.py data/                       the normal way
    python pipeline.py data/ --limit 100           quick trial with the first 100 messages
    python pipeline.py --messages messages.json    skip reading files (use an earlier ingest result)
    python pipeline.py --selftest                  check that it works
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol

from cluster import Clusterer
from db import DEFAULT_DB, Database, DatabaseError
from models import Extraction, Message, RejectedInput

log = logging.getLogger("pipeline")
DEFAULT_LANDMARKS = str(Path(__file__).with_name("landmarks.json"))
DEFAULT_BATCH = 20      # messages sent to the AI at a time (so Ctrl+C reacts quickly and progress is saved)


class PipelineError(Exception):
    """A problem we can explain in plain words."""


class _Extractor(Protocol):
    def extract_many(self, messages: list[Message], progress_every: int = ...) -> list[Extraction]: ...


# =====================================================================
# 1. WAVES
# =====================================================================

def assign_waves(messages: list[Message]) -> list[int]:
    """Make sure every message has a wave number. Returns the list of waves, in order.

    - Messages already tagged by ingest.py (from folder/file names or a column) keep their wave.
    - A message with no wave takes the wave of the message just before it in time.
    - If the dataset has no waves at all, everything is wave 1."""
    ordered = sorted(messages, key=lambda m: (m.timestamp, m.id))
    known = sorted({m.wave for m in ordered if m.wave is not None})
    if not known:
        for m in ordered:
            m.wave = 1
        return [1]
    current = known[0]
    for m in ordered:
        if m.wave is None:
            m.wave = current
        else:
            current = m.wave
    return sorted({m.wave for m in messages})


def wave_overlap_warnings(messages: list[Message], waves: list[int]) -> list[str]:
    """Waves should be in time order. If they overlap we say so (results are still correct per wave)."""
    spans = {}
    for w in waves:
        times = [m.timestamp for m in messages if m.wave == w]
        spans[w] = (min(times), max(times))
    out = []
    for a, b in zip(waves, waves[1:]):
        if spans[b][0] < spans[a][1]:
            out.append(f"Wave {b} starts ({spans[b][0]:%d %b %H:%M}) before wave {a} ends "
                       f"({spans[a][1]:%d %b %H:%M}). Waves are processed in number order.")
    return out


# =====================================================================
# 2. LOADING AN EARLIER ingest.py RESULT
# =====================================================================

def load_messages_json(path: str | Path) -> tuple[list[Message], list[RejectedInput], list[str]]:
    file = Path(path)
    if not file.is_file():
        raise PipelineError(f"File not found: {file}")
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PipelineError(f"{file.name} is not valid JSON (line {exc.lineno}): {exc.msg}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        raise PipelineError(f"{file.name} should be the file made by 'python ingest.py' (it has no 'messages' list).")
    messages, rejected = [], []
    warnings = [str(w) for w in data.get("warnings", [])]
    for n, d in enumerate(data["messages"], start=1):
        try:
            messages.append(Message.from_dict(d))
        except (KeyError, ValueError, TypeError) as exc:
            rejected.append(RejectedInput(origin=f"{file.name} message {n}", reason=f"Unreadable saved message ({exc})"))
    for n, d in enumerate(data.get("rejected", []), start=1):
        try:
            when = d.get("timestamp")
            rejected.append(RejectedInput(origin=str(d.get("origin", "")), reason=str(d.get("reason", "")),
                                          timestamp=datetime.fromisoformat(when) if when else datetime.now()))
        except (AttributeError, ValueError, TypeError):
            warnings.append(f"Saved rejected entry {n} could not be read and was skipped.")
    return messages, rejected, warnings


# =====================================================================
# 3. THE MAIN RUN
# =====================================================================

def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def run_pipeline(messages: list[Message], rejected: list[RejectedInput], warnings: list[str],
                 extractor: _Extractor, resolver: Any, db: Database,
                 batch_size: int = DEFAULT_BATCH, say: Callable[[str], None] = print,
                 info: dict | None = None) -> dict:
    """Run extract -> locations -> cluster -> save, wave by wave. Returns a summary dict.

    resolver : anything with .resolve(text) and (optionally) .unmatched, e.g. LocationResolver
    info     : extra facts to store with the run (provider, source...)"""
    started = time.time()
    messages = sorted(messages, key=lambda m: (m.timestamp, m.id))
    waves = assign_waves(messages) if messages else []
    warnings = list(warnings) + (wave_overlap_warnings(messages, waves) if waves else [])

    # Start from an empty database; the first thing saved is the input itself.
    db.clear_all()
    run_info = dict(info or {}, started_at=datetime.now().isoformat(timespec="seconds"), complete=False,
                    waves=waves, messages_in=len(messages), warnings=warnings[:200])
    db.set_meta("run", run_info)
    db.save_rejected(rejected)
    db.save_messages(messages)

    clusterer = Clusterer(resolver.resolve)
    snapshots: dict[int, list[dict]] = {}
    previous = 0
    for wave in waves:
        wave_messages = [m for m in messages if m.wave == wave]
        say(f"\nWave {wave}: {len(wave_messages)} messages")
        extractions: dict[str, Extraction] = {}
        for batch in _batches(wave_messages, max(1, batch_size)):
            results = extractor.extract_many(batch, progress_every=0)
            if len(results) != len(batch):
                raise PipelineError("The extractor returned a different number of results than messages sent.")
            extractions.update({m.id: e for m, e in zip(batch, results)})
            db.save_extractions({m.id: e for m, e in zip(batch, results)})
            say(f"  read {len(extractions)} / {len(wave_messages)} messages")

        clusterer.run([(m, extractions[m.id]) for m in wave_messages])
        snapshots[wave] = clusterer.snapshot()
        result = clusterer.result()
        db.save_results(result.incidents, result.decisions, result.held, snapshots, result.stats.__dict__)
        now = len(result.incidents)
        say(f"  -> {now - previous} new incident(s); {now} in total so far")
        previous = now

    summary = _finish(db, resolver, extractor, messages, rejected, warnings, run_info, started)
    return summary


def _finish(db: Database, resolver: Any, extractor: Any, messages: list[Message],
            rejected: list[RejectedInput], warnings: list[str], run_info: dict, started: float) -> dict:
    headline = db.headline()
    unmatched = getattr(resolver, "unmatched", None)
    unmatched_top = unmatched.most_common(10) if unmatched else []
    unplaced = sum(1 for i in db.list_incidents() if i.location is None)
    stats = getattr(extractor, "stats", None)
    extract_stats = dict(vars(stats)) if stats is not None and hasattr(stats, "__dict__") else {}
    run_info.update(complete=True, finished_at=datetime.now().isoformat(timespec="seconds"),
                    seconds=round(time.time() - started, 1), headline=headline,
                    extract_stats=extract_stats, unplaced_incidents=unplaced,
                    unmatched_places=unmatched_top)
    db.set_meta("run", run_info)
    return {"headline": headline, "warnings": warnings, "rejected": len(rejected),
            "unmatched_places": unmatched_top, "unplaced_incidents": unplaced,
            "extract_stats": extract_stats, "seconds": run_info["seconds"]}


def print_report(summary: dict, db_path: str | Path, say: Callable[[str], None] = print) -> None:
    h = summary["headline"]
    say("\n" + "=" * 64)
    say(f"  {h['messages']} messages became {h['incidents']} incidents")
    say("=" * 64)
    say(f"  Incidents: {h['active']} active, {h['rescue_in_progress']} rescue in progress, {h['resolved']} resolved")
    say(f"  Merged into existing incidents : {h['merged']}")
    say(f"  Noise removed                  : {h['noise']}")
    say(f"  Held for human review          : {h['held']} ({h['rumors_held']} rumors, {h['unclear_held']} unclear)")
    say(f"  Late / unlinked updates ignored: {h['ignored_resolved'] + h['ignored_unlinked']}")
    say(f"  Flagged 'possibly the same'    : {h['possible_duplicates']}")
    say(f"  Rejected inputs (bad files)    : {summary['rejected']}")
    stats = summary.get("extract_stats") or {}
    if stats:
        say(f"  AI reading: {stats.get('api_calls', 0)} AI calls, {stats.get('cache_hits', 0)} from cache, "
            f"{stats.get('fallbacks', 0)} with the basic keyword reader")
        if stats.get("fallbacks"):
            say("  NOTE: some messages used the basic keyword reader (weaker). Fix the AI problem shown above and "
                "run the same command again; finished messages are cached.")
    if summary["unplaced_incidents"]:
        say(f"  {summary['unplaced_incidents']} incident(s) could not be placed on the map (place not in landmarks.json).")
    if summary["unmatched_places"]:
        words = ", ".join(f"'{w}' x{n}" for w, n in summary["unmatched_places"][:8])
        say(f"  Place words not found in landmarks.json (add them to improve the map): {words}")
        say("  Tip: python locations.py --suggest extractions.json   (extractions.json from: python extract.py messages.json)")
    if summary["warnings"]:
        say(f"  {len(summary['warnings'])} warning(s) while reading files. First ones:")
        for w in summary["warnings"][:5]:
            say(f"    - {w}")
    say(f"\n  Finished in {summary['seconds']} seconds. Database: {db_path}")
    say("  Next: streamlit run app.py")


# =====================================================================
# 4. COMMAND LINE
# =====================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the whole assistant: files -> incidents -> database.")
    parser.add_argument("path", nargs="?", help="folder or file with the dataset (audio + messages)")
    parser.add_argument("--messages", help="use a messages.json made earlier by ingest.py instead of reading files")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"database file to create (default {DEFAULT_DB})")
    parser.add_argument("--landmarks", default=DEFAULT_LANDMARKS, help="landmarks.json (known places)")
    parser.add_argument("--whisper-size", default="small",
                        help="speech model: tiny, base, small, medium, large-v3, or a folder with downloaded model files")
    parser.add_argument("--provider", help="gemini, groq, ollama or custom (overrides .env)")
    parser.add_argument("--model", help="AI model name (overrides .env)")
    parser.add_argument("--no-api", action="store_true", help="do not use any AI; use the basic keyword reader only")
    parser.add_argument("--limit", type=int, help="only use the first N messages (quick trial)")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="messages per AI batch (default %(default)s)")
    parser.add_argument("--selftest", action="store_true", help="run built-in checks and exit")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("extract").setLevel(logging.WARNING)      # the pipeline prints its own progress
    if args.selftest:
        _selftest()
        return 0
    if bool(args.path) == bool(args.messages):
        parser.error("give EITHER a data folder/file OR --messages messages.json (or use --selftest)")

    from extract import Extractor, ExtractorError, _load_dotenv
    from ingest import Ingestor, WhisperTranscriber
    from locations import LocationError, LocationResolver

    try:
        # 1. read the input
        if args.messages:
            messages, rejected, warnings = load_messages_json(args.messages)
            print(f"Loaded {len(messages)} messages from {args.messages} ({len(rejected)} rejected earlier)")
        else:
            print(f"Reading files from {args.path} ...")
            result = Ingestor(transcriber=WhisperTranscriber(model_size=args.whisper_size)).run(args.path)
            messages, rejected, warnings = result.messages, result.rejected, result.warnings
            print(result.summary())
        if args.limit:
            messages = sorted(messages, key=lambda m: (m.timestamp, m.id))[:max(0, args.limit)]
            print(f"Trial mode: using only the first {len(messages)} messages")

        # 2. set up the helpers
        _load_dotenv()
        if args.provider:
            os.environ["LLM_PROVIDER"] = args.provider
        if args.model:
            os.environ["LLM_MODEL"] = args.model
        extractor = Extractor(use_api=not args.no_api)
        resolver = LocationResolver(args.landmarks)
        info = {"source": args.messages or args.path, "provider": extractor.config.provider,
                "model": extractor.config.model, "ai_enabled": not args.no_api}

        # 3. run
        with Database(args.db) as db:
            if not messages:
                db.clear_all()
                db.save_rejected(rejected)
                db.set_meta("run", dict(info, complete=True, messages_in=0, waves=[], warnings=warnings[:200]))
                print("\nNo usable messages were found, so there are no incidents.")
                print(f"The {len(rejected)} rejected input(s) were saved so the dashboard can show why.")
                for r in rejected[:10]:
                    print(f"  - {r.origin}: {r.reason}")
                return 1
            summary = run_pipeline(messages, rejected, warnings, extractor, resolver, db,
                                   batch_size=args.batch, info=info)
        print_report(summary, args.db)
        return 0
    except KeyboardInterrupt:
        print("\n\nStopped by you. Nothing is lost: finished AI answers are cached and the database holds every "
              "wave that was completed.\nRun the same command again to continue.")
        return 130
    except (PipelineError, ExtractorError, LocationError, DatabaseError) as exc:
        print(f"\nProblem: {exc}")
        return 1


# =====================================================================
# 5. SELF-TEST  (run:  python pipeline.py --selftest)
# =====================================================================

def _selftest() -> None:
    import csv
    import tempfile
    from extract import Extractor, LLMConfig
    from ingest import Ingestor
    from locations import LocationResolver
    from models import Status

    # ---- waves ----
    def m(i, minute, wave=None):
        return Message(id=i, timestamp=datetime(2026, 3, 1, 8, minute), source="sms", text="x " + i, wave=wave)
    a, b, c = m("a", 0), m("b", 5), m("c", 10)
    assert assign_waves([a, b, c]) == [1] and {a.wave, b.wave, c.wave} == {1}              # no waves at all -> 1
    a, b, c, d = m("a", 0), m("b", 5, 2), m("c", 10), m("d", 15, 3)
    assert assign_waves([c, a, d, b]) == [2, 3] and (a.wave, c.wave) == (2, 2)             # carried forward; first gets the smallest
    assert wave_overlap_warnings([m("p", 0, 1), m("q", 5, 2)], [1, 2]) == []
    assert wave_overlap_warnings([m("p", 30, 1), m("q", 5, 2)], [1, 2])                    # overlap is reported

    # ---- a small invented dataset on disk: two stage folders, one broken file ----
    landmarks = {"meta": {"city": "Test"}, "landmarks": [
        {"id": "alpha", "name": "Alphaville", "type": "area", "lat": 13.00, "lon": 80.20, "area": "alpha",
         "aliases": ["alphaville"], "context": []},
        {"id": "alpha_gate", "name": "Alphaville Gate", "type": "landmark", "lat": 13.001, "lon": 80.201,
         "area": "alpha", "aliases": ["alphaville gate", "alpha gate"], "context": []},
        {"id": "beta", "name": "Betaton", "type": "area", "lat": 13.08, "lon": 80.27, "area": "beta",
         "aliases": ["betaton"], "context": []}]}

    class FakeExtractor:
        """Stands in for the AI: reads a tiny scripted table. Records how it was called."""
        def __init__(self):
            self.calls, self.stats = [], None

        def extract_many(self, msgs, progress_every=0):
            self.calls.append(len(msgs))
            out = []
            for msg in msgs:
                t = msg.text.lower()
                if "weather" in t:
                    out.append(Extraction(intent="noise", confidence=0.9))
                elif "forwarded" in t:
                    out.append(Extraction(intent="rumor", location_text="Betaton", confidence=0.6))
                else:
                    place = "Alphaville Gate" if "alphaville" in t else "Betaton"
                    count = next((int(w) for w in t.split() if w.isdigit()), None)
                    hint = "resolved" if "rescued" in t else "none"
                    out.append(Extraction(intent="emergency", location_text=place, people_count=count,
                                          resource="rescue", status_hint=hint, confidence=0.9,
                                          worsening="rising" in t))
            return out

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "data" / "stage_1").mkdir(parents=True)
        (root / "data" / "stage_2").mkdir(parents=True)
        lm_path = root / "landmarks.json"
        lm_path.write_text(json.dumps(landmarks), encoding="utf-8")

        def write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "sender", "text"])
                w.writerows([(t, f"s{i}", text) for i, (t, text) in enumerate(rows)])

        write_csv(root / "data" / "stage_1" / "sms.csv", [
            ("2026-03-01 08:00", "4 people trapped at Alphaville"),
            ("2026-03-01 08:10", "10 people now at Alphaville, water rising"),
            ("2026-03-01 08:12", "what is the weather today"),
            ("2026-03-01 08:20", "Forwarded: Betaton is flooded completely!!"),
        ])
        write_csv(root / "data" / "stage_2" / "sms.csv", [
            ("2026-03-01 09:00", "3 people stuck at Betaton"),
            ("2026-03-01 09:30", "Alphaville everyone rescued"),
        ])
        (root / "data" / "stage_2" / "broken.json").write_text("{not json", encoding="utf-8")

        ingested = Ingestor(transcriber=None).run(root / "data")
        assert len(ingested.messages) == 6 and ingested.rejected, ingested.summary()

        fake = FakeExtractor()
        resolver = LocationResolver(lm_path)
        db_path = root / "out.db"
        lines: list[str] = []
        with Database(db_path) as db:
            summary = run_pipeline(ingested.messages, ingested.rejected, ingested.warnings, fake, resolver, db,
                                   batch_size=2, say=lines.append, info={"source": "test"})
            h = summary["headline"]
            assert h["messages"] == 6 and h["incidents"] == 2, h
            assert h["noise"] == 1 and h["rumors_held"] == 1 and h["merged"] == 2 and h["resolved"] == 1, h
            assert h["messages"] == h["created"] + h["merged"] + h["noise"] + h["held"] + \
                   h["ignored_resolved"] + h["ignored_unlinked"]
            assert db.waves() == [1, 2]
            assert fake.calls == [2, 2, 2]                                   # batches of 2, three batches in total
            # Replay: after wave 1 there is one open incident with the larger count; nothing resolved yet
            w1 = db.list_incidents(as_of_wave=1)
            assert len(w1) == 1 and w1[0].people_count == 10 and w1[0].is_open
            assert db.headline(as_of_wave=1)["messages"] == 4
            finished = [i for i in db.list_incidents() if i.status is Status.RESOLVED]
            still_open = [i for i in db.list_incidents() if i.is_open]
            assert len(finished) == 1 and len(still_open) == 1
            assert finished[0].title == "Alphaville Gate" and finished[0].people_count == 10   # resolved, count kept
            assert still_open[0].title == "Betaton" and still_open[0].people_count == 3
            assert len(db.get_rejected()) == len(ingested.rejected) >= 1
            run = db.get_meta("run")
            assert run["complete"] is True and run["waves"] == [1, 2] and run["source"] == "test"
            print_report(summary, db_path, say=lines.append)
            assert any("6 messages became 2 incidents" in line for line in lines)

            # Safe to re-run: same input again gives exactly the same database content
            before = [i.to_dict() for i in db.list_incidents()]
            run_pipeline(ingested.messages, ingested.rejected, ingested.warnings, FakeExtractor(), LocationResolver(lm_path),
                         db, batch_size=5, say=lambda s: None)
            assert [i.to_dict() for i in db.list_incidents()] == before
            assert db.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 6
            assert db.conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 6

            # No messages at all: runs cleanly and leaves an empty database
            empty = run_pipeline([], ingested.rejected, [], FakeExtractor(), LocationResolver(lm_path), db,
                                 say=lambda s: None)
            assert empty["headline"]["messages"] == 0 and len(db.get_rejected()) == len(ingested.rejected)

        # ---- the REAL extractor with no AI (offline fallback) must also run end to end ----
        real = Extractor(config=LLMConfig(provider="gemini", api_key="", base_url="x", model="x", rpm=10, workers=1),
                         cache_dir=root / "cache", use_api=False)
        with Database(root / "real.db") as db:
            s = run_pipeline(ingested.messages, [], [], real, LocationResolver(lm_path), db, say=lambda s: None)
            assert s["headline"]["messages"] == 6 and db.get_meta("run")["complete"]

        # ---- loading an earlier messages.json, including a broken entry ----
        saved = root / "messages.json"
        saved.write_text(json.dumps({
            "messages": [x.to_dict() for x in ingested.messages] + [{"id": "bad"}],
            "rejected": [{"origin": "z.csv", "reason": "Row has no message text", "timestamp": "2026-03-01T07:00:00"}],
            "warnings": ["a warning"]}), encoding="utf-8")
        msgs, rej, warns = load_messages_json(saved)
        assert len(msgs) == 6 and len(rej) == 2 and warns == ["a warning"]
        for bad, text in (("nope.json", None), ("bad.json", "{oops"), ("list.json", "[1,2]")):
            if text is not None:
                (root / bad).write_text(text, encoding="utf-8")
            try:
                load_messages_json(root / bad)
                raise AssertionError("bad file accepted: " + bad)
            except PipelineError:
                pass

        # ---- command line wiring (--messages path, no AI) ----
        code = main(["--messages", str(saved), "--db", str(root / "cli.db"), "--landmarks", str(lm_path), "--no-api",
                     "--limit", "5"])
        assert code == 0
        with Database(root / "cli.db", readonly=True) as db:
            assert db.headline()["messages"] == 5
        code = main(["--messages", str(root / "bad.json"), "--db", str(root / "cli2.db"), "--no-api"])
        assert code == 1                                                      # explained problem, not a crash
    print("pipeline.py: all checks passed")


if __name__ == "__main__":
    sys.exit(main())