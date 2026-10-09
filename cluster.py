"""
cluster.py  --  step 4 of the pipeline (the hardest one).

Job: for every message, decide ONE of these:
    created           -> a new incident is opened
    merged            -> it is added to an existing incident (and rules.py applies what changed)
    noise             -> not an emergency, dropped
    held_rumor        -> forwarded / unverified: kept for a human to review, NEVER becomes an incident or changes one
    held_unclear      -> real-looking, but too vague to place anywhere (needs a human)
    ignored_resolved  -> a late duplicate about an incident that is already resolved (resolved is never reopened)
    ignored_unlinked  -> a progress update ("rescue team reached") that cannot be tied to any incident

How it chooses the incident: it adds up EVIDENCE, each piece with a plain-words reason, and merges only when the
total reaches a threshold. When unsure it does NOT merge (a hidden real emergency is worse than a duplicate); it opens a
separate incident and writes "possibly the same as INC-x" on it, so a human can decide.

    Evidence                                   Points
    same specific place                          +0.60      (same area-level place only: +0.40)
    within 400 m / same area as a specific place +0.50 / +0.35
    different place (conflict)                   -0.60
    same sender talked about this incident       +0.40      (+0.15 if the message names no other place,
                                                             +0.10 more if they said "same place")
    message arrived within 15 min / 60 min       +0.10 / +0.05
    similar wording                              +0.15 / +0.08
    unplaced words match ("the station")         +0.20      (+0.15 if it is the only such incident)
    different kind of help than so far           -0.10
    merge at 0.50 or more; "possibly the same" from 0.30 up to 0.50.

Lesson built in: "same place" messages are linked through THE SENDER'S last incident, never "the most recent incident"
(that breaks as soon as two incidents are active).

The decisions about WHAT CHANGES (counts, urgency, status) live in rules.py, not here.

Try it on your real data (after ingest.py and extract.py):
    python cluster.py messages.json extractions.json --out incidents.json
    python cluster.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable

import rules
from locations import TYPE_HINTS, LocationResolver, normalize_tokens, proximity
from models import (EventKind, Extraction, Incident, Intent, Message, ResolvedLocation, Resource, Status,
                    StatusHint, Urgency)

Locate = Callable[[str | None], ResolvedLocation | None]
Similarity = Callable[[str, list[str]], float]

# Words that carry no meaning when comparing wording or place descriptions.
_STOP = {"the", "and", "for", "are", "with", "this", "that", "here", "there", "have", "has", "was", "were", "from",
         "very", "please", "now", "also", "still", "same", "place", "people", "person", "help", "need", "needs",
         "near", "behind", "opposite", "side", "our", "you", "your", "can", "not", "but", "all", "any", "out", "its",
         "just", "been", "will", "they", "them", "some", "more", "water"}
_PLACE_STOP = {"the", "a", "an", "on", "at", "in", "of", "to", "by", "side", "near", "behind", "opposite", "beside",
               "next", "front", "close", "adjacent", "across", "back", "backside", "inside"}


# =====================================================================
# 1. SETTINGS, RESULTS
# =====================================================================

@dataclass
class ClusterConfig:
    merge_threshold: float = 0.50                       # evidence needed to merge
    possible_threshold: float = 0.30                    # evidence needed to write "possibly the same as ..."
    sender_window: timedelta = timedelta(hours=2)       # a sender's "same place" only links for this long
    stale_after: timedelta = timedelta(hours=12)        # an open incident silent this long is not a candidate
    resolved_quiet: timedelta = timedelta(minutes=30)   # reports this soon after resolution are late duplicates
    min_detail_substance: bool = True                   # a report with no place, count, need or vulnerable is held


@dataclass
class Decision:
    """What happened to one message and why (the audit trail shown on the dashboard)."""
    message_id: str
    action: str
    incident_id: str | None = None
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    possible_duplicate_of: str | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return {"message_id": self.message_id, "action": self.action, "incident_id": self.incident_id,
                "score": round(self.score, 2), "reasons": self.reasons,
                "possible_duplicate_of": self.possible_duplicate_of, "note": self.note}


@dataclass
class HeldReport:
    """A report kept for a human to look at. It never changed any incident."""
    message: Message
    extraction: Extraction
    reason: str
    related_incident_id: str | None = None

    def to_dict(self) -> dict:
        return {"message": self.message.to_dict(), "extraction": self.extraction.to_dict(),
                "reason": self.reason, "related_incident_id": self.related_incident_id}


@dataclass
class ClusterStats:
    messages: int = 0
    noise: int = 0
    rumors_held: int = 0
    unclear_held: int = 0
    created: int = 0
    merged: int = 0
    ignored_resolved: int = 0
    ignored_unlinked: int = 0
    possible_duplicates: int = 0

    def headline(self) -> str:
        open_reports = self.created + self.merged
        return (f"{self.messages} messages -> {self.created} incidents ({self.merged} later reports merged in); "
                f"{self.noise} noise removed, {self.rumors_held + self.unclear_held} held for review, "
                f"{self.ignored_resolved + self.ignored_unlinked} late or unlinked updates set aside "
                f"[{open_reports} messages became part of an incident]")


@dataclass
class ClusterResult:
    incidents: list[Incident]
    decisions: list[Decision]
    held: list[HeldReport]
    stats: ClusterStats


@dataclass
class _Scored:
    score: float
    incident: Incident
    reasons: list[str]
    type_compat: bool = False


# =====================================================================
# 2. WORDING HELPERS
# =====================================================================

def content_tokens(text: str) -> frozenset[str]:
    return frozenset(t for t in normalize_tokens(text) if len(t) >= 3 and t not in _STOP and not t.isdigit())


def place_tokens(text: str | None) -> frozenset[str]:
    return frozenset(t for t in normalize_tokens(text or "") if t not in _PLACE_STOP)


def jaccard(a: frozenset[str] | set[str], b: frozenset[str] | set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def default_similarity(text: str, earlier: list[str]) -> float:
    """Word overlap between a message and an incident's earlier messages (0-1). You can pass a better one
    (for example one based on sentence embeddings) to Clusterer(similarity=...)."""
    tokens = content_tokens(text)
    return max((jaccard(tokens, content_tokens(t)) for t in earlier), default=0.0)


# =====================================================================
# 3. THE CLUSTERER
# =====================================================================

class Clusterer:
    def __init__(self, locate: Locate, config: ClusterConfig | None = None,
                 similarity: Similarity = default_similarity):
        """`locate` turns place words into a ResolvedLocation or None, e.g. LocationResolver(...).resolve"""
        self.locate = locate
        self.config = config or ClusterConfig()
        self.similarity = similarity
        self.incidents: list[Incident] = []
        self.decisions: list[Decision] = []
        self.held: list[HeldReport] = []
        self.stats = ClusterStats()
        self._last_by_sender: dict[str, str] = {}
        self._texts: dict[str, list[str]] = {}          # incident id -> its recent message texts
        self._by_id: dict[str, Incident] = {}

    # ---- public ----

    def run(self, items: Iterable[tuple[Message, Extraction]]) -> ClusterResult:
        """Process (message, extraction) pairs in TIME ORDER (sorted here, so input order does not matter).
        You may call run() again with the next wave; memory carries over."""
        for message, extraction in sorted(items, key=lambda p: (p[0].timestamp, p[0].id)):
            self.process(message, extraction)
        return self.result()

    def result(self) -> ClusterResult:
        return ClusterResult(self.incidents, self.decisions, self.held, self.stats)

    def snapshot(self) -> list[dict]:
        """A JSON-safe copy of every incident right now (db.py stores one per wave)."""
        return [i.to_dict() for i in self.incidents]

    def process(self, message: Message, ex: Extraction) -> Decision:
        cfg, st = self.config, self.stats
        st.messages += 1

        if ex.intent is Intent.NOISE:
            st.noise += 1
            return self._log(Decision(message.id, "noise", note="Not an emergency report"))

        location = self.locate(ex.location_text) if ex.location_text else None
        scored = self._score_candidates(message, ex, location)
        best_open = max((s for s in scored if s.incident.is_open), default=None,
                        key=lambda s: (s.score, s.incident.last_update))
        best_done = max((s for s in scored if not s.incident.is_open), default=None,
                        key=lambda s: (s.score, s.incident.last_update))

        # Rumors never create or change an incident. We only note which incident they seem to be about.
        if ex.intent is Intent.RUMOR:
            related = best_open.incident.id if best_open and best_open.score >= cfg.merge_threshold else None
            st.rumors_held += 1
            self.held.append(HeldReport(message, ex, "Unverified or forwarded report. Not treated as confirmed", related))
            return self._log(Decision(message.id, "held_rumor", related, best_open.score if related else 0.0,
                                      best_open.reasons if related else [],
                                      note="Held for review (unverified)"))

        # A real report that matches an open incident -> merge.
        if best_open and best_open.score >= cfg.merge_threshold:
            inc = best_open.incident
            rules.apply_update(inc, message, ex, location)
            self._remember(inc, message)
            st.merged += 1
            return self._log(Decision(message.id, "merged", inc.id, best_open.score, best_open.reasons))

        # It matches an incident that was JUST resolved -> late duplicate. Never reopen.
        if best_done and best_done.score >= cfg.merge_threshold:
            st.ignored_resolved += 1
            return self._log(Decision(message.id, "ignored_resolved", best_done.incident.id, best_done.score,
                                      best_done.reasons, note="Late report about an incident already resolved"))

        # From here the message belongs to NO existing incident.
        has_place = bool(location or ex.location_text)
        progress_only = ex.status_hint is not StatusHint.NONE and not has_place
        if progress_only:
            st.ignored_unlinked += 1
            return self._log(Decision(message.id, "ignored_unlinked",
                                      note="Progress update that could not be tied to any incident"))

        substance = bool(has_place or ex.people_count is not None or ex.resource is not Resource.UNKNOWN or ex.vulnerable)
        if ex.says_same_place and not has_place:
            st.unclear_held += 1
            reason = "Refers to an earlier place but it is not clear which incident"
            self.held.append(HeldReport(message, ex, reason))
            return self._log(Decision(message.id, "held_unclear", note=reason))
        if cfg.min_detail_substance and not substance:
            st.unclear_held += 1
            reason = "Too little detail to open an incident (no place, number or need)"
            self.held.append(HeldReport(message, ex, reason))
            return self._log(Decision(message.id, "held_unclear", note=reason))

        return self._create(message, ex, location, best_open)

    # ---- creating ----

    def _create(self, message: Message, ex: Extraction, location: ResolvedLocation | None,
                best_open: _Scored | None) -> Decision:
        cfg = self.config
        inc = rules.open_incident(f"INC-{len(self.incidents) + 1:03d}", message, ex, location)
        self.incidents.append(inc)
        self._by_id[inc.id] = inc
        self._remember(inc, message)
        self.stats.created += 1
        when = message.timestamp

        if location is None:
            inc.add_event(when, EventKind.DETAILS_ADDED,
                          f'Location could not be placed on the map. Reported as: "{ex.location_text}"'
                          if ex.location_text else "Location not known. Needs follow-up", message.id)

        possible = None
        if best_open and cfg.possible_threshold <= best_open.score < cfg.merge_threshold:
            possible = best_open.incident.id
            self.stats.possible_duplicates += 1
            inc.add_event(when, EventKind.DETAILS_ADDED,
                          f"Possibly the same as {possible} ({'; '.join(best_open.reasons)}). A responder should check",
                          message.id)
        if location is not None:                       # an earlier incident here that was resolved: say so, do not reopen
            for old in self.incidents[:-1]:
                if not old.is_open and proximity(location, old.location) >= 0.9:
                    inc.add_event(when, EventKind.DETAILS_ADDED,
                                  f"An earlier incident here ({old.id}) was resolved at {old.last_update:%H:%M}. "
                                  "This is a new report", message.id)
                    break
        return self._log(Decision(message.id, "created", inc.id,
                                  best_open.score if best_open else 0.0,
                                  best_open.reasons if best_open else [], possible_duplicate_of=possible))

    # ---- the evidence ----

    def _score_candidates(self, message: Message, ex: Extraction, location: ResolvedLocation | None) -> list[_Scored]:
        cfg = self.config
        out: list[_Scored] = []
        for inc in self.incidents:
            dt = abs(message.timestamp - inc.last_update)
            if inc.is_open and dt > cfg.stale_after:
                continue                               # silent for too long: treat as over
            if not inc.is_open and dt > cfg.resolved_quiet:
                continue                               # resolved long ago: not a late duplicate
            out.append(self._score(inc, message, ex, location, dt))
        # "behind the station": if exactly one nearby incident is at a station, that is a hint
        compat = [s for s in out if s.type_compat and s.incident.is_open
                  and abs(message.timestamp - s.incident.last_update) <= timedelta(minutes=60)]
        if len(compat) == 1:
            compat[0].score += 0.15
            compat[0].reasons.append("the only incident at that kind of place")
        return out

    def _score(self, inc: Incident, message: Message, ex: Extraction, location: ResolvedLocation | None,
               dt: timedelta) -> _Scored:
        score, why, compat = 0.0, [], False

        # 1. Where
        if location is not None and inc.location is not None:
            p = proximity(location, inc.location)
            coarse_same = location.landmark_id == inc.location.landmark_id and location.kind in ("area", "locality")
            if coarse_same:
                score += 0.40; why.append(f"same area ({inc.location.name})")
            elif p >= 0.9:
                score += 0.60; why.append(f"same place ({inc.location.name})")
            elif p >= 0.75:
                score += 0.50; why.append("within 400 m")
            elif p >= 0.5:
                score += 0.35; why.append("same area or within 800 m")
            elif p >= 0.25:
                score += 0.10; why.append("within 1.5 km")
            else:
                score -= 0.60; why.append(f"different place (not {inc.location.name})")
        elif ex.location_text and inc.location_text and (location is None or inc.location is None):
            sim = jaccard(place_tokens(ex.location_text), place_tokens(inc.location_text))
            if sim >= 0.6:
                score += 0.40; why.append("same place words")
            elif sim >= 0.3:
                score += 0.20; why.append("similar place words")
        if location is None and ex.location_text and inc.location is not None:
            hinted = {t for tok in normalize_tokens(ex.location_text) for t in TYPE_HINTS.get(tok, ())}
            if inc.location.kind in hinted:
                score += 0.20; compat = True
                why.append(f"mentions a {inc.location.kind.replace('_', ' ')}, and the incident is at one")

        # 2. Who (only for a known sender)
        if message.sender != "unknown" and self._last_by_sender.get(message.sender) == inc.id \
                and dt <= self.config.sender_window:
            score += 0.40; why.append("same sender")
            if location is None and not ex.location_text:
                score += 0.15; why.append("names no other place")       # nothing contradicts the link
            if ex.says_same_place:
                score += 0.10; why.append('said "same place"')

        # 3. When
        if dt <= timedelta(minutes=15):
            score += 0.10; why.append("within 15 minutes")
        elif dt <= timedelta(minutes=60):
            score += 0.05; why.append("within an hour")

        # 4. Wording
        sim = self.similarity(message.text, self._texts.get(inc.id, []))
        if sim >= 0.5:
            score += 0.15; why.append("very similar wording")
        elif sim >= 0.3:
            score += 0.08; why.append("similar wording")

        # 5. Kind of help
        if ex.resource is not Resource.UNKNOWN and inc.needs and ex.resource.value not in inc.needs:
            score -= 0.10; why.append("asks for a different kind of help")

        return _Scored(score, inc, why, compat)

    # ---- small things ----

    def _remember(self, inc: Incident, message: Message) -> None:
        if message.sender != "unknown":
            self._last_by_sender[message.sender] = inc.id
        texts = self._texts.setdefault(inc.id, [])
        texts.append(message.text)
        del texts[:-6]                                 # only the last few messages are compared

    def _log(self, decision: Decision) -> Decision:
        self.decisions.append(decision)
        return decision


# =====================================================================
# 4. COMMAND LINE + SELF-TEST
# =====================================================================

def _selftest() -> None:
    from locations import LocationResolver

    mini = {"landmarks": [
        {"id": "rt", "name": "Rivertown", "type": "area", "lat": 13.00, "lon": 80.20, "aliases": ["rivertown"]},
        {"id": "rt_stn", "name": "Rivertown Station", "type": "mrts", "lat": 13.0010, "lon": 80.2010, "area": "rt",
         "aliases": ["rivertown mrts"], "context": ["eastside"]},
        {"id": "ht", "name": "Hilltop", "type": "area", "lat": 13.10, "lon": 80.30, "aliases": ["hilltop"]},
        {"id": "city_hosp", "name": "City Hospital", "type": "hospital", "lat": 13.0100, "lon": 80.2100,
         "area": "rt", "aliases": ["city hospital"]},
        {"id": "bridge_a", "name": "Old Bridge", "type": "bridge", "lat": 13.0500, "lon": 80.2500,
         "aliases": ["old bridge"]},
    ]}
    t0 = datetime(2026, 1, 1, 8, 0)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mini.json"
        path.write_text(json.dumps(mini), encoding="utf-8")
        resolver = LocationResolver(path)

        def msg(minutes, sender, text="report", id=None):
            return Message(id=id or f"m{minutes}-{sender}", timestamp=t0 + timedelta(minutes=minutes),
                           source="sms", text=text, sender=sender)

        def ex(**kw):
            base = dict(intent="emergency", confidence=0.9)
            base.update(kw)
            return Extraction(**base)

        def fresh(**kw) -> Clusterer:
            return Clusterer(resolver.resolve, **kw)

        # ---- 1. The full story: one incident from six messages, several senders ----
        c = fresh()
        story = [
            (msg(0, "S1", "6 people trapped at Rivertown MRTS"), ex(location_text="Rivertown MRTS", people_count=6, resource="rescue", resource_detail="boat")),
            (msg(16, "S1", "elderly person is inside"), ex(vulnerable=["elderly"])),                     # no place: linked by sender
            (msg(25, "S2", "10 people now at Rivertown station"), ex(location_text="Rivertown MRTS", people_count=10)),
            (msg(40, "S1", "water rising"), ex(worsening=True, says_same_place=True)),
            (msg(65, "S3", "rescue team reached Rivertown MRTS"), ex(location_text="Rivertown MRTS", status_hint="rescue_started")),
            (msg(90, "S1", "everyone rescued"), ex(status_hint="resolved", says_same_place=True)),
        ]
        r = c.run(story)
        assert len(r.incidents) == 1 and [d.action for d in r.decisions] == ["created"] + ["merged"] * 5
        inc = r.incidents[0]
        assert inc.status is Status.RESOLVED and inc.people_count == 10 and inc.urgency is Urgency.CRITICAL
        assert len(inc.message_ids) == 6 and "same sender" in r.decisions[1].reasons

        # ---- 2. The old bug: "same place" must follow the SENDER, not the most recent incident ----
        c = fresh()
        r = c.run([
            (msg(0, "S1", "trapped at Rivertown MRTS"), ex(location_text="Rivertown MRTS", people_count=5, resource="rescue")),
            (msg(5, "S2", "trapped at Hilltop"), ex(location_text="Hilltop", people_count=8, resource="rescue")),
            (msg(10, "S1", "same place, water rising"), ex(worsening=True, says_same_place=True)),
            (msg(12, "S2", "same place, 2 more"), ex(people_count=2, count_is_additional=True, says_same_place=True)),
        ])
        a, b = r.incidents
        assert a.location.landmark_id == "rt_stn" and b.location.landmark_id == "ht"
        assert a.urgency.rank > Urgency.MEDIUM.rank and b.people_count == 10 and a.people_count == 5

        # ---- 3. Late duplicate after resolution is ignored; a real new emergency later is a NEW incident ----
        c = fresh()
        r = c.run([
            (msg(0, "S1", "trapped Rivertown MRTS"), ex(location_text="Rivertown MRTS", people_count=3, resource="rescue")),
            (msg(30, "S1", "all rescued"), ex(status_hint="resolved", says_same_place=True)),
            (msg(40, "S9", "everyone rescued at Rivertown MRTS"), ex(location_text="Rivertown MRTS", status_hint="resolved")),
            (msg(41, "S8", "rescue team reached Rivertown MRTS"), ex(location_text="Rivertown MRTS", status_hint="rescue_started")),
        ])
        assert len(r.incidents) == 1 and [d.action for d in r.decisions][2:] == ["ignored_resolved"] * 2
        assert r.incidents[0].status is Status.RESOLVED
        before = r.incidents[0].to_dict()
        r = c.run([(msg(150, "S7", "water again at Rivertown MRTS, 4 people"), ex(location_text="Rivertown MRTS", people_count=4, resource="rescue"))])
        assert len(r.incidents) == 2 and r.incidents[0].to_dict() == before          # old one untouched
        assert any("was resolved" in e.text for e in r.incidents[1].events)

        # ---- 4. Do not over-merge ----
        c = fresh()
        r = c.run([
            (msg(0, "S1", "trapped Rivertown MRTS"), ex(location_text="Rivertown MRTS", people_count=3, resource="rescue")),
            (msg(2, "S1", "also trapped at Hilltop"), ex(location_text="Hilltop", people_count=9, resource="rescue")),   # same sender, other place
            (msg(3, "S2", "city hospital flooded"), ex(location_text="City Hospital", people_count=4, resource="rescue")),  # nearby but another place
        ])
        assert len(r.incidents) == 3
        # an area-level report from a stranger is NOT merged with a specific place, but is flagged for a human
        c = fresh()
        r = c.run([
            (msg(0, "S1", "trapped Rivertown MRTS"), ex(location_text="Rivertown MRTS", people_count=3, resource="rescue")),
            (msg(4, "S2", "flooding in Rivertown"), ex(location_text="Rivertown", people_count=10, resource="rescue")),
        ])
        assert len(r.incidents) == 2 and r.decisions[1].possible_duplicate_of == "INC-001"
        assert any("Possibly the same as INC-001" in e.text for e in r.incidents[1].events)
        # ...but the same area-level words from the same sender do merge
        c = fresh()
        r = c.run([
            (msg(0, "S1", "trapped Rivertown MRTS"), ex(location_text="Rivertown MRTS", people_count=3, resource="rescue")),
            (msg(4, "S1", "Rivertown people increasing"), ex(location_text="Rivertown", people_count=10)),
        ])
        assert len(r.incidents) == 1

        # ---- 5. Rumor, noise, vague, unlinked ----
        c = fresh()
        r = c.run([
            (msg(0, "S1", "trapped Rivertown MRTS"), ex(location_text="Rivertown MRTS", people_count=3, resource="rescue")),
            (msg(1, "S5", "bus running?"), ex(intent="noise")),
            (msg(2, "S6", "forwarded: dam will burst at Rivertown MRTS"), ex(intent="rumor", location_text="Rivertown MRTS", people_count=999)),
            (msg(3, "S7", "help"), ex()),
            (msg(4, "S8", "same place still here"), ex(says_same_place=True, vulnerable=["child"])),
            (msg(5, "S9", "everyone is safe now"), ex(status_hint="resolved")),
        ])
        acts = [d.action for d in r.decisions]
        assert acts == ["created", "noise", "held_rumor", "held_unclear", "held_unclear", "ignored_unlinked"]
        assert r.incidents[0].people_count == 3 and r.decisions[2].incident_id == "INC-001"   # rumor changed nothing
        assert len(r.held) == 3 and r.stats.noise == 1
        s = r.stats
        assert s.messages == s.noise + s.rumors_held + s.unclear_held + s.created + s.merged + s.ignored_resolved + s.ignored_unlinked

        # ---- 6. Place words we cannot map: same words merge, different words do not; stale incidents expire ----
        c = fresh()
        r = c.run([
            (msg(0, "S1", "stuck behind the pump house"), ex(location_text="behind the pump house", people_count=4, resource="rescue")),
            (msg(5, "S2", "near the pump house 6 people"), ex(location_text="near the pump house", people_count=6)),
            (msg(8, "S3", "at the school gate"), ex(location_text="the school gate", people_count=2, resource="rescue")),
        ])
        assert len(r.incidents) == 2 and r.incidents[0].people_count == 6 and r.incidents[0].location is None
        assert any("could not be placed" in e.text for e in r.incidents[0].events)
        c = fresh()
        r = c.run([
            (msg(0, "S1", "trapped Rivertown MRTS"), ex(location_text="Rivertown MRTS", people_count=3, resource="rescue")),
            (msg(60 * 13, "S2", "trapped Rivertown MRTS"), ex(location_text="Rivertown MRTS", people_count=3, resource="rescue")),
        ])
        assert len(r.incidents) == 2                                                  # 13 h of silence: a new incident

        # ---- 7. Unplaced words ("behind the station"): a stranger is only FLAGGED, the same sender merges ----
        c = fresh()
        r = c.run([
            (msg(0, "S1", "trapped Rivertown MRTS"), ex(location_text="Rivertown MRTS", people_count=3, resource="rescue")),
            (msg(5, "S2", "opposite the shop near the station backside"), ex(location_text="opposite the shop near the station backside road", people_count=2)),
        ])
        assert len(r.incidents) == 2 and r.decisions[1].possible_duplicate_of == "INC-001"
        c = fresh()
        r = c.run([
            (msg(0, "S1", "trapped Rivertown MRTS"), ex(location_text="Rivertown MRTS", people_count=3, resource="rescue")),
            (msg(5, "S1", "opposite the shop near the station backside"), ex(location_text="opposite the shop near the station backside road", people_count=2)),
        ])
        assert len(r.incidents) == 1 and r.incidents[0].people_count == 3              # 2 < 3 while active: kept higher

        # ---- 8. Order independence, waves, snapshots ----
        pairs = [(msg(0, "S1", "trapped Rivertown MRTS"), ex(location_text="Rivertown MRTS", people_count=6, resource="rescue")),
                 (msg(10, "S2", "10 people Rivertown MRTS"), ex(location_text="Rivertown MRTS", people_count=10)),
                 (msg(20, "S3", "trapped at Hilltop"), ex(location_text="Hilltop", people_count=4, resource="rescue"))]
        one = fresh().run(pairs)
        two = fresh().run(list(reversed(pairs)))
        assert [i.to_dict() for i in one.incidents] == [i.to_dict() for i in two.incidents]
        c = fresh()
        c.run(pairs[:2])
        snap1 = c.snapshot()
        c.run(pairs[2:])
        snap2 = c.snapshot()
        assert len(snap1) == 1 and len(snap2) == 2 and snap1[0]["people_count"] == 10
        assert Incident.from_dict(snap2[1]).title == "Hilltop"
        assert "3 messages" in c.stats.headline()

    # ---- 9. The brief's three descriptions, with the real landmark list ----
    if (Path(__file__).with_name("landmarks.json")).is_file():
        real = LocationResolver(Path(__file__).with_name("landmarks.json"))
        c = Clusterer(real.resolve)
        t = datetime(2026, 1, 1, 8, 0)
        mk = lambda mins, s, i: Message(id=i, timestamp=t + timedelta(minutes=mins), source="sms", text=i, sender=s)
        r = c.run([
            (mk(0, "S1", "A"), ex(location_text="Near the Velachery MRTS west gate.", people_count=6, resource="rescue")),
            (mk(4, "S2", "B"), ex(location_text="Behind the MRTS station, on the Vijayanagar side.", people_count=8)),
            (mk(9, "S3", "C"), ex(location_text="Opposite the medical shop near the station backside road.", people_count=7)),
        ])
        assert [d.action for d in r.decisions] == ["created", "merged", "created"]    # A and B are one incident
        assert r.decisions[2].possible_duplicate_of == "INC-001"                       # C: flagged, not guessed
        print("real landmark check: A and B merged, C flagged as a possible duplicate")
    print("cluster.py: all checks passed")


def _load_pairs(messages_path: str, extractions_path: str) -> tuple[list[tuple[Message, Extraction]], int]:
    msgs = json.loads(Path(messages_path).read_text(encoding="utf-8"))["messages"]
    exts = json.loads(Path(extractions_path).read_text(encoding="utf-8"))["extractions"]
    pairs, missing = [], 0
    for d in msgs:
        e = exts.get(d["id"])
        if e is None:
            missing += 1
            continue
        pairs.append((Message.from_dict(d), Extraction.from_dict(e)))
    return pairs, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Group extracted messages into incidents.")
    parser.add_argument("messages_json", nargs="?", help="messages.json from ingest.py")
    parser.add_argument("extractions_json", nargs="?", help="extractions.json from extract.py")
    parser.add_argument("--landmarks", default=str(Path(__file__).with_name("landmarks.json")))
    parser.add_argument("--out", default="incidents.json", help="where to save the incidents")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        _selftest()
        return 0
    if not (args.messages_json and args.extractions_json):
        parser.error("give messages.json and extractions.json, or use --selftest")

    pairs, missing = _load_pairs(args.messages_json, args.extractions_json)
    resolver = LocationResolver(args.landmarks)
    result = Clusterer(resolver.resolve).run(pairs)

    order = sorted(result.incidents, key=lambda i: (not i.is_open, -i.urgency.rank, i.last_update))
    print(result.stats.headline())
    if missing:
        print(f"({missing} messages had no extraction and were skipped)")
    print(f"\n{'ID':8} {'Status':19} {'Urgency':9} {'People':7} {'Reports':8} Place")
    for i in order[:40]:
        people = "?" if i.people_count is None else i.people_count
        print(f"{i.id:8} {i.status.label:19} {i.urgency.label:9} {str(people):7} {len(i.message_ids):<8} {i.title}")
    if len(order) > 40:
        print(f"... and {len(order) - 40} more")
    counts = Counter(d.action for d in result.decisions)
    print("\nDecisions:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    Path(args.out).write_text(json.dumps({
        "incidents": [i.to_dict() for i in result.incidents],
        "decisions": [d.to_dict() for d in result.decisions],
        "held": [h.to_dict() for h in result.held],
        "stats": result.stats.__dict__}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())