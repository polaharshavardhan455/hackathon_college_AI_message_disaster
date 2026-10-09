"""
models.py  --  the shared "forms" every other file uses.

This file has NO AI and NO business logic. It only defines:
  1. The allowed values (Source, Urgency, Status, ...)
  2. The data shapes (Message, Extraction, ResolvedLocation, Incident, ...)
  3. Basic safety checks, so bad data is caught early instead of crashing later.

Data flows through the project like this:

  raw file --ingest--> Message --extract--> Extraction --locations--> ResolvedLocation
                                                  \\                     /
                                                   `--cluster + rules--> Incident
Requires Python 3.10+
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any


# =====================================================================
# 1. ALLOWED VALUES (enums)
#    Using fixed values means nobody can write "Rescue In Progress" in
#    one file and "rescue_in_progress" in another.
# =====================================================================

class _LooseEnum(str, Enum):
    """Base class: lets us turn messy text (from an AI or a file) into a clean enum value."""

    @classmethod
    def parse(cls, value: Any, default: "_LooseEnum") -> "_LooseEnum":
        if isinstance(value, cls):
            return value
        if value is None:
            return default
        key = str(value).strip().lower().replace(" ", "_").replace("-", "_")
        for member in cls:
            if member.value == key or member.name.lower() == key:
                return member
        return default  # unknown text -> safe default, never a crash

    @property
    def label(self) -> str:
        """Human-friendly text for the dashboard, e.g. 'Rescue in progress'."""
        return self.value.replace("_", " ").capitalize()


class Source(_LooseEnum):
    CALL = "call"
    WHATSAPP = "whatsapp"
    SMS = "sms"
    SOCIAL = "social"
    UNKNOWN = "unknown"


class Intent(_LooseEnum):
    """What the AI decides a message is."""
    EMERGENCY = "emergency"   # a real report that needs attention
    RUMOR = "rumor"           # forwarded or unverified: hold for review, never auto-confirm
    NOISE = "noise"           # questions, chatter, anything not an emergency


class Urgency(_LooseEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return _URGENCY_ORDER.index(self)

    def raised(self, steps: int = 1) -> "Urgency":
        """Return a higher urgency (stops at CRITICAL). Urgency only goes up, never silently down."""
        return _URGENCY_ORDER[min(self.rank + steps, len(_URGENCY_ORDER) - 1)]


_URGENCY_ORDER = [Urgency.LOW, Urgency.MEDIUM, Urgency.HIGH, Urgency.CRITICAL]


class Status(_LooseEnum):
    ACTIVE = "active"
    RESCUE_IN_PROGRESS = "rescue_in_progress"
    RESOLVED = "resolved"

    def can_move_to(self, new: "Status") -> bool:
        """RESOLVED is final. This one rule stops old incidents from being re-opened by accident."""
        if self is Status.RESOLVED:
            return new is Status.RESOLVED
        return True


class StatusHint(_LooseEnum):
    """What a single message says about progress (the AI's reading, not yet a decision)."""
    NONE = "none"
    RESCUE_STARTED = "rescue_started"
    RESOLVED = "resolved"


class Resource(_LooseEnum):
    RESCUE = "rescue"             # boats, rescue teams
    MEDICAL = "medical"           # ambulance, medicine, doctor
    FOOD_WATER = "food_water"
    SHELTER = "shelter"
    ROAD_CLEARANCE = "road_clearance"
    OTHER = "other"
    UNKNOWN = "unknown"


class EventKind(_LooseEnum):
    """Types of entries on an incident's timeline."""
    CREATED = "created"
    PEOPLE_CHANGED = "people_changed"
    ESCALATED = "escalated"
    STATUS_CHANGED = "status_changed"
    DETAILS_ADDED = "details_added"
    REPORT_ADDED = "report_added"


# =====================================================================
# 2. SMALL HELPERS
# =====================================================================

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _to_jsonable(obj: Any) -> Any:
    """Turn any of our objects into plain dicts/lists/strings so they can be saved as JSON."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


def _parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


# =====================================================================
# 3. DATA SHAPES
# =====================================================================

@dataclass
class Message:
    """One raw report, after ingest.py has converted it to a common format.
    Audio files become Messages too (their transcript goes in `text`)."""

    id: str
    timestamp: datetime
    source: Source
    text: str
    sender: str = "unknown"        # phone number / handle / account. Helps link follow-up messages.
    wave: int | None = None        # which dataset stage it came from (1-4), if known
    origin: str = ""               # file name or row number, for tracing problems

    def __post_init__(self) -> None:
        self.source = Source.parse(self.source, Source.UNKNOWN)
        self.text = (self.text or "").strip()
        if not self.id:
            raise ValueError("Message needs an id")
        if not self.text:
            # ingest.py catches this and records it in the 'rejected' list instead of crashing.
            raise ValueError("Message text is empty")
        if not isinstance(self.timestamp, datetime):
            raise ValueError("Message timestamp must be a datetime")

    def to_dict(self) -> dict:
        return _to_jsonable(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        return cls(
            id=d["id"], timestamp=_parse_time(d["timestamp"]), source=d.get("source"),
            text=d.get("text", ""), sender=d.get("sender", "unknown"),
            wave=d.get("wave"), origin=d.get("origin", ""),
        )


@dataclass
class Extraction:
    """What the AI (extract.py) understood from ONE message. This is a reading, not a decision."""

    intent: Intent = Intent.NOISE
    location_text: str | None = None          # the place words exactly as written
    people_count: int | None = None           # None = not mentioned
    count_is_additional: bool = False         # True for "2 MORE people": add to the running total
    resource: Resource = Resource.UNKNOWN
    resource_detail: str = ""                 # e.g. "boat", "insulin"
    vulnerable: list[str] = field(default_factory=list)   # e.g. ["elderly", "child"]
    worsening: bool = False                   # message says things are getting worse
    says_same_place: bool = False             # "same place", "still here", etc.
    status_hint: StatusHint = StatusHint.NONE
    confidence: float = 0.0                   # 0.0 - 1.0, how sure the AI is
    summary: str = ""                         # one short line for the dashboard

    def __post_init__(self) -> None:
        self.intent = Intent.parse(self.intent, Intent.NOISE)
        self.resource = Resource.parse(self.resource, Resource.UNKNOWN)
        self.status_hint = StatusHint.parse(self.status_hint, StatusHint.NONE)
        self.confidence = _clamp(float(self.confidence or 0.0), 0.0, 1.0)
        if self.people_count is not None:
            try:
                self.people_count = max(0, int(self.people_count))
            except (TypeError, ValueError):
                self.people_count = None
        self.location_text = (self.location_text or "").strip() or None
        self.vulnerable = [str(v).strip().lower() for v in (self.vulnerable or []) if str(v).strip()]

    def to_dict(self) -> dict:
        return _to_jsonable(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Extraction":
        """Build from AI output. Unknown or missing keys are ignored, so a sloppy AI reply won't crash us."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ResolvedLocation:
    """Where locations.py thinks a report is. Always an estimate, never claimed as exact."""

    name: str                         # best-known place, e.g. a landmark name
    landmark_id: str | None = None    # id from landmarks.json (None if only a rough match)
    lat: float | None = None
    lon: float | None = None
    confidence: float = 0.0           # 0.0 - 1.0
    matched_text: str = ""            # the words that led to this match (shown as evidence)
    kind: str = ""                    # "area" (coarse), "locality", "mrts", "hospital", ... (how precise the point is)
    area_id: str | None = None        # the wider area this place belongs to
    qualifier: str = ""               # extra detail from the report, e.g. "behind; west gate"
    alternatives: list[str] = field(default_factory=list)   # other places the text could also mean

    def __post_init__(self) -> None:
        self.confidence = _clamp(float(self.confidence or 0.0), 0.0, 1.0)

    @property
    def has_coordinates(self) -> bool:
        return self.lat is not None and self.lon is not None

    @property
    def confidence_label(self) -> str:
        if self.confidence >= 0.8:
            return "High"
        if self.confidence >= 0.5:
            return "Medium"
        return "Low"

    def to_dict(self) -> dict:
        return _to_jsonable(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ResolvedLocation":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class IncidentEvent:
    """One line on an incident's timeline: what changed, when, and which message caused it."""

    timestamp: datetime
    kind: EventKind
    text: str
    message_id: str | None = None

    def __post_init__(self) -> None:
        self.kind = EventKind.parse(self.kind, EventKind.REPORT_ADDED)

    def to_dict(self) -> dict:
        return _to_jsonable(self)

    @classmethod
    def from_dict(cls, d: dict) -> "IncidentEvent":
        return cls(timestamp=_parse_time(d["timestamp"]), kind=d.get("kind"),
                   text=d.get("text", ""), message_id=d.get("message_id"))


@dataclass
class Incident:
    """ONE evolving record for one real-world emergency. Many messages -> one Incident."""

    id: str
    created_at: datetime
    last_update: datetime
    location: ResolvedLocation | None = None
    location_text: str = ""                   # the original wording, kept as evidence
    people_count: int | None = None
    resource: Resource = Resource.UNKNOWN
    resource_detail: str = ""
    vulnerable: list[str] = field(default_factory=list)
    needs: list[str] = field(default_factory=list)          # every kind of help asked for (Resource values), in order seen
    peak_people_count: int | None = None                    # highest count ever reported (kept for history)
    urgency: Urgency = Urgency.MEDIUM
    status: Status = Status.ACTIVE
    summary: str = ""
    message_ids: list[str] = field(default_factory=list)
    events: list[IncidentEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.resource = Resource.parse(self.resource, Resource.UNKNOWN)
        self.urgency = Urgency.parse(self.urgency, Urgency.MEDIUM)
        self.status = Status.parse(self.status, Status.ACTIVE)

    # ---- small convenience helpers (no decisions are made here) ----

    @property
    def is_open(self) -> bool:
        return self.status is not Status.RESOLVED

    @property
    def title(self) -> str:
        place = self.location.name if self.location else (self.location_text or "Location unknown")
        return place

    def add_event(self, when: datetime, kind: EventKind, text: str, message_id: str | None = None) -> None:
        self.events.append(IncidentEvent(when, kind, text, message_id))

    # ---- saving / loading ----

    def to_dict(self) -> dict:
        return _to_jsonable(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Incident":
        loc = d.get("location")
        return cls(
            id=d["id"],
            created_at=_parse_time(d["created_at"]),
            last_update=_parse_time(d["last_update"]),
            location=ResolvedLocation.from_dict(loc) if loc else None,
            location_text=d.get("location_text", ""),
            people_count=d.get("people_count"),
            resource=d.get("resource"),
            resource_detail=d.get("resource_detail", ""),
            vulnerable=list(d.get("vulnerable", [])),
            needs=list(d.get("needs", [])),
            peak_people_count=d.get("peak_people_count"),
            urgency=d.get("urgency"),
            status=d.get("status"),
            summary=d.get("summary", ""),
            message_ids=list(d.get("message_ids", [])),
            events=[IncidentEvent.from_dict(e) for e in d.get("events", [])],
        )


@dataclass
class RejectedInput:
    """A file or row we could not use (empty, silent, broken). Shown on the dashboard, never hidden."""

    origin: str                       # file name / row number
    reason: str                       # plain-language explanation
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return _to_jsonable(self)


# =====================================================================
# 4. SELF-CHECK  (run:  python models.py)
# =====================================================================

if __name__ == "__main__":
    # 1) Messy values are cleaned, not crashed on
    assert Urgency.parse("  HIGH ", Urgency.LOW) is Urgency.HIGH
    assert Status.parse("Rescue in progress", Status.ACTIVE) is Status.RESCUE_IN_PROGRESS
    assert Resource.parse("banana", Resource.UNKNOWN) is Resource.UNKNOWN

    # 2) Urgency only goes up and stops at the top
    assert Urgency.HIGH.raised() is Urgency.CRITICAL
    assert Urgency.CRITICAL.raised() is Urgency.CRITICAL

    # 3) A resolved incident cannot be re-opened
    assert not Status.RESOLVED.can_move_to(Status.ACTIVE)
    assert Status.ACTIVE.can_move_to(Status.RESCUE_IN_PROGRESS)

    # 4) Empty text is rejected clearly
    try:
        Message(id="m1", timestamp=datetime.now(), source="sms", text="   ")
        raise SystemExit("FAIL: empty message was accepted")
    except ValueError:
        pass

    # 5) A sloppy AI reply is cleaned up
    ex = Extraction.from_dict({
        "intent": "EMERGENCY", "people_count": "-4", "confidence": 7,
        "vulnerable": [" Elderly ", ""], "unexpected_key": 123,
    })
    assert ex.intent is Intent.EMERGENCY and ex.people_count == 0
    assert ex.confidence == 1.0 and ex.vulnerable == ["elderly"]

    # 6) An incident survives save -> load unchanged
    now = datetime(2026, 1, 1, 8, 0)
    inc = Incident(id="INC-1", created_at=now, last_update=now,
                   location=ResolvedLocation(name="Test place", lat=1.0, lon=2.0, confidence=0.9,
                                             kind="area", area_id="a1", qualifier="behind", alternatives=["Other"]),
                   people_count=5, urgency="high", needs=["rescue"], peak_people_count=7)
    inc.add_event(now, EventKind.CREATED, "Incident created", "m1")
    copy = Incident.from_dict(inc.to_dict())
    assert copy.to_dict() == inc.to_dict()
    assert copy.location.confidence_label == "High" and copy.location.alternatives == ["Other"]
    assert copy.needs == ["rescue"] and copy.peak_people_count == 7

    print("models.py: all checks passed")