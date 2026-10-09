"""
rules.py  --  the decisions, written as plain code (NO AI).

The AI only READS a message (extract.py). This file decides what that message CHANGES in an incident:
people count, urgency, status, help needed, vulnerable people, location. Every change is written to
the incident's timeline in plain words, so a responder can always see WHAT changed and WHY.

cluster.py decides WHICH incident a message belongs to. This file then applies it:

    incident = open_incident(new_id, message, extraction, location)      # a brand-new incident
    events   = apply_update(incident, message, extraction, location)     # an existing incident

Run   python rules.py --rules      to print every rule in plain language (good for explaining or presenting)
Run   python rules.py --selftest   to run the built-in checks
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta

from locations import COARSE_KINDS
from models import (EventKind, Extraction, Incident, IncidentEvent, Message, Resource,
                    ResolvedLocation, Source, Status, StatusHint, Urgency)

# ---------------------------------------------------------------------
# The numbers behind the rules. Change them here and the whole system follows.
# ---------------------------------------------------------------------
LARGE_GROUP = 10                              # this many people or more adds 1 urgency point
VERY_LARGE_GROUP = 50                         # this many or more adds 1 more point
WORSENING_COOLDOWN = timedelta(minutes=30)    # "water rising" can raise urgency once per 30 minutes, so
                                              # twenty duplicate reports cannot push it to the top alone
MAX_DETAIL_ITEMS = 4                          # keep the "help needed" detail short and readable
SPECIFIC_PLACE_BONUS = 0.3                    # a specific place beats a neighbourhood-level guess
LOCATION_MARGIN = 0.05                        # a new location must be clearly better to replace the old one

# Which help is shown as the "primary" resource when several are needed (first one wins).
RESOURCE_PRIORITY = [Resource.MEDICAL, Resource.RESCUE, Resource.SHELTER, Resource.FOOD_WATER,
                     Resource.ROAD_CLEARANCE, Resource.OTHER]

_SOURCE_WORDS = {Source.CALL: "phone call", Source.WHATSAPP: "WhatsApp message", Source.SMS: "SMS",
                 Source.SOCIAL: "social media post", Source.UNKNOWN: "report"}


# =====================================================================
# 1. URGENCY (from the FACTS known about the incident)
# =====================================================================

def _level_from_points(points: int) -> Urgency:
    if points >= 4:
        return Urgency.CRITICAL
    if points >= 2:
        return Urgency.HIGH
    if points == 1:
        return Urgency.MEDIUM
    return Urgency.LOW


def compute_urgency(inc: Incident, extra_points: int = 0) -> tuple[Urgency, list[str]]:
    """Urgency the known facts justify, plus the reasons in plain words.
         medical help needed        +2        people need rescue or shelter   +1
         vulnerable people          +1        10 or more people               +1
         50 or more people          +1 more   (reported worsening is handled separately)
       0 points = Low, 1 = Medium, 2-3 = High, 4 or more = Critical."""
    points, reasons = 0, []
    needs = {Resource.parse(n, Resource.UNKNOWN) for n in inc.needs}
    if Resource.MEDICAL in needs:
        points += 2
        reasons.append("medical help needed")
    if needs & {Resource.RESCUE, Resource.SHELTER}:
        points += 1
        reasons.append("people need rescue or shelter")
    if inc.vulnerable:
        points += 1
        reasons.append("vulnerable people: " + ", ".join(inc.vulnerable[:3]))
    n = inc.people_count or 0
    if n >= LARGE_GROUP:
        points += 1
        reasons.append(f"{n} people affected")
    if n >= VERY_LARGE_GROUP:
        points += 1
    points += extra_points
    return _level_from_points(points), reasons


def _worsening_allowed(inc: Incident, now: datetime) -> bool:
    """False if a 'reported worsening' escalation already happened in the last 30 minutes."""
    for event in reversed(inc.events):
        if event.kind is EventKind.ESCALATED and "reported worsening" in event.text.lower():
            return now - event.timestamp >= WORSENING_COOLDOWN
    return True


# =====================================================================
# 2. SMALL HELPERS
# =====================================================================

def primary_resource(needs: list[str]) -> Resource:
    have = {Resource.parse(n, Resource.UNKNOWN) for n in needs}
    for resource in RESOURCE_PRIORITY:
        if resource in have:
            return resource
    return Resource.UNKNOWN


def build_summary(inc: Incident) -> str:
    """One calm sentence built from the record itself (so it always matches the data)."""
    if inc.people_count is None:
        people = "Number of people not known"
    elif inc.people_count == 0:
        people = "No people reported hurt or trapped"
    else:
        people = f"{inc.people_count} {'person' if inc.people_count == 1 else 'people'}"
    parts = [people]
    if inc.vulnerable:
        parts.append("including " + " and ".join(inc.vulnerable[:3]))
    text = ", ".join(parts) + "."
    help_text = inc.resource_detail or (inc.resource.label.lower() if inc.resource is not Resource.UNKNOWN else "")
    if help_text:
        text += f" Needs: {help_text}."
    return text


def _add(inc: Incident, when: datetime, kind: EventKind, text: str, message_id: str,
         added: list[IncidentEvent]) -> None:
    # Note-type events are not repeated word for word (thousands of duplicate reports must not flood the timeline).
    if kind is EventKind.DETAILS_ADDED and any(e.kind is kind and e.text == text for e in inc.events):
        return
    event = IncidentEvent(when, kind, text, message_id)
    inc.events.append(event)
    added.append(event)


def _location_quality(loc: ResolvedLocation) -> float:
    return loc.confidence + (0.0 if loc.kind in COARSE_KINDS else SPECIFIC_PLACE_BONUS)


# =====================================================================
# 3. THE TWO ENTRY POINTS
# =====================================================================

def open_incident(incident_id: str, message: Message, ex: Extraction,
                  location: ResolvedLocation | None) -> Incident:
    """Create a new incident from its first report. (cluster.py should not call this for a message that only
    says 'everyone rescued': there is nothing to open. If it does, the resolved hint is ignored.)"""
    inc = Incident(id=incident_id, created_at=message.timestamp, last_update=message.timestamp,
                   urgency=Urgency.LOW, status=Status.ACTIVE)
    _update(inc, message, ex, location, is_new=True)
    return inc


def apply_update(inc: Incident, message: Message, ex: Extraction,
                 location: ResolvedLocation | None) -> list[IncidentEvent]:
    """Apply one more report to an existing incident. Returns the timeline events it caused (empty list if
    the report added nothing new, or if the incident is already resolved: resolved is final)."""
    return _update(inc, message, ex, location, is_new=False)


# =====================================================================
# 4. THE RULES, IN ORDER
# =====================================================================

def _update(inc: Incident, message: Message, ex: Extraction, location: ResolvedLocation | None,
            is_new: bool) -> list[IncidentEvent]:
    if not inc.is_open:
        return []                                   # RESOLVED IS FINAL: nothing can change it
    added: list[IncidentEvent] = []
    when, mid = message.timestamp, message.id

    def add(kind: EventKind, text: str) -> None:
        _add(inc, when, kind, text, mid, added)

    if mid not in inc.message_ids:
        inc.message_ids.append(mid)
    if when > inc.last_update:
        inc.last_update = when
    if is_new:
        add(EventKind.CREATED, f"Incident opened from a {_SOURCE_WORDS.get(message.source, 'report')}")

    hint = ex.status_hint
    # Rule 1: LOCATION. Keep the best estimate. A specific place beats a neighbourhood guess; never erase a known place.
    if location is not None:
        if inc.location is None:
            inc.location = location
            inc.location_text = ex.location_text or ""
            add(EventKind.DETAILS_ADDED,
                f"Location interpreted as {location.name} ({location.confidence_label.lower()} confidence)")
        elif (location.landmark_id != inc.location.landmark_id
              and _location_quality(location) > _location_quality(inc.location) + LOCATION_MARGIN):
            inc.location = location
            inc.location_text = ex.location_text or inc.location_text
            add(EventKind.DETAILS_ADDED,
                f"Location refined to {location.name} ({location.confidence_label.lower()} confidence)")
    if not inc.location_text and ex.location_text:
        inc.location_text = ex.location_text         # keep the original words as evidence

    # Rule 2: PEOPLE COUNT.
    #   "N more"            -> add to the total
    #   a plain number      -> the newest number replaces the old one
    #   a LOWER plain number while still ACTIVE   -> treated as a conflicting report: keep the higher (safer)
    #   a LOWER number once rescue has started    -> accepted: it means people are being rescued, N remain
    #   a count on a "resolved" message           -> ignored, nothing is changing
    n = ex.people_count
    if n is not None and hint is not StatusHint.RESOLVED:
        old = inc.people_count
        rescue_underway = inc.status is Status.RESCUE_IN_PROGRESS or hint is StatusHint.RESCUE_STARTED
        if ex.count_is_additional and old is not None:
            new = old + n
            add(EventKind.PEOPLE_CHANGED, f"{n} more reported: people count changed from {old} to {new}")
            inc.people_count = new
        elif old is None or ex.count_is_additional:
            inc.people_count = n
            add(EventKind.DETAILS_ADDED, f"People affected: {n}")
        elif n > old:
            inc.people_count = n
            add(EventKind.PEOPLE_CHANGED, f"People count changed from {old} to {n}")
        elif n < old:
            if rescue_underway:
                inc.people_count = n
                add(EventKind.PEOPLE_CHANGED, f"People count changed from {old} to {n} (rescue is under way)")
            else:
                add(EventKind.DETAILS_ADDED,
                    f"Another report says {n} people. Keeping the higher count of {old} until rescue starts")
        if inc.people_count is not None:
            inc.peak_people_count = max(inc.peak_people_count or 0, inc.people_count)

    # Rule 3: HELP NEEDED. Remember every kind of help; show the most critical as the primary one.
    detail = (ex.resource_detail or "").strip()
    parts = [p.strip() for p in inc.resource_detail.split(",") if p.strip()]
    new_need = ex.resource is not Resource.UNKNOWN and ex.resource.value not in inc.needs
    new_detail = bool(detail) and detail.casefold() not in {p.casefold() for p in parts} and len(parts) < MAX_DETAIL_ITEMS
    if new_need:
        inc.needs.append(ex.resource.value)
    if new_detail:
        parts.append(detail)
        inc.resource_detail = ", ".join(parts)
    inc.resource = primary_resource(inc.needs)
    if new_need or new_detail:
        add(EventKind.DETAILS_ADDED, "Help needed: " + (inc.resource_detail or inc.resource.label.lower()))

    # Rule 4: VULNERABLE PEOPLE. Add any new group.
    fresh = [v for v in ex.vulnerable if v not in inc.vulnerable]
    if fresh:
        inc.vulnerable.extend(fresh)
        add(EventKind.DETAILS_ADDED, "Vulnerable people reported: " + ", ".join(fresh))

    # Rule 5: URGENCY. It only goes UP while the incident is open. It follows the facts, and a report of
    # worsening ("water rising") raises it one more step, at most once per 30 minutes.
    if is_new:
        level, reasons = compute_urgency(inc, extra_points=1 if ex.worsening else 0)
        if ex.worsening:
            reasons.append("reported worsening")
        inc.urgency = level
        add(EventKind.DETAILS_ADDED, f"Urgency set to {level.label}" + (f" ({'; '.join(reasons)})" if reasons else ""))
    else:
        level, reasons = compute_urgency(inc)
        candidate = level if level.rank > inc.urgency.rank else inc.urgency
        if ex.worsening and _worsening_allowed(inc, when):
            raised = candidate.raised()
            if raised.rank > candidate.rank:
                candidate = raised
                reasons.insert(0, "reported worsening")
        if candidate.rank > inc.urgency.rank:
            old_level, inc.urgency = inc.urgency, candidate
            add(EventKind.ESCALATED, f"Urgency raised from {old_level.label} to {candidate.label}"
                + (f" ({'; '.join(reasons)})" if reasons else ""))

    # Rule 6: STATUS. Active -> Rescue in progress -> Resolved. Resolved is final (Status.can_move_to).
    target = {StatusHint.RESCUE_STARTED: Status.RESCUE_IN_PROGRESS, StatusHint.RESOLVED: Status.RESOLVED}.get(hint)
    if target is not None and not (is_new and target is Status.RESOLVED) \
            and target is not inc.status and inc.status.can_move_to(target):
        old_status, inc.status = inc.status, target
        add(EventKind.STATUS_CHANGED, f"Status changed from {old_status.label} to {target.label}")

    inc.summary = build_summary(inc)
    # A report that adds nothing new (a pure duplicate) causes no event. It still counts in inc.message_ids.
    return added


# =====================================================================
# 5. COMMAND LINE + SELF-TEST
# =====================================================================

RULES_TEXT = f"""THE RULES (plain language)

Counting people
  - "2 more people"  -> added to the total.
  - A new plain number -> replaces the old one when it is higher.
  - A LOWER number before rescue starts -> treated as a conflicting report; the higher (safer) number is kept
    and a note is added to the timeline.
  - A lower number once rescue is under way -> accepted (people are being rescued; fewer remain).
  - The highest number ever reported is always remembered.

Urgency (only goes up while the incident is open)
  - Medical help needed ................ +2 points
  - Rescue or shelter needed ........... +1
  - Vulnerable people (elderly, child, infant, pregnant, disabled, patient) .... +1
  - {LARGE_GROUP}+ people ...................... +1      {VERY_LARGE_GROUP}+ people ...... +1 more
  - 0 points Low, 1 Medium, 2-3 High, 4+ Critical.
  - A report of worsening ("water rising") raises urgency one more step, at most once every
    {int(WORSENING_COOLDOWN.total_seconds() // 60)} minutes, so duplicates cannot inflate it.

Status
  - "Rescue team reached" -> Rescue in progress.   "Everyone rescued" -> Resolved.
  - Resolved is FINAL. Nothing reopens it.

Help needed
  - Every kind of help is remembered. The primary one shown is: medical, then rescue, shelter, food and water,
    road clearance, other.

Location
  - Keep the best estimate. A specific place beats a neighbourhood guess. A known place is never erased.
  - The reporter's own words behind the chosen place are kept as evidence (all reports stay in the grouped list).
"""


def _selftest() -> None:
    t0 = datetime(2026, 1, 1, 8, 0)

    def msg(minutes, text="x", id=None, source="sms"):
        return Message(id=id or f"m{minutes}", timestamp=t0 + timedelta(minutes=minutes), source=source, text=text)

    def ex(**kw):
        base = dict(intent="emergency", confidence=0.9)
        base.update(kw)
        return Extraction(**base)

    def loc(name, lid, kind, conf, lat=1.0, lon=1.0):
        return ResolvedLocation(name=name, landmark_id=lid, lat=lat, lon=lon, confidence=conf, kind=kind)

    # ---- the full story: create, elderly, more people, water rising, rescue, resolved ----
    area = loc("Rivertown", "rt", "area", 0.7)
    inc = open_incident("INC-1", msg(0, id="a1", source="call"),
                        ex(location_text="near the old bridge", people_count=6, resource="rescue",
                           resource_detail="boat"), area)
    assert inc.urgency is Urgency.MEDIUM and inc.people_count == 6 and inc.peak_people_count == 6
    assert inc.events[0].kind is EventKind.CREATED and "phone call" in inc.events[0].text
    assert inc.location.name == "Rivertown" and inc.location_text == "near the old bridge"
    assert inc.summary == "6 people. Needs: boat."

    ev = apply_update(inc, msg(16, id="a2"), ex(vulnerable=["elderly"], says_same_place=True), None)
    assert inc.urgency is Urgency.HIGH and any(e.kind is EventKind.ESCALATED for e in ev)
    ev = apply_update(inc, msg(25, id="a3"), ex(people_count=10), None)
    assert inc.people_count == 10 and inc.urgency is Urgency.HIGH
    assert any("changed from 6 to 10" in e.text for e in ev)
    ev = apply_update(inc, msg(40, id="a4"), ex(worsening=True), None)            # "water rising"
    assert inc.urgency is Urgency.CRITICAL and "reported worsening" in ev[0].text
    ev = apply_update(inc, msg(45, id="a5"), ex(worsening=True), None)            # duplicate report
    assert ev == [] and inc.urgency is Urgency.CRITICAL
    ev = apply_update(inc, msg(65, id="a6"), ex(status_hint="rescue_started"), None)
    assert inc.status is Status.RESCUE_IN_PROGRESS and ev[0].kind is EventKind.STATUS_CHANGED
    ev = apply_update(inc, msg(90, id="a7"), ex(status_hint="resolved", people_count=3), None)
    assert inc.status is Status.RESOLVED and inc.people_count == 10                 # count ignored on the resolve message
    assert inc.message_ids == ["a1", "a2", "a3", "a4", "a5", "a6", "a7"] and inc.last_update == t0 + timedelta(minutes=90)
    # resolved is final: nothing changes it, not even a scary report
    before = inc.to_dict()
    assert apply_update(inc, msg(120, id="a8"), ex(worsening=True, people_count=50, status_hint="rescue_started"), None) == []
    assert inc.to_dict() == before

    # ---- worsening is limited to once per 30 minutes ----
    inc = open_incident("INC-2", msg(0), ex(people_count=4, resource="rescue", vulnerable=["child"]), None)
    assert inc.urgency is Urgency.HIGH
    apply_update(inc, msg(5), ex(worsening=True), None)
    assert inc.urgency is Urgency.CRITICAL
    inc2 = open_incident("INC-3", msg(0), ex(people_count=4, resource="rescue"), None)    # Medium
    apply_update(inc2, msg(5), ex(worsening=True), None)
    assert inc2.urgency is Urgency.HIGH
    apply_update(inc2, msg(15), ex(worsening=True), None)                          # only 10 minutes later: ignored
    assert inc2.urgency is Urgency.HIGH
    apply_update(inc2, msg(40), ex(worsening=True), None)                          # 35 minutes later: allowed
    assert inc2.urgency is Urgency.CRITICAL

    # ---- counts: additional, conflicting, remaining, peak ----
    inc = open_incident("INC-4", msg(0), ex(people_count=10, resource="rescue"), None)
    apply_update(inc, msg(5), ex(people_count=2, count_is_additional=True), None)
    assert inc.people_count == 12
    ev = apply_update(inc, msg(6), ex(people_count=6), None)                       # another reporter says fewer
    assert inc.people_count == 12 and "Keeping the higher count" in ev[0].text
    assert apply_update(inc, msg(7), ex(people_count=6), None) == []               # same note is not repeated
    apply_update(inc, msg(10), ex(people_count=35), None)
    ev = apply_update(inc, msg(20), ex(people_count=12, status_hint="rescue_started"), None)
    assert inc.people_count == 12 and inc.peak_people_count == 35 and inc.status is Status.RESCUE_IN_PROGRESS
    assert any("rescue is under way" in e.text for e in ev)
    inc = open_incident("INC-5", msg(0), ex(resource="rescue"), None)              # count not known yet
    assert inc.people_count is None and inc.summary == "Number of people not known. Needs: rescue."
    apply_update(inc, msg(3), ex(people_count=2, count_is_additional=True), None)
    assert inc.people_count == 2

    # ---- the medical story: elderly diabetic needs insulin = high urgency, medical is primary ----
    inc = open_incident("INC-6", msg(0), ex(people_count=6, resource="rescue", resource_detail="boat"), None)
    apply_update(inc, msg(5), ex(resource="medical", resource_detail="insulin", vulnerable=["elderly", "patient"]), None)
    assert inc.resource is Resource.MEDICAL and inc.resource_detail == "boat, insulin"
    assert inc.urgency is Urgency.CRITICAL                                         # medical 2 + rescue 1 + vulnerable 1
    first = open_incident("INC-7", msg(0), ex(people_count=1, resource="medical", resource_detail="insulin",
                                             vulnerable=["elderly"]), None)
    assert first.urgency is Urgency.HIGH and first.resource is Resource.MEDICAL    # "high-urgency medical emergency"
    assert first.summary == "1 person, including elderly. Needs: insulin."

    # ---- food and water: medium, then high at 50+ people; road clearing with nobody hurt is low ----
    inc = open_incident("INC-8", msg(0), ex(people_count=40, resource="food_water", resource_detail="food"), None)
    assert inc.urgency is Urgency.MEDIUM
    apply_update(inc, msg(10), ex(people_count=60), None)
    assert inc.urgency is Urgency.HIGH
    inc = open_incident("INC-9", msg(0), ex(people_count=0, resource="road_clearance", resource_detail="tree removal"), None)
    assert inc.urgency is Urgency.LOW and inc.summary == "No people reported hurt or trapped. Needs: tree removal."
    inc = open_incident("INC-10", msg(0), ex(), None)
    assert inc.urgency is Urgency.LOW and inc.resource is Resource.UNKNOWN
    inc = open_incident("INC-11", msg(0), ex(worsening=True, resource="rescue", people_count=3), None)
    assert inc.urgency is Urgency.HIGH                                              # worsening counts at creation too
    inc = open_incident("INC-12", msg(0), ex(status_hint="resolved", resource="rescue"), None)
    assert inc.status is Status.ACTIVE                                              # nothing to resolve yet

    # ---- location: specific beats area; never erased; wording kept ----
    inc = open_incident("INC-13", msg(0), ex(location_text="Rivertown", resource="rescue"), loc("Rivertown", "rt", "area", 0.7))
    apply_update(inc, msg(1), ex(), None)
    assert inc.location.landmark_id == "rt"                                         # None never erases
    apply_update(inc, msg(2), ex(location_text="behind the station"), loc("Station", "st", "mrts", 0.6))
    assert inc.location.landmark_id == "st" and inc.location_text == "behind the station"
    apply_update(inc, msg(3), ex(location_text="Hilltop"), loc("Hilltop", "ht", "area", 0.7))
    assert inc.location.landmark_id == "st"                                         # a coarse guess does not replace it
    inc = open_incident("INC-14", msg(0), ex(location_text="near the pump house", resource="rescue"), None)
    assert inc.location is None and inc.location_text == "near the pump house"
    apply_update(inc, msg(1), ex(location_text="Rivertown"), loc("Rivertown", "rt", "area", 0.7))
    assert inc.location.name == "Rivertown" and inc.location_text == "Rivertown"     # words behind the chosen place

    # ---- housekeeping ----
    assert primary_resource(["food_water", "rescue"]) is Resource.RESCUE and primary_resource([]) is Resource.UNKNOWN
    level, reasons = compute_urgency(Incident(id="x", created_at=t0, last_update=t0, needs=["medical"], people_count=12))
    assert level is Urgency.HIGH and "medical help needed" in reasons
    copy = Incident.from_dict(inc.to_dict())
    assert copy.to_dict() == inc.to_dict()
    print("rules.py: all checks passed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The decision rules (plain code, no AI).")
    parser.add_argument("--rules", action="store_true", help="print every rule in plain language")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        _selftest()
        return 0
    if args.rules:
        print(RULES_TEXT)
        return 0
    parser.error("use --rules or --selftest")
    return 2


if __name__ == "__main__":
    sys.exit(main())