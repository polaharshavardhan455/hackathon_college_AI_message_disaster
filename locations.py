"""
locations.py  --  step 3 of the pipeline.

Job: turn the place words from a report ("behind the MRTS station, Vijayanagar side") into a
best-known location: a name, coordinates, and HOW SURE we are. It never pretends to be exact.

How it decides (in this order, most specific first):
    1. A specific place is named          ("Velachery MRTS", "Stanley hospital")      -> high confidence
    2. A generic word + a nearby name      ("the station" + "Vijayanagar")              -> medium confidence
    3. Only a locality is named            ("Vijayanagar")                              -> medium
    4. Only an area is named               ("Velachery")                                -> coarse, lower confidence
    5. Nothing in our list matches         -> returns None (the raw words are kept by the caller)

It copes with: different spellings ("Velacheri"), Tamil script with endings ("வேளச்சேரியில்"),
Tanglish endings ("Velacherila"), dots/spaces ("T.Nagar", "tnagar"), and directions ("behind", "west gate").
If the text could mean two far-apart places it picks the first mentioned, lowers the confidence and lists the other.

It also provides `proximity(a, b)`, a 0-1 score of how likely two locations are the same spot.
cluster.py uses that to decide whether two reports belong to the same incident.

Uses only the Python standard library.

Try it
    python locations.py "Behind the MRTS station, on the Vijayanagar side"
    python locations.py --check                       (validate landmarks.json)
    python locations.py --suggest extractions.json    (find place names missing from landmarks.json)
    python locations.py --selftest
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from models import ResolvedLocation

log = logging.getLogger("locations")

DEFAULT_LANDMARKS = Path(__file__).with_name("landmarks.json")
COARSE_KINDS = {"area", "locality"}              # points that are a neighbourhood centre, not an exact spot
CITY_BOX = (12.70, 13.50, 79.80, 80.50)           # lat_min, lat_max, lon_min, lon_max (sanity check only)

# Words that describe a TYPE of place. On their own they cannot identify a place.
GENERIC = {"station", "mrts", "railway", "road", "street", "bridge", "hospital", "bus", "stand", "stop",
           "school", "temple", "market", "signal", "junction", "gate", "side", "lake", "main", "the", "of",
           "and", "near", "airport", "terminus", "nagar", "colony", "area", "side", "beach"}

# A generic word in the text points to these landmark types.
TYPE_HINTS = {
    "mrts": {"mrts"}, "station": {"mrts", "railway_station"}, "railway": {"railway_station", "mrts"},
    "hospital": {"hospital"}, "bridge": {"bridge"}, "lake": {"lake"}, "airport": {"airport"},
    "terminus": {"bus_terminus"},
}

_RELATION = re.compile(r"\b(near|behind|opposite|beside|next to|in front of|close to|adjacent to|across from|"
                       r"inside|backside|back side|back of)\b")
_QUALIFIER = re.compile(r"\b((?:north|south|east|west|northeast|northwest|southeast|southwest)\s+"
                        r"(?:side|gate|end|entrance|exit|block|wing)|(?:back|front|rear)\s*"
                        r"(?:side|gate|road|entrance)|subway|platform\s*\d+)\b")


class LocationError(Exception):
    """landmarks.json is missing or unusable."""


# =====================================================================
# 1. TEXT NORMALISATION AND TOKEN MATCHING
# =====================================================================

def normalize_tokens(text: object) -> list[str]:
    """Lower-case, strip punctuation (but keep Tamil/Hindi letter signs), split into words.
    'T.Nagar' -> ['tnagar'],  'M.R.T.S' -> ['mrts'],  'Near, Velachery!' -> ['near', 'velachery']"""
    s = unicodedata.normalize("NFC", str(text or "")).casefold().replace("&", " and ")
    out: list[str] = []
    for i, ch in enumerate(s):
        cat = unicodedata.category(ch)
        if ch == "." and 0 < i < len(s) - 1 and unicodedata.category(s[i - 1])[0] in "LNM" \
                and unicodedata.category(s[i + 1])[0] in "LN":
            continue                                  # dot between letters is joined, not split
        if cat[0] in "LNM":
            out.append(ch)
        elif cat == "Cf":
            continue                                  # invisible joiners: drop
        else:
            out.append(" ")
    return "".join(out).split()


@lru_cache(maxsize=200_000)
def token_sim(text_token: str, alias_token: str) -> float:
    """How well one word from the report matches one word of a landmark name (0 = no match)."""
    a, b = text_token, alias_token
    if a == b:
        return 1.0
    if not (a.isascii() and b.isascii()):
        # Tamil/Hindi words get endings glued on ("வேளச்சேரியில்" = "in Velachery"): accept a longer word.
        return 0.9 if len(b) >= 3 and a.startswith(b) else 0.0
    if len(b) >= 6 and a.startswith(b) and len(a) - len(b) <= 4:
        return 0.9                                    # Tanglish endings: "velacherila"
    if len(a) >= 5 and len(b) >= 5:
        ratio = SequenceMatcher(None, a, b).ratio()   # spelling variants: "velacheri"
        if ratio >= 0.85:
            return ratio
    return 0.0


def find_sequence(tokens: list[str], alias: list[str]) -> tuple[float, int]:
    """Best place in the report where ALL words of `alias` appear in order. Returns (score, position)."""
    n = len(alias)
    best = (0.0, -1)
    for i in range(len(tokens) - n + 1):
        sims = [token_sim(tokens[i + j], alias[j]) for j in range(n)]
        if min(sims) > 0:
            score = sum(sims) / n
            if score > best[0]:
                best = (score, i)
    return best


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line distance in metres (haversine)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


def proximity(a: ResolvedLocation | None, b: ResolvedLocation | None) -> float:
    """0-1: how likely two locations are the same spot. Used by cluster.py.
         1.0  same landmark              0.9  within 150 m        0.75 within 400 m
         0.5  within 800 m               0.25 within 1.5 km       0.0  further, or unknown
    A neighbourhood-level point is coarse, so it never scores above 0.5 against anything."""
    if a is None or b is None:
        return 0.0
    if a.landmark_id and a.landmark_id == b.landmark_id:
        return 1.0
    if not (a.has_coordinates and b.has_coordinates):
        return 0.0
    d = distance_m(a.lat, a.lon, b.lat, b.lon)
    score = 0.9 if d <= 150 else 0.75 if d <= 400 else 0.5 if d <= 800 else 0.25 if d <= 1500 else 0.0
    coarse = a.kind in COARSE_KINDS or b.kind in COARSE_KINDS
    if coarse:
        score = min(score, 0.5)
        if a.area_id and a.area_id == b.area_id:     # a specific place inside the named area
            score = 0.5
    return score


# =====================================================================
# 2. THE LANDMARK LIST
# =====================================================================

@dataclass
class Landmark:
    id: str
    name: str
    kind: str
    lat: float
    lon: float
    area_id: str | None
    aliases: list[list[str]]        # each alias as a list of normalised words
    context: list[list[str]]        # nearby names that point to this place

    @property
    def coarse(self) -> bool:
        return self.kind in COARSE_KINDS


def _alias_variants(raw: list[str]) -> list[list[str]]:
    """Normalise aliases; also add the joined form of short-initial names ('t nagar' -> 'tnagar')."""
    out: list[list[str]] = []
    for text in raw:
        toks = normalize_tokens(text)
        if not toks or all(t in GENERIC for t in toks):
            continue                                  # a name made only of generic words cannot identify anything
        out.append(toks)
        if len(toks) >= 2 and len(toks[0]) <= 3:
            out.append(["".join(toks)])               # "t nagar" -> "tnagar", "k k nagar" -> "kknagar"
    return out


def load_landmarks(path: str | Path) -> tuple[list[Landmark], list[str]]:
    """Read landmarks.json. Returns (landmarks, warnings). Raises LocationError for fatal problems."""
    file = Path(path)
    if not file.is_file():
        raise LocationError(f"Landmark file not found: {file}")
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LocationError(f"{file.name} is not valid JSON (line {exc.lineno}): {exc.msg}") from exc
    items = data.get("landmarks") if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        raise LocationError(f"{file.name} has no 'landmarks' list")

    warnings: list[str] = []
    landmarks: list[Landmark] = []
    seen_ids: set[str] = set()
    for n, item in enumerate(items, start=1):
        try:
            lm_id, name = str(item["id"]).strip(), str(item["name"]).strip()
            lat, lon = float(item["lat"]), float(item["lon"])
        except (KeyError, TypeError, ValueError):
            warnings.append(f"Entry {n}: needs id, name, lat and lon. Skipped.")
            continue
        if not lm_id or not name:
            warnings.append(f"Entry {n}: empty id or name. Skipped.")
            continue
        if lm_id in seen_ids:
            warnings.append(f"Entry {n}: duplicate id '{lm_id}'. Skipped.")
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            warnings.append(f"'{lm_id}': coordinates are not on Earth ({lat}, {lon}). Skipped.")
            continue
        if not (CITY_BOX[0] <= lat <= CITY_BOX[1] and CITY_BOX[2] <= lon <= CITY_BOX[3]):
            warnings.append(f"'{lm_id}': ({lat}, {lon}) is outside the expected city area. Check lat/lon are not swapped.")
        seen_ids.add(lm_id)
        kind = str(item.get("type") or "landmark").strip().lower()
        aliases = _alias_variants([name] + list(item.get("aliases") or []))
        if not aliases:
            warnings.append(f"'{lm_id}': no usable name or alias (only generic words). Add a more specific alias.")
        landmarks.append(Landmark(
            id=lm_id, name=name, kind=kind, lat=lat, lon=lon,
            area_id=(str(item["area"]).strip() or None) if item.get("area") else (lm_id if kind == "area" else None),
            aliases=aliases,
            context=[t for t in (normalize_tokens(c) for c in (item.get("context") or [])) if t]))

    ids = {lm.id for lm in landmarks}
    for lm in landmarks:
        if lm.area_id and lm.area_id not in ids:
            warnings.append(f"'{lm.id}': area '{lm.area_id}' is not in the list.")
    owners: dict[tuple[str, ...], list[str]] = {}
    for lm in landmarks:
        for alias in lm.aliases:
            owners.setdefault(tuple(alias), []).append(lm.id)
    for alias, who in owners.items():
        if len(set(who)) > 1:
            warnings.append(f"Alias '{' '.join(alias)}' belongs to several places: {', '.join(sorted(set(who)))}.")
    if not landmarks:
        raise LocationError(f"{file.name}: no usable landmarks.")
    return landmarks, warnings


# =====================================================================
# 3. THE RESOLVER
# =====================================================================

class LocationResolver:
    def __init__(self, landmarks_path: str | Path = DEFAULT_LANDMARKS):
        self.landmarks, self.warnings = load_landmarks(landmarks_path)
        for w in self.warnings:
            log.warning("landmarks: %s", w)
        self._by_id = {lm.id: lm for lm in self.landmarks}
        # A specific place also answers to the names of the area it sits in ("station" + "Velachery").
        self._context: dict[str, list[list[str]]] = {}
        for lm in self.landmarks:
            terms = list(lm.context)
            area = self._by_id.get(lm.area_id) if lm.area_id and lm.area_id != lm.id else None
            if area:
                terms += [a for a in area.aliases if not all(t in GENERIC for t in a)]
            self._context[lm.id] = terms
        self.unmatched: Counter[str] = Counter()

    def get(self, landmark_id: str) -> Landmark | None:
        return self._by_id.get(landmark_id)

    def resolve(self, text: str | None) -> ResolvedLocation | None:
        """Best-known location for these place words, or None if nothing in the list matches."""
        raw = str(text or "").strip()[:300]
        tokens = normalize_tokens(raw)
        if not tokens:
            return None

        entities = []                                  # (score, position, landmark, matched words)
        for lm in self.landmarks:
            best = (0.0, -1, 0)
            for alias in lm.aliases:
                score, pos = find_sequence(tokens, alias)
                if score > best[0]:
                    best = (score, pos, len(alias))
            if best[0] >= 0.8:
                entities.append((best[0], best[1], lm, " ".join(tokens[best[1]:best[1] + best[2]])))

        specific = sorted((e for e in entities if not e[2].coarse), key=lambda e: (-e[0], e[1]))
        localities = sorted((e for e in entities if e[2].kind == "locality"), key=lambda e: (-e[0], e[1]))
        areas = sorted((e for e in entities if e[2].kind == "area"), key=lambda e: (-e[0], e[1]))

        # Generic word ("the station") + a nearby name ("Vijayanagar") -> the matching specific place
        inferred = []
        hinted = {t for tok in tokens for t in TYPE_HINTS.get(tok, ())}
        if hinted:
            named = {e[2].id for e in specific}
            for lm in self.landmarks:
                if lm.coarse or lm.kind not in hinted or lm.id in named:
                    continue
                for term in self._context[lm.id]:
                    score, pos = find_sequence(tokens, term)
                    if score >= 0.9:
                        inferred.append((0.6, pos, lm, f"{'/'.join(sorted(hinted))} + {' '.join(term)}"))
                        break
            inferred.sort(key=lambda e: e[1])

        if specific:
            pool, base = specific, 0.9
        elif inferred:
            pool, base = inferred, 1.0                 # score already holds the 0.6
        elif localities:
            pool, base = localities, 0.7
        elif areas:
            pool, base = areas, 0.7
        else:
            self.unmatched[" ".join(tokens)] += 1
            return None

        score, _, chosen, matched = pool[0]
        confidence = base * score
        # Could the text also mean a clearly different place? Then say so and lower the confidence.
        alternatives = [e[2].name for e in pool[1:]
                        if e[2].id != chosen.id and
                        distance_m(chosen.lat, chosen.lon, e[2].lat, e[2].lon) > 2000]
        if alternatives:
            confidence *= 0.7

        joined = " ".join(tokens)
        relation = _RELATION.search(joined)
        qualifiers = [m.group(1) for m in _QUALIFIER.finditer(joined)]
        qualifier = "; ".join(dict.fromkeys(([relation.group(1)] if relation else []) + qualifiers))

        return ResolvedLocation(
            name=chosen.name, landmark_id=chosen.id, lat=chosen.lat, lon=chosen.lon,
            confidence=round(confidence, 2), matched_text=matched, kind=chosen.kind,
            area_id=chosen.area_id, qualifier=qualifier, alternatives=alternatives)

    def suggest(self, texts: list[str | None], top: int = 20) -> list[tuple[str, int]]:
        """Place words we could NOT match, most frequent first: these are the ones to add to landmarks.json."""
        counts: Counter[str] = Counter()
        for t in texts:
            if t and str(t).strip() and self.resolve(t) is None:
                words = _RELATION.sub(" ", " ".join(normalize_tokens(t))).split()
                key = " ".join(w for w in words if w not in {"the", "a", "on", "at", "side"})
                if key:
                    counts[key] += 1          # "near X" and "behind the X" count as the same missing place
        return counts.most_common(top)


# =====================================================================
# 4. COMMAND LINE + SELF-TEST
# =====================================================================

def _selftest() -> None:
    # ---- part 1: the matching logic, on a tiny invented list ----
    mini = {"landmarks": [
        {"id": "rt", "name": "Rivertown", "type": "area", "lat": 13.00, "lon": 80.20, "aliases": ["rivertown"]},
        {"id": "rt_stn", "name": "Rivertown Station", "type": "mrts", "lat": 13.0010, "lon": 80.2010, "area": "rt",
         "aliases": ["rivertown mrts"], "context": ["eastside"]},
        {"id": "rt_east", "name": "Eastside", "type": "locality", "lat": 13.0030, "lon": 80.2040, "area": "rt",
         "aliases": ["eastside"]},
        {"id": "ht", "name": "Hilltop", "type": "area", "lat": 13.10, "lon": 80.30, "aliases": ["hilltop", "மலை உச்சி"]},
        {"id": "city_hosp", "name": "City Hospital", "type": "hospital", "lat": 13.0100, "lon": 80.2100,
         "area": "rt", "aliases": ["city hospital", "cityhospital"]},
        {"id": "ab", "name": "A. B. Nagar", "type": "area", "lat": 13.05, "lon": 80.25, "aliases": ["a b nagar"]},
    ]}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mini.json"
        path.write_text(json.dumps(mini, ensure_ascii=False), encoding="utf-8")
        r = LocationResolver(path)

        a = r.resolve("Near the Rivertown MRTS west gate.")
        assert a.landmark_id == "rt_stn" and a.confidence == 0.9 and a.kind == "mrts"
        assert a.qualifier == "near; west gate", a.qualifier
        assert a.lat == 13.0010 and a.area_id == "rt"

        b = r.resolve("Behind the station, on the Eastside side.")          # generic word + nearby name
        assert b.landmark_id == "rt_stn" and b.confidence == 0.6 and "eastside" in b.matched_text

        assert r.resolve("opposite the medical shop near the station backside road") is None   # nothing to go on
        assert r.unmatched

        c = r.resolve("Rivertown")                                          # only an area
        assert c.landmark_id == "rt" and c.kind == "area" and c.confidence == 0.7

        assert r.resolve("Rivertwn mrts").landmark_id == "rt_stn"           # spelling mistake
        assert r.resolve("Rivertownla irukkom").landmark_id == "rt"         # Tanglish ending
        assert r.resolve("மலை உச்சியில் 10 பேர்").landmark_id == "ht"        # Tamil script with an ending
        assert r.resolve("A.B.Nagar").landmark_id == "ab" and r.resolve("ab nagar") is None  # dots joined; no loose guess
        assert r.resolve("city hospital gate").landmark_id == "city_hosp"
        assert r.resolve("") is None and r.resolve(None) is None and r.resolve("...!!") is None

        d = r.resolve("between Hilltop and Rivertown")                      # two far-apart areas
        assert d.landmark_id == "ht" and d.alternatives == ["Rivertown"] and d.confidence == 0.49

        # specific beats area: a place inside the named area wins
        assert r.resolve("Rivertown MRTS, Rivertown").landmark_id == "rt_stn"

        # proximity
        L = {k: r.resolve(t) for k, t in {"stn": "Rivertown MRTS", "area": "Rivertown",
                                          "east": "Eastside", "far": "Hilltop", "hosp": "City Hospital"}.items()}
        assert proximity(L["stn"], L["stn"]) == 1.0
        assert proximity(L["stn"], L["area"]) == 0.5                        # specific place inside a named area
        assert proximity(L["stn"], L["far"]) == 0.0 and proximity(None, L["stn"]) == 0.0
        assert 0.0 < proximity(L["stn"], L["hosp"]) < 0.9                   # nearby but a different place
        assert r.suggest(["Zeta Colony", "near the zeta colony", "Rivertown", None, ""]) == [("zeta colony", 2)]

        # bad files
        bad = Path(tmp) / "bad.json"
        bad.write_text("{oops", encoding="utf-8")
        for p, msg in ((bad, "not valid JSON"), (Path(tmp) / "missing.json", "not found")):
            try:
                load_landmarks(p)
                raise SystemExit("FAIL: bad file accepted")
            except LocationError as exc:
                assert msg in str(exc)
        odd = Path(tmp) / "odd.json"
        odd.write_text(json.dumps({"landmarks": [
            {"id": "x", "name": "X Place", "lat": 13.0, "lon": 80.2, "aliases": ["x place"]},
            {"id": "x", "name": "Dup", "lat": 13.0, "lon": 80.2},
            {"id": "y", "name": "No coords"},
            {"id": "z", "name": "Swapped", "lat": 80.2, "lon": 13.0, "aliases": ["swapped"]},
            {"id": "g", "name": "Station", "lat": 13.0, "lon": 80.2}]}), encoding="utf-8")
        lms, warns = load_landmarks(odd)
        text = " ".join(warns)
        assert {l.id for l in lms} == {"x", "z", "g"}
        assert "duplicate id" in text and "needs id, name" in text and "outside the expected" in text
        assert "only generic words" in text

    # ---- part 2: the real landmarks.json, using the three descriptions from the challenge brief ----
    if DEFAULT_LANDMARKS.is_file():
        real = LocationResolver(DEFAULT_LANDMARKS)
        a = real.resolve("Near the Velachery MRTS west gate.")
        b = real.resolve("Behind the MRTS station, on the Vijayanagar side.")
        c = real.resolve("Opposite the medical shop near the station backside road.")
        assert a.landmark_id == b.landmark_id == "velachery_mrts"            # two descriptions, one place
        assert a.confidence > b.confidence >= 0.6
        assert c is None                                                       # honest: cannot be placed from words alone
        assert proximity(a, b) == 1.0
        assert real.resolve("வேளச்சேரியில் 10 பேர் மாட்டிக்கிட்டாங்க").landmark_id == "velachery"
        assert real.resolve("T.Nagar bus stand").landmark_id == "t_nagar"
        assert real.resolve("near Velacheri MRTS").landmark_id == "velachery_mrts"
        assert proximity(real.resolve("Velachery MRTS"), real.resolve("Taramani MRTS")) < 0.5   # different stations
        print(f"real landmarks.json: {len(real.landmarks)} places, {len(real.warnings)} warnings")
    print("locations.py: all checks passed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Turn place words into a best-known location.")
    parser.add_argument("text", nargs="?", help="place words to look up")
    parser.add_argument("--landmarks", default=str(DEFAULT_LANDMARKS), help="path to landmarks.json")
    parser.add_argument("--check", action="store_true", help="validate landmarks.json and show warnings")
    parser.add_argument("--suggest", metavar="EXTRACTIONS_JSON",
                        help="list place words from extract.py output that are missing from landmarks.json")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if args.selftest:
        _selftest()
        return 0
    try:
        resolver = LocationResolver(args.landmarks)
    except LocationError as exc:
        print(f"Problem: {exc}")
        return 1

    if args.check:
        print(f"{len(resolver.landmarks)} places loaded from {args.landmarks}")
        for w in resolver.warnings:
            print("  warning:", w)
        if not resolver.warnings:
            print("  no problems found")
        return 0

    if args.suggest:
        data = json.loads(Path(args.suggest).read_text(encoding="utf-8"))
        texts = [e.get("location_text") for e in data.get("extractions", {}).values()]
        texts = [t for t in texts if t]
        missing = resolver.suggest(texts, top=25)
        matched = len(texts) - sum(c for _, c in resolver.suggest(texts, top=10_000))
        print(f"{len(texts)} reports had place words; {matched} matched ({100 * matched // max(1, len(texts))}%).")
        if not missing:
            print("Every place was matched.")
            return 0
        print("\nNot matched, most frequent first. Add the important ones to landmarks.json:\n")
        for words, count in missing:
            print(f'  {count:4d} x  "{words}"')
        print("\nTemplate for one entry (get lat/lon from Google Maps: right-click the spot, click the numbers):")
        print(json.dumps({"id": "short_id", "name": missing[0][0].title(), "type": "landmark", "lat": 0.0,
                          "lon": 0.0, "area": "area_id_or_null", "aliases": [missing[0][0]], "context": []}, indent=1))
        return 0

    if not args.text:
        parser.error("give place words, or use --check, --suggest or --selftest")
    result = resolver.resolve(args.text)
    print(json.dumps(result.to_dict() if result else {"result": "no match in landmarks.json"},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())