from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import re


@dataclass(frozen=True)
class TechRule:
    name: str
    prereqs: List[str]


@dataclass(frozen=True)
class UnitRule:
    name: str
    cost: int
    attack: int
    defense: int
    hp: int
    firepower: int
    moves: int
    req_techs: List[str]
    flags: List[str]
    obsolete_by: Optional[str]


@dataclass(frozen=True)
class BuildingRule:
    name: str
    cost: int
    req_techs: List[str]
    req_buildings: List[str]
    genus: str
    flags: List[str]


@dataclass(frozen=True)
class Civ2Civ3Ruleset:
    techs: Tuple[str, ...]
    tech_prereqs: Dict[str, List[str]]
    units: Tuple[UnitRule, ...]
    buildings: Tuple[BuildingRule, ...]


_QUOTED_RE = re.compile(r"\"([^\"]+)\"")


def _ruleset_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "freeciv" / "data" / "minimal"


def _strip_inline_comment(line: str) -> str:
    in_quote = False
    for idx, ch in enumerate(line):
        if ch == '"':
            in_quote = not in_quote
        if ch == ";" and not in_quote:
            return line[:idx].rstrip()
    return line.rstrip()


def _extract_first_quoted(value: str) -> str:
    matches = _QUOTED_RE.findall(value)
    return matches[0].strip() if matches else value.strip()


def _normalize_name(name: str) -> str:
    if name.startswith("?") and ":" in name:
        return name.split(":", 1)[1]
    return name


def _parse_req_block(lines: Sequence[str]) -> List[Dict[str, str]]:
    tokens_by_line = [_QUOTED_RE.findall(line) for line in lines]
    if not tokens_by_line:
        return []
    header = tokens_by_line[0]
    if header and header[0].lower() == "type":
        columns = header
        data_tokens = [tok for line in tokens_by_line[1:] for tok in line]
    else:
        columns = ["type", "name", "range"]
        data_tokens = [tok for line in tokens_by_line for tok in line]
    stride = len(columns)
    reqs: List[Dict[str, str]] = []
    for idx in range(0, len(data_tokens), stride):
        chunk = data_tokens[idx : idx + stride]
        if len(chunk) < stride:
            break
        reqs.append(dict(zip(columns, chunk)))
    return reqs


def _parse_ruleset_sections(
    path: Path, section_prefix: str
) -> List[Dict[str, object]]:
    sections: List[Dict[str, object]] = []
    current: Optional[Dict[str, object]] = None
    current_key: Optional[str] = None
    current_value: List[str] = []
    in_reqs = False
    req_lines: List[str] = []

    def flush_key() -> None:
        nonlocal current_key, current_value
        if current is None or current_key is None:
            return
        current["fields"][current_key] = " ".join(current_value).strip()
        current_key = None
        current_value = []

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            stripped = _strip_inline_comment(raw.strip())
            if not stripped:
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                flush_key()
                if current is not None:
                    sections.append(current)
                section_name = stripped[1:-1].strip()
                if section_name.startswith(section_prefix):
                    current = {"section": section_name, "fields": {}, "reqs": []}
                else:
                    current = None
                in_reqs = False
                req_lines = []
                continue

            if current is None:
                continue

            if in_reqs:
                if stripped.startswith("}"):
                    current["reqs"] = _parse_req_block(req_lines)
                    in_reqs = False
                    req_lines = []
                else:
                    req_lines.append(stripped)
                continue

            if stripped.startswith("reqs"):
                flush_key()
                in_reqs = True
                req_lines = []
                if "{" in stripped:
                    tail = stripped.split("{", 1)[1].strip()
                    if tail:
                        req_lines.append(tail)
                continue

            if "=" in stripped:
                flush_key()
                key, value = stripped.split("=", 1)
                current_key = key.strip()
                current_value = [value.strip()]
            elif current_key is not None:
                current_value.append(stripped)

    flush_key()
    if current is not None:
        sections.append(current)
    return sections


def _reqs_by_type(reqs: Iterable[Dict[str, str]], req_type: str) -> List[str]:
    out: List[str] = []
    for req in reqs:
        if req.get("type") != req_type:
            continue
        present = req.get("present")
        if present is not None and present.upper() not in {"TRUE", "1", "YES"}:
            continue
        name = req.get("name")
        if name:
            out.append(_normalize_name(name))
    return out


def load_civ2civ3_techs() -> Tuple[Tuple[str, ...], Dict[str, List[str]]]:
    tech_path = _ruleset_dir() / "techs.ruleset"
    sections = _parse_ruleset_sections(tech_path, "advance_")
    techs: List[str] = []
    prereqs: Dict[str, List[str]] = {}
    for entry in sections:
        fields = entry["fields"]
        name = _normalize_name(
            _extract_first_quoted(fields.get("rule_name", fields.get("name", "")))
        )
        if not name:
            continue
        req1 = _normalize_name(_extract_first_quoted(str(fields.get("req1", ""))))
        req2 = _normalize_name(_extract_first_quoted(str(fields.get("req2", ""))))
        reqs = [req for req in (req1, req2) if req and req not in {"None", "Never"}]
        techs.append(name)
        prereqs[name] = reqs
    return tuple(techs), prereqs


def load_civ2civ3_tech_cost_factor() -> float:
    effects_path = _ruleset_dir() / "effects.ruleset"
    sections = _parse_ruleset_sections(effects_path, "effect_")
    best: Optional[float] = None
    for entry in sections:
        fields = entry.get("fields", {})
        effect_type = _extract_first_quoted(str(fields.get("type", "")))
        if effect_type != "Tech_Cost_Factor":
            continue
        reqs = entry.get("reqs", [])
        if reqs:
            continue
        try:
            value = float(fields.get("value", 0))
        except Exception:
            continue
        if best is None or value > best:
            best = value
    return best if best is not None else 1.0


def load_civ2civ3_research_config() -> Tuple[str, float, float]:
    game_path = _ruleset_dir() / "game.ruleset"
    sections = _parse_ruleset_sections(game_path, "research")
    if not sections:
        return "Linear", 10.0, 10.0
    fields = sections[0]["fields"]
    style_raw = _extract_first_quoted(str(fields.get("tech_cost_style", "Linear")))
    style = style_raw or "Linear"
    try:
        base_cost = float(fields.get("base_tech_cost", 10))
    except Exception:
        base_cost = 10.0
    try:
        min_cost = float(fields.get("min_tech_cost", base_cost))
    except Exception:
        min_cost = base_cost
    return style, base_cost, min_cost


def load_civ2civ3_units() -> Tuple[UnitRule, ...]:
    units_path = _ruleset_dir() / "units.ruleset"
    sections = _parse_ruleset_sections(units_path, "unit_")
    rules: List[UnitRule] = []
    for entry in sections:
        fields = entry["fields"]
        name = _normalize_name(
            _extract_first_quoted(fields.get("rule_name", fields.get("name", "")))
        )
        if not name:
            continue
        flags = _QUOTED_RE.findall(str(fields.get("flags", "")))
        if "NoBuild" in flags:
            continue
        try:
            cost = int(float(fields.get("build_cost", 0)))
        except Exception:
            cost = 0
        if cost <= 0:
            continue
        try:
            attack = int(float(fields.get("attack", 0)))
            defense = int(float(fields.get("defense", 0)))
            hp = int(float(fields.get("hitpoints", 1)))
            firepower = int(float(fields.get("firepower", 1)))
            moves = int(float(fields.get("move_rate", 1)))
        except Exception:
            attack, defense, hp, firepower, moves = 0, 0, 1, 1, 1
        reqs = entry.get("reqs", [])
        req_techs = _reqs_by_type(reqs, "Tech")
        obsolete_raw = _extract_first_quoted(str(fields.get("obsolete_by", "")))
        obsolete_name = _normalize_name(obsolete_raw) if obsolete_raw else ""
        obsolete_by = None
        if obsolete_name and obsolete_name not in {"None", "Never"}:
            obsolete_by = obsolete_name
        rules.append(
            UnitRule(
                name=name,
                cost=cost,
                attack=attack,
                defense=defense,
                hp=hp,
                firepower=firepower,
                moves=moves,
                req_techs=req_techs,
                flags=flags,
                obsolete_by=obsolete_by,
            )
        )
    return tuple(rules)


def load_civ2civ3_buildings() -> Tuple[BuildingRule, ...]:
    buildings_path = _ruleset_dir() / "buildings.ruleset"
    sections = _parse_ruleset_sections(buildings_path, "building_")
    rules: List[BuildingRule] = []
    for entry in sections:
        fields = entry["fields"]
        name = _normalize_name(
            _extract_first_quoted(fields.get("rule_name", fields.get("name", "")))
        )
        if not name:
            continue
        genus = _extract_first_quoted(str(fields.get("genus", "")))
        flags = _QUOTED_RE.findall(str(fields.get("flags", "")))
        if genus == "Convert" or "Gold" in flags:
            continue
        try:
            cost = int(float(fields.get("build_cost", 0)))
        except Exception:
            cost = 0
        if cost <= 0:
            continue
        reqs = entry.get("reqs", [])
        req_techs = _reqs_by_type(reqs, "Tech")
        req_buildings = _reqs_by_type(reqs, "Building")
        rules.append(
            BuildingRule(
                name=name,
                cost=cost,
                req_techs=req_techs,
                req_buildings=req_buildings,
                genus=genus,
                flags=flags,
            )
        )
    return tuple(rules)


def load_civ2civ3_ruleset() -> Civ2Civ3Ruleset:
    techs, prereqs = load_civ2civ3_techs()
    units = load_civ2civ3_units()
    buildings = load_civ2civ3_buildings()
    return Civ2Civ3Ruleset(
        techs=techs,
        tech_prereqs=prereqs,
        units=units,
        buildings=buildings,
    )


def load_civ2civ3_unlocks() -> List[Dict[str, object]]:
    ruleset = load_civ2civ3_ruleset()
    tech_index = {name: idx for idx, name in enumerate(ruleset.techs)}

    def pick_unlock_tech(reqs: List[str]) -> Optional[str]:
        if not reqs:
            return None
        ranked = sorted(reqs, key=lambda name: tech_index.get(name, -1))
        return ranked[-1] if ranked else None

    unlocks: Dict[Optional[str], List[Dict[str, object]]] = {}

    for unit in ruleset.units:
        tech = pick_unlock_tech(unit.req_techs)
        unlocks.setdefault(tech, []).append(
            {
                "kind": "unit",
                "name": unit.name,
                "attack": unit.attack,
                "defense": unit.defense,
                "hp": unit.hp,
                "firepower": unit.firepower,
                "moves": unit.moves,
                "cost": unit.cost,
            }
        )

    for building in ruleset.buildings:
        tech = pick_unlock_tech(building.req_techs)
        unlocks.setdefault(tech, []).append(
            {
                "kind": "building",
                "name": building.name,
                "cost": building.cost,
            }
        )

    ordered: List[Dict[str, object]] = []
    ordered.append({"tech": None, "unlocks": unlocks.get(None, [])})
    for tech in ruleset.techs:
        ordered.append({"tech": tech, "unlocks": unlocks.get(tech, [])})
    return ordered
