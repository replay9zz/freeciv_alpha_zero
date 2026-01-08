from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .ruleset_loader import (
    load_civ2civ3_research_config,
    load_civ2civ3_tech_cost_factor,
    load_civ2civ3_techs,
)

# Core tech list for civ2civ3 ruleset (order defines action indices).
RESEARCH_TECHS, TECH_PREREQS = load_civ2civ3_techs()
TECH_COST_STYLE, BASE_TECH_COST, MIN_TECH_COST = load_civ2civ3_research_config()
TECH_COST_FACTOR = load_civ2civ3_tech_cost_factor()

# The primary milestone tech to mark research completion.
TARGET_TECH_NAME: str = "Iron Working"

# Preferred research sequence for deterministic scenarios.
# Note: "Howitzer" is a unit; its unlock tech is "Robotics".
RESEARCH_PRIORITY_CHAIN: Tuple[str, ...] = (
    "Warrior Code",
    "Bronze Working",
    "Monarchy",
    "Gunpowder",
    "Democracy",
    "Robotics",
)


def tech_index_map() -> dict[str, int]:
    """Helper to map tech name to its index in RESEARCH_TECHS."""
    return {name: idx for idx, name in enumerate(RESEARCH_TECHS)}


def pick_next_goal_tech(
    goal: str,
    flags: Dict[str, bool],
    prereqs: Optional[Dict[str, List[str]]] = None,
) -> Optional[str]:
    """
    Pick the next tech on the path to goal, respecting explicit prereqs.
    """
    idx_map = tech_index_map()
    if goal not in idx_map:
        return None
    if flags.get(goal, False):
        return goal
    prereq_map = prereqs or TECH_PREREQS
    seen: set[str] = set()

    def walk(tech: str) -> Optional[str]:
        if tech in seen:
            return None
        seen.add(tech)
        if flags.get(tech, False):
            return None
        reqs = prereq_map.get(tech, [])
        for req in reqs:
            if flags.get(req, False):
                continue
            candidate = walk(req)
            if candidate:
                return candidate
        return tech

    return walk(goal)


def pick_next_priority_tech(
    flags: Dict[str, bool],
    priority_chain: Optional[Sequence[str]] = None,
    prereqs: Optional[Dict[str, List[str]]] = None,
) -> Optional[str]:
    """
    Pick the next tech from the priority chain (first missing entry).
    """
    chain = priority_chain or RESEARCH_PRIORITY_CHAIN
    idx_map = tech_index_map()
    for tech in chain:
        if tech not in idx_map:
            continue
        if flags.get(tech, False):
            continue
        return pick_next_goal_tech(tech, flags, prereqs=prereqs)
    return None


# Optional per-node inheritance ratios for the implicit binary tree.
# Key: parent tech name -> (left_ratio, right_ratio)
TECH_INHERITANCE: Dict[str, Tuple[float, float]] = {}
# Optional per-child ratios keyed by (parent -> child_name -> ratio).
# This is safer when req1/req2 order might swap in the UI but names stay stable.
TECH_CHILD_INHERITANCE: Dict[str, Dict[str, float]] = {}
# TECH_PREREQS is loaded from civ2civ3 techs.ruleset above.


def _collect_prereqs(
    tech: str,
    prereqs: Dict[str, List[str]],
    memo: Dict[str, set[str]],
) -> set[str]:
    if tech in memo:
        return memo[tech]
    out: set[str] = set()
    for req in prereqs.get(tech, []):
        out.add(req)
        out.update(_collect_prereqs(req, prereqs, memo))
    memo[tech] = out
    return out


def _tech_cost(num_reqs: int, style: str, base_cost: float) -> float:
    label = (style or "").strip().lower()
    normalized = ''.join(ch for ch in label if ch.isalnum())
    if normalized in {"civ1civ2", "civiii"}:
        return base_cost
    reqs = max(1, num_reqs)
    if label == "linear":
        return base_cost * reqs
    if label in {"classic", "classic+"}:
        return base_cost * (1.0 + reqs) * math.sqrt(1.0 + reqs) / 2.0
    if label in {"experimental", "experimental+"}:
        return base_cost * (
            (reqs ** 2) / (1.0 + math.sqrt(math.sqrt(reqs + 1.0))) - 0.5
        )
    return base_cost * max(1.0, reqs)


def _round_cost(value: float) -> int:
    if value <= 0:
        return 0
    return int(math.floor(value + 0.5))


def build_tech_costs(
    prereqs: Dict[str, List[str]],
    *,
    style: str = TECH_COST_STYLE,
    base_cost: float = BASE_TECH_COST,
    min_cost: float = MIN_TECH_COST,
    cost_factor: float = TECH_COST_FACTOR,
) -> Dict[str, float]:
    memo: Dict[str, set[str]] = {}
    costs: Dict[str, float] = {}
    min_cost_val = _round_cost(min_cost)
    factor = max(0.0, float(cost_factor))
    for tech in RESEARCH_TECHS:
        reqs = _collect_prereqs(tech, prereqs, memo)
        raw = _tech_cost(len(reqs) + 1, style, base_cost)
        costs[tech] = max(min_cost_val, _round_cost(raw * factor))
    return costs


TECH_COSTS: Dict[str, float] = build_tech_costs(TECH_PREREQS)


def _child_indices(index: int) -> Tuple[Optional[int], Optional[int]]:
    left = 2 * index + 1
    right = 2 * index + 2
    return (
        left if left < len(RESEARCH_TECHS) else None,
        right if right < len(RESEARCH_TECHS) else None,
    )


def propagate_goal_scores(
    goals: Dict[str, float],
    inheritance: Optional[Dict[str, Tuple[float, float]]] = None,
    default_ratio: Tuple[float, float] = (0.5, 0.5),
) -> Dict[str, float]:
    """
    Distribute goal scores down the implicit binary tech tree.

    Example: {"Iron Working": 100} with inheritance {"Alphabet": (0.7, 0.3)}
    will push 100 to Iron Working, then pass fractions to its children, and so on.
    """
    if not goals:
        return {tech: 0.0 for tech in RESEARCH_TECHS}

    idx_map = tech_index_map()
    ratios = inheritance or TECH_INHERITANCE
    scores: Dict[str, float] = {tech: 0.0 for tech in RESEARCH_TECHS}

    for goal_name, goal_score in goals.items():
        if goal_name not in idx_map:
            raise ValueError(f"Unknown tech '{goal_name}'")
        stack: List[Tuple[str, float]] = [(goal_name, float(goal_score))]
        while stack:
            tech_name, score = stack.pop()
            if score <= 0:
                continue
            if score > scores[tech_name]:
                scores[tech_name] = score
            parent_idx = idx_map[tech_name]
            left_idx, right_idx = _child_indices(parent_idx)
            if left_idx is None and right_idx is None:
                continue
            left_ratio, right_ratio = ratios.get(tech_name, default_ratio)
            if left_idx is not None and left_ratio > 0:
                stack.append((RESEARCH_TECHS[left_idx], score * left_ratio))
            if right_idx is not None and right_ratio > 0:
                stack.append((RESEARCH_TECHS[right_idx], score * right_ratio))
    return scores


def rank_research_by_score(
    goals: Dict[str, float],
    inheritance: Optional[Dict[str, Tuple[float, float]]] = None,
    default_ratio: Tuple[float, float] = (0.5, 0.5),
) -> List[Tuple[str, float]]:
    """
    Return techs sorted by propagated score (high to low).
    """
    scores = propagate_goal_scores(goals, inheritance=inheritance, default_ratio=default_ratio)
    ranked = [(tech, score) for tech, score in scores.items() if score > 0]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def _next_nodes(
    tech_name: str,
    prereqs: Dict[str, List[str]],
) -> List[str]:
    # Use explicit prereq list if present; otherwise fall back to the implicit tree.
    if tech_name in prereqs:
        return [child for child in prereqs[tech_name] if child in tech_index_map()]
    # implicit tree fallback
    idx = tech_index_map().get(tech_name)
    if idx is None:
        return []
    left_idx, right_idx = _child_indices(idx)
    children: List[str] = []
    if left_idx is not None:
        children.append(RESEARCH_TECHS[left_idx])
    if right_idx is not None:
        children.append(RESEARCH_TECHS[right_idx])
    return children


def propagate_goal_scores_named(
    goals: Dict[str, float],
    prereqs: Optional[Dict[str, List[str]]] = None,
    child_ratios: Optional[Dict[str, Dict[str, float]]] = None,
    default_split: float = 0.5,
) -> Dict[str, float]:
    """
    Name-stable propagation that follows techs.ruleset prereqs.
    child_ratios[parent][child] overrides the split; otherwise the goal score
    is divided equally among the listed prereqs.
    """
    if not goals:
        return {tech: 0.0 for tech in RESEARCH_TECHS}

    idx_map = tech_index_map()
    prereq_map = prereqs or TECH_PREREQS
    ratio_map = child_ratios or TECH_CHILD_INHERITANCE
    scores: Dict[str, float] = {tech: 0.0 for tech in RESEARCH_TECHS}

    for goal_name, goal_score in goals.items():
        if goal_name not in idx_map:
            raise ValueError(f"Unknown tech '{goal_name}'")
        stack: List[Tuple[str, float]] = [(goal_name, float(goal_score))]
        expanded: set[str] = set()
        while stack:
            tech, score = stack.pop()
            if score <= 0:
                continue
            scores[tech] += score  # accumulate contributions from multiple parents
            # Avoid cycles: if a tech lists itself (directly or indirectly), skip revisiting.
            if tech in expanded:
                continue
            expanded.add(tech)
            children = _next_nodes(tech, prereq_map)
            if not children:
                continue
            weights = ratio_map.get(tech, {})
            total = sum(max(0.0, weights.get(child, 0.0)) for child in children)
            if total <= 0:
                split = score / len(children)
                for child in children:
                    stack.append((child, split))
            else:
                for child in children:
                    w = max(0.0, weights.get(child, 0.0))
                    if w <= 0:
                        continue
                    stack.append((child, score * (w / total)))
    return scores


def reward_map_from_goals(
    goals: Dict[str, float],
    base_reward: float,
    prereqs: Optional[Dict[str, List[str]]] = None,
    child_ratios: Optional[Dict[str, Dict[str, float]]] = None,
    default_split: float = 0.5,
    min_reward: Optional[float] = None,
    max_reward: Optional[float] = None,
) -> Dict[str, float]:
    """
    Build a per-tech reward map that boosts techs related to the given goals.
    - goals: {"Iron Working": 1.0, ...}
    - base_reward: fallback when no boost applies
    Rewards scale with propagated goal scores (normalized to max score).
    """
    scores = propagate_goal_scores_named(
        goals,
        prereqs=prereqs,
        child_ratios=child_ratios,
        default_split=default_split,
    )
    max_score = max(scores.values()) if scores else 0.0
    reward_map: Dict[str, float] = {}
    for tech, score in scores.items():
        if max_score <= 0 or score <= 0:
            reward = base_reward
        else:
            factor = score / max_score
            reward = base_reward * (1.0 + factor)
        if min_reward is not None:
            reward = max(min_reward, reward)
        if max_reward is not None:
            reward = min(max_reward, reward)
        reward_map[tech] = reward
    return reward_map
