"""A bounded, typed evidence graph over catalogue metadata and prior reviews.

Device extraction records title claims, not verified compatibility. Review edges
are observations, not purchases. Candidate/model overlays require separate inputs.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

PARSER_VERSION = "phone-title-v1"
_DEVICES = re.compile(
    r"\b(?:apple\s+)?iphone\s*(?:[4-9]s?|1\d)(?:\s*(?:pro\s*max|plus|pro|mini))?\b"
    r"|\b(?:apple\s+)?iphone\s*(?:xs\s*max|xr|xs|x|se(?:\s*\(?20(?:16|20|22)\)?)?)\b"
    r"|\b(?:samsung\s+)?galaxy\s*(?:s\d{1,2}|note\s*\d{1,2}|a\d{1,2}s?)(?:\s*(?:ultra|plus|\+|fe))?(?:\s*[45]g)?(?![a-z0-9])"
    r"|\b(?:google\s+)?pixel\s*\d{1,2}a?(?:\s*(?:pro(?:\s*xl)?|xl|fold))?(?:\s*[45]g)?\b",
    re.IGNORECASE,
)
_UNIVERSAL = re.compile(r"\b(?:universal|all\s+(?:phones|smartphones)|any\s+(?:phone|smartphone))\b", re.I)
_NEGATIVE = re.compile(r"\b(?:not\s+(?:for|compatible\s+with|fit(?:s)?|support(?:s)?)|does\s+not\s+fit|except(?:\s+for)?|excluding)\b", re.I)


def _canonical_device(value: str) -> str:
    value = re.sub(r"\s+", " ", value.lower().strip())
    value = re.sub(r"^(?:apple|samsung|google)\s+", "", value)
    value = re.sub(r"(iphone|pixel|galaxy\s+note)(?=\d)", r"\1 ", value)
    value = re.sub(r"galaxy\s*(?=[sa]\d)", "galaxy ", value)
    value = value.replace("+", " plus")
    value = re.sub(r"(\d)(pro|max|plus|mini|ultra|fe)", r"\1 \2", value)
    value = value.replace("promax", "pro max")
    value = re.sub(r"(ultra|plus|fe)([45]g)\b", r"\1 \2", value)
    value = re.sub(r"\s+", " ", value)
    return value


def parse_devices(title: str) -> dict:
    """Extract explicit common-phone mentions, retaining negation and spans.

    Shorthand such as 'iPhone 12/13' is left ambiguous rather than expanded.
    Universal titles never establish a known device relation.
    """
    title = str(title or "")
    mentions = []
    ambiguous = False
    for match in _DEVICES.finditer(title):
        # An unqualified shorthand continuation may change the variant suffix.
        if re.match(r"\s*[/&]\s*\d", title[match.end():]):
            ambiguous = True
            continue
        prefix = re.split(r"[;.!]|\bbut\b|\bhowever\b", title[:match.start()], flags=re.I)[-1][-70:]
        neg = list(_NEGATIVE.finditer(prefix))
        excluded = bool(neg)
        if neg:
            # A later explicit positive clause ends a preceding exclusion.
            after = prefix[neg[-1].end():]
            excluded = not bool(re.search(r"\b(?:compatible\s+with|fits|designed\s+for)\b", after, re.I))
        mentions.append({"device": _canonical_device(match.group()), "start": match.start(),
                         "end": match.end(), "text": match.group(), "excluded": excluded})
    universal = bool(_UNIVERSAL.search(title))
    positive = sorted({m["device"] for m in mentions if not m["excluded"]})
    negative = sorted({m["device"] for m in mentions if m["excluded"]})
    status = "universal" if universal else "ambiguous" if ambiguous else "parsed" if positive else "negated" if negative else "unknown"
    return {"status": status, "positive": positive, "negative": negative, "mentions": mentions}


def device_relation(left: dict, right: dict) -> str:
    """Known title-set overlap/disjointness; not a compatibility assertion."""
    if left["status"] != "parsed" or right["status"] != "parsed":
        return "unknown"
    a, b = set(left["positive"]), set(right["positive"])
    return "match" if a & b else "mismatch" if a and b else "unknown"


def known_equal(left: str | None, right: str | None) -> bool:
    """Missing catalogue values do not constitute a shared relation."""
    return bool(pd.notna(left) and pd.notna(right) and str(left).strip() and str(right).strip() and left == right)


def co_review_witnesses(reviews: pd.DataFrame, history_items: list[int], candidate: int,
                        current_user: int, cutoff: int) -> pd.DataFrame:
    """Return source-backed two-hop paths, excluding the query user's links."""
    prior = reviews[(reviews["timestamp"] < cutoff) & (reviews["user"] != current_user)]
    h = prior[prior["item"].isin(set(history_items) - {candidate})]
    c = prior[prior["item"] == candidate]
    return h[["user", "item", "timestamp"]].rename(columns={"item": "history_item", "timestamp": "history_ts"}).merge(
        c[["user", "item", "timestamp"]].rename(columns={"item": "candidate", "timestamp": "candidate_ts"}), on="user"
    ).sort_values(["user", "history_item", "history_ts", "candidate_ts"]).reset_index(drop=True)


def make_audit(items: pd.DataFrame, parsed: list[dict], t1: int, seed: int = 0) -> pd.DataFrame:
    """Select 200 distinct titles; labels remain blank for human review."""
    table = items[["item", "parent_asin", "title", "first_ts"]].copy()
    table["new_at_t1"] = table["first_ts"] >= t1
    table["parser_status"] = [p["status"] for p in parsed]
    table["parsed_positive_devices"] = [json.dumps(p["positive"]) for p in parsed]
    table["parsed_excluded_devices"] = [json.dumps(p["negative"]) for p in parsed]
    rng = np.random.default_rng(seed)
    selected = []
    groups = list(table.groupby(["new_at_t1", "parser_status"], sort=True))
    budget = min(200, len(table))
    quota = max(1, budget // max(1, len(groups)))
    for _, group in groups:
        ids = group.sort_values("item").index.to_numpy()
        selected.extend(rng.choice(ids, min(quota, len(ids)), replace=False).tolist())
    remainder = table.index.difference(selected).to_numpy()
    selected.extend(rng.choice(remainder, budget - len(selected), replace=False).tolist())
    selected = rng.permutation(selected)
    audit = table.loc[selected].reset_index(drop=True)
    audit.insert(0, "audit_split", ["development" if i < min(100, len(audit) // 2) else "locked" for i in range(len(audit))])
    for column in ("reviewed", "expected_positive_devices", "expected_excluded_devices", "expected_status", "reviewer_notes"):
        audit[column] = ""
    return audit


def assess_device_audit(audit: pd.DataFrame, minimum_precision: float = 0.95, required_locked: int = 100) -> dict:
    """Gate device-profile claims on completed locked manual labels.

    Precision is edge-level against human positive-device labels. Its interval
    and support remain necessary; a point threshold does not certify all devices.
    """
    locked = audit[audit["audit_split"] == "locked"]
    done = locked["reviewed"].fillna("").astype(str).str.lower().isin(["yes", "true", "1"])
    result = {"status": "blocked", "minimum_precision": minimum_precision,
              "required_locked_titles": required_locked, "reviewed_locked_titles": int(done.sum()),
              "reason": "Locked human device audit is incomplete; no quantitative device claims."}
    if len(locked) < required_locked or not done.all():
        return result
    true_positive = predicted = expected = 0
    for _, row in locked.iterrows():
        p = set(json.loads(row["parsed_positive_devices"]))
        label = json.loads(row["expected_positive_devices"])
        if not isinstance(label, list) or not all(isinstance(x, str) for x in label):
            raise ValueError("expected_positive_devices must be a JSON list of canonical strings")
        e = set(label)
        true_positive += len(p & e)
        predicted += len(p)
        expected += len(e)
    if not predicted:
        result["reason"] = "No positive device edges were audited; precision is undefined."
        return result
    precision = true_positive / predicted
    z = 1.959963984540054
    centre = (precision + z*z/(2*predicted)) / (1 + z*z/predicted)
    half = z * math.sqrt(precision*(1-precision)/predicted + z*z/(4*predicted*predicted)) / (1 + z*z/predicted)
    result.update(precision=precision, precision_wilson95=[centre-half, centre+half],
                  recall=true_positive/expected if expected else None, positive_edges=predicted,
                  status="passed" if precision >= minimum_precision else "blocked",
                  reason="Locked positive-edge precision threshold met; report scope/support/interval." if precision >= minimum_precision else "Locked precision below threshold.")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_ego_html(item: pd.Series, parsed: dict) -> str:
    """A bounded metadata-only ego view; all labels point to source fields."""
    title = str(item["title"])
    neighbours = [("leaf category", str(item.get("leaf_cat", ""))), ("store", str(item.get("store", "")))]
    neighbours += [("excluded title mention" if m["excluded"] else "title device mention", m["device"]) for m in parsed["mentions"]]
    neighbours = [(r, v) for r, v in neighbours if v][:7]
    height = max(250, 75 * len(neighbours) + 65)
    svg = [f'<svg viewBox="0 0 1000 {height}" role="img" aria-label="Product metadata evidence graph">',
           f'<rect x="20" y="{height/2-30}" width="180" height="60" rx="8" fill="#ddeaff"/>',
           f'<text x="30" y="{height/2}">item {int(item["item"])}</text>']
    for i, (relation, value) in enumerate(neighbours):
        y = 45 + i*75
        svg += [f'<line x1="200" y1="{height/2}" x2="500" y2="{y+15}" stroke="#888"/>',
                f'<text x="280" y="{y+5}" font-size="12">{html.escape(relation)}</text>',
                f'<rect x="500" y="{y-8}" width="480" height="46" rx="6" fill="#eef4ed"/>',
                f'<text x="510" y="{y+18}" font-size="12">{html.escape(value[:70])}</text>']
    svg.append('</svg>')
    return '<!doctype html><html><meta charset="utf-8"><title>Metadata evidence</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px}svg{width:100%}li{margin:8px 0}</style><h1>Metadata evidence</h1>' + f'<p><b>{html.escape(str(item["parent_asin"]))}</b>: {html.escape(title)}</p><p>Parser status: {parsed["status"]}. Unverified title claims; no model outcome or causal attribution. Catalogue metadata is a crawl-time snapshot.</p>' + ''.join(svg) + '<h2>Source evidence</h2><ul>' + ''.join(f'<li>{html.escape(m["device"])}: title[{m["start"]}:{m["end"]}] = {html.escape(m["text"])}; excluded={m["excluded"]}</li>' for m in parsed["mentions"]) + '</ul></html>'
