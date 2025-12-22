from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import partial
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit

import pandas as pd


# ----------------------------
# Canon + name map (updated)
# ----------------------------

_CO_SUFFIX_RE = re.compile(
    r"""
    (?:
        ,?\s+Inc\.? |
        ,?\s+Incorporated |
        ,?\s+Corp\.? |
        ,?\s+Corporation |
        ,?\s+Co\.? |
        ,?\s+Company |
        ,?\s+Ltd\.? |
        ,?\s+Limited |
        ,?\s+PLC |
        ,?\s+p\.l\.c\. |
        ,?\s+S\.A\. |
        ,?\s+AG |
        ,?\s+N\.V\. |
        ,?\s+B\.V\. |
        ,?\s+AB |
        ,?\s+ASA |
        ,?\s+SE |
        ,?\s+GmbH |
        ,?\s+S\.r\.l\. |
        ,?\s+S\.p\.A\. |
        ,?\s+S\.A\.S\. |
        ,?\s+Pte\.?\s+Ltd\.? |
        ,?\s+Holdings? |
        ,?\s+Group
    )\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

_CLASS_SUFFIX_RE = re.compile(r"\s+Class\s+[A-Z]\b.*$", re.IGNORECASE)
_COMMON_STOCK_RE = re.compile(r"\s+Common Stock.*$", re.IGNORECASE)
_ORD_SHARES_RE = re.compile(r"\s+Ordinary Shares.*$", re.IGNORECASE)
_DEPOSITARY_RE = re.compile(r"\s+Depositary Shares.*$", re.IGNORECASE)
_SERIES_RE = re.compile(r"\s+Series\s+[A-Z0-9]+\b.*$", re.IGNORECASE)

# Remove common “trading-name noise”
_NOISE_TAIL_RE = re.compile(
    r"\s+(?:ADS|ADR|A\.D\.R\.|American Depositary Shares)\b.*$",
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _canon_security_name(name: str | None) -> str:
    if name is None or not isinstance(name, str):
        return ""
    n = _norm(name)

    # Prefer “base name” before dash qualifiers
    n = n.split(" - ")[0].strip()

    # Remove common security-type tails
    n = _COMMON_STOCK_RE.sub("", n)
    n = _ORD_SHARES_RE.sub("", n)
    n = _DEPOSITARY_RE.sub("", n)
    n = _NOISE_TAIL_RE.sub("", n)

    # Remove class/series tails
    n = _CLASS_SUFFIX_RE.sub("", n)
    n = _SERIES_RE.sub("", n)

    return n.strip()


def _dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        x = _norm(x)
        if not x:
            continue
        k = x.casefold()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def _strip_corp_suffix(name: str) -> str:
    n = _norm(name)
    # Iteratively strip stacked suffixes ("Holdings Inc." -> "Holdings" then maybe "…")
    prev = None
    while prev != n:
        prev = n
        n = _CO_SUFFIX_RE.sub("", n).strip()
    return n


def build_name_map(universe_df: pd.DataFrame) -> dict[str, list[str]]:
    m: dict[str, list[str]] = {}
    df = universe_df[["symbol", "Security Name"]].dropna()

    for sym, sec in df.itertuples(index=False):
        sym = str(sym).strip().upper()
        base = _canon_security_name(sec)

        aliases: list[str] = [sym]
        if base:
            aliases.append(base)

            base2 = _strip_corp_suffix(base)
            if base2 and base2.casefold() != base.casefold():
                aliases.append(base2)

            # Also try stripping suffix from already stripped versions of dash-removed names etc.
            base3 = _strip_corp_suffix(_norm(sec).split(" - ")[0])
            if base3 and base3.casefold() not in {a.casefold() for a in aliases}:
                aliases.append(base3)

        # Filter, dedupe, and sort by length desc for safe replacement ordering
        aliases = [a for a in _dedupe_keep_order(aliases) if len(a) >= 2]
        aliases.sort(key=len, reverse=True)

        m[sym] = aliases

    return m


# ----------------------------
# Anonymizer (drop-in)
# ----------------------------

URL_RE = re.compile(r"(https?://[^\s)\]}>\"',]+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"__(?:TARGET|OTHER)__")
URL_PH_RE = re.compile(r"__URL_(\d+)__")
EMAIL_PH_RE = re.compile(r"__EMAIL_(\d+)__")

EXCH_TICKER_RE = re.compile(
    r"\(\s*(?:NYSE|NASDAQ|AMEX|OTC|OTCQB|TSX|TSXV|LSE|XETRA|FRA|ASX|HKEX|TSE|CSE|BIT|BSE|NSE)\s*:\s*([A-Z][A-Z0-9.\-]{0,9})\s*\)",
    re.IGNORECASE,
)
BARE_TICKER_CTX_RE = re.compile(
    r"(?:(?:\bTicker\b|\bSymbol\b|\bshares?\b|\bstock\b|\bcommon stock\b)\s*[:\-]?\s*|\$)([A-Z][A-Z0-9.\-]{0,9})\b",
    re.IGNORECASE,
)


def _boundary_pat(lit: str) -> str:
    # Match word boundaries, also capturing optional trailing possessive 's/'
    return rf"(?<![\w]){re.escape(lit)}(?:'s?)?(?![\w])"


def _is_placeholder_chunk(s: str) -> bool:
    return bool(URL_PH_RE.search(s) or EMAIL_PH_RE.search(s))


def _rewrite_url(url: str, alias_rules: list[tuple[str, str]]) -> str:
    try:
        parts = urlsplit(url)
    except Exception:
        return url

    def sub_tokens(piece: str) -> str:
        if not piece or TOKEN_RE.search(piece):
            return piece
        low = piece.lower()
        for alias_l, rep in alias_rules:
            if alias_l and alias_l in low:
                piece = re.sub(re.escape(alias_l), rep, piece, flags=re.IGNORECASE)
                low = piece.lower()
        return piece

    return urlunsplit((parts.scheme, sub_tokens(parts.netloc), sub_tokens(parts.path), sub_tokens(parts.query), parts.fragment))


@dataclass(frozen=True)
class AnonymizeConfig:
    target_symbol: str
    name_map: dict[str, list[str]]
    never_redact: tuple[str, ...] = ()
    protect_urls: bool = True
    protect_emails: bool = True
    rewrite_urls: bool = True
    # Minimum alias length for context-free matching (short symbols need context)
    min_contextfree_len: int = 3


def _restore_placeholders(text: str, urls: list[str], emails: list[str]) -> str:
    if urls:
        for i, u in enumerate(urls):
            text = text.replace(f"__URL_{i}__", u)
    if emails:
        for i, e in enumerate(emails):
            text = text.replace(f"__EMAIL_{i}__", e)
    return text


def _make_stash_url(
    urls: list[str], rewrite: bool, alias_rules: list[tuple[str, str]]
) -> callable:
    """Factory to avoid closure-over-loop-variable issues."""
    def stash_url(m: re.Match) -> str:
        u = m.group(1)
        if rewrite:
            u = _rewrite_url(u, alias_rules)
        idx = len(urls)
        urls.append(u)
        return f"__URL_{idx}__"
    return stash_url


def _make_stash_email(emails: list[str]) -> callable:
    """Factory to avoid closure-over-loop-variable issues."""
    def stash_email(m: re.Match) -> str:
        e = m.group(0)
        idx = len(emails)
        emails.append(e)
        return f"__EMAIL_{idx}__"
    return stash_email


def _make_token_sub(token: str) -> callable:
    """Factory for token substitution to avoid closure issues."""
    def _sub(m: re.Match) -> str:
        s = m.group(0)
        if TOKEN_RE.search(s) or _is_placeholder_chunk(s):
            return s
        return token
    return _sub


def _build_alternation_pattern(
    aliases: list[str], never: set[str], min_len: int
) -> str | None:
    """Build a single alternation pattern from aliases, filtering short ones."""
    parts = []
    for a in aliases:
        a = _norm(a)
        if not a or a.casefold() in never or len(a) < min_len:
            continue
        parts.append(_boundary_pat(a))
    return "|".join(parts) if parts else None


class Anonymizer:
    """Pre-compiled anonymizer for batch processing. Build once, call many times."""

    def __init__(
        self,
        name_map: dict[str, list[str]],
        never_redact: tuple[str, ...] = (),
        protect_urls: bool = True,
        protect_emails: bool = True,
        rewrite_urls: bool = True,
        min_contextfree_len: int = 3,
    ):
        self.protect_urls = protect_urls
        self.protect_emails = protect_emails
        self.rewrite_urls = rewrite_urls

        never = {x.casefold() for x in never_redact}

        # Clean and normalize the name map once
        self.clean_map: dict[str, list[str]] = {}
        for sym, aliases in (name_map or {}).items():
            sym_u = sym.upper().strip()
            ali = _dedupe_keep_order([sym_u, *(aliases or [])])
            ali = [a for a in ali if a and len(a) >= 2 and a.casefold() not in never and not TOKEN_RE.fullmatch(a)]
            ali.sort(key=len, reverse=True)
            self.clean_map[sym_u] = ali

        self.sym_set = set(self.clean_map.keys())

        # Pre-build URL rewrite rules for all "other" companies (target-agnostic part)
        self.other_url_rules: list[tuple[str, str]] = [
            (a.casefold(), "__OTHER__")
            for sym, aliases in self.clean_map.items()
            for a in aliases
        ]

        # Pre-compile the "other" pattern (same for all targets)
        all_other_aliases = [a for aliases in self.clean_map.values() for a in aliases]
        other_pattern = _build_alternation_pattern(all_other_aliases, never, min_contextfree_len)
        self.other_re = re.compile(other_pattern, re.IGNORECASE) if other_pattern else None

        # Cache for per-target compiled patterns
        self._target_re_cache: dict[str, re.Pattern | None] = {}
        self.never = never
        self.min_contextfree_len = min_contextfree_len

    def _get_target_re(self, target_sym: str) -> re.Pattern | None:
        if target_sym not in self._target_re_cache:
            tgt_aliases = self.clean_map.get(target_sym, [target_sym])
            pat = _build_alternation_pattern(tgt_aliases, self.never, self.min_contextfree_len)
            self._target_re_cache[target_sym] = re.compile(pat, re.IGNORECASE) if pat else None
        return self._target_re_cache[target_sym]

    def __call__(self, text: str, target_symbol: str) -> str:
        if not isinstance(text, str) or not text:
            return text

        text = _norm(text)
        target_sym = target_symbol.upper().strip()
        tgt_aliases = self.clean_map.get(target_sym, [target_sym])

        # URL rewrite rules: target first, then others
        alias_rules_url = [(a.casefold(), "__TARGET__") for a in tgt_aliases] + self.other_url_rules

        # Protect URLs/emails
        urls: list[str] = []
        emails: list[str] = []

        if self.protect_urls:
            text = URL_RE.sub(_make_stash_url(urls, self.rewrite_urls, alias_rules_url), text)

        if self.protect_emails:
            text = EMAIL_RE.sub(_make_stash_email(emails), text)

        # Replace tickers in common contexts
        def repl_exch(m: re.Match) -> str:
            sym = m.group(1).upper()
            if sym not in self.sym_set:
                return m.group(0)
            tok = "__TARGET__" if sym == target_sym else "__OTHER__"
            return re.sub(re.escape(m.group(1)), tok, m.group(0), flags=re.IGNORECASE)

        text = EXCH_TICKER_RE.sub(repl_exch, text)

        def repl_ctx(m: re.Match) -> str:
            sym = m.group(1).upper()
            if sym not in self.sym_set:
                return m.group(0)
            tok = "__TARGET__" if sym == target_sym else "__OTHER__"
            return m.group(0).replace(m.group(1), tok)

        text = BARE_TICKER_CTX_RE.sub(repl_ctx, text)

        # Apply target pattern first
        target_re = self._get_target_re(target_sym)
        if target_re:
            text = target_re.sub(_make_token_sub("__TARGET__"), text)

        # Apply other pattern (replaces target matches too, but target already done)
        if self.other_re:
            text = self.other_re.sub(_make_token_sub("__OTHER__"), text)

        # De-dup weird repeats
        text = re.sub(r"(__TARGET__)\s*(?:\1)+", r"\1", text)
        text = re.sub(r"(__OTHER__)\s*(?:\1)+", r"\1", text)

        return _restore_placeholders(text, urls, emails)


def anonymize_text(text: str, cfg: AnonymizeConfig) -> str:
    """Convenience wrapper - creates Anonymizer per call. Use Anonymizer class for batch."""
    anon = Anonymizer(
        name_map=cfg.name_map,
        never_redact=cfg.never_redact,
        protect_urls=cfg.protect_urls,
        protect_emails=cfg.protect_emails,
        rewrite_urls=cfg.rewrite_urls,
        min_contextfree_len=cfg.min_contextfree_len,
    )
    return anon(text, cfg.target_symbol)


# ----------------------------
# Minimal smoke test
# ----------------------------
if __name__ == "__main__":
    universe = pd.DataFrame(
        {
            "symbol": ["EH", "MSFT", "ABCL"],
            "Security Name": ["eHealth, Inc. Common Stock", "Microsoft Corporation", "Applied Blockchain, Inc."],
        }
    )
    name_map = build_name_map(universe)

    cfg = AnonymizeConfig(target_symbol="EH", name_map=name_map, never_redact=("OpenAI",))
    blob = """eHealth, Inc. (NASDAQ: EH) partnered with Microsoft Corporation.
    URL https://ir.ehealthinsurance.com/press?x=eHealth Email foo@bar.com
    """
    print(anonymize_text(blob, cfg))