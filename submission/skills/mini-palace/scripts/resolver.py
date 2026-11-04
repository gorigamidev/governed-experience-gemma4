"""Lexical resolver: statement terms -> ranked files and symbols of the repo."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from indexer import Symbol

WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
INLINE_CODE_RE = re.compile(r"`([^`\n]{1,120})`")
URL_RE = re.compile(r"https?://\S+")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

STOPWORDS = set("""
a an the and or but if then else when while for to of in on at by with from into onto over under as is are was were
be been being it its this that these those there here we you i he she they them our your my me us do does did done
not no yes can could should would will may might must shall so than too very just also only more most less some any
all each every both either neither such same other own which who whom whose what where why how about after before
again against between through during above below up down out off once further few nor fix fixes fixed fixing add adds
added adding update updates updated remove removes removed use uses used using make makes made support supports
allow allows new now issue issues pr pull request github com https http www please thanks thank see also e g ie etc
closes close resolves resolve related discussion description summary example examples case cases work works working
way want wants need needs like get gets set sets value values thing things one two when able instead currently
""".split())


def split_identifier(word: str) -> list[str]:
    out = []
    for piece in word.split("_"):
        out.extend(p.lower() for p in CAMEL_RE.findall(piece))
    return out


def normalize(token: str) -> str:
    """Light stemming applied identically to query and code tokens (escaping/escaped/escape -> escap)."""
    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
    elif len(token) > 4 and token.endswith("ed"):
        token = token[:-2]
    elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    if len(token) > 4 and token.endswith("e"):
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    tokens = []
    for word in WORD_RE.findall(text):
        for t in split_identifier(word):
            if len(t) >= 2 and t not in STOPWORDS and not t.isdigit():
                tokens.append(normalize(t))
    return tokens


def clean_query(text: str) -> str:
    return URL_RE.sub(" ", HTML_COMMENT_RE.sub(" ", text))


def query_identifiers(text: str) -> set[str]:
    """Exact code names the statement mentions (inline code and snake/camel/dotted words)."""
    names = set()
    for code in INLINE_CODE_RE.findall(text):
        names.update(w for w in WORD_RE.findall(code) if len(w) >= 3)
    for w in WORD_RE.findall(text):
        if "_" in w.strip("_") or re.search(r"[a-z][A-Z]", w):
            names.add(w)
    return names


@dataclass
class FileScore:
    file: str
    score: float = 0.0
    parts: dict = field(default_factory=dict)
    top_symbols: list = field(default_factory=list)


class LexicalIndex:
    def __init__(self, files: list[str], symbols: list[Symbol]):
        self.files = files
        self.symbols = symbols
        self.sym_tokens = [set(map(normalize, split_identifier(s.qualname.rsplit(".", 1)[-1]))) for s in symbols]
        self.path_tokens = {f: set(tokenize(f.removesuffix(".py").replace("/", " "))) for f in files}
        self.by_name = defaultdict(list)
        for i, s in enumerate(symbols):
            self.by_name[s.qualname.rsplit(".", 1)[-1]].append(i)
        df = Counter(t for toks in self.sym_tokens for t in toks)
        df.update(t for toks in self.path_tokens.values() for t in toks)
        n = len(symbols) + len(files) + 1
        self.idf = {t: math.log(n / (1 + c)) for t, c in df.items()}

    def rank_files(self, text: str, k: int | None = None, boosts: dict[str, float] | None = None,
                   symbol_boosts: dict[str, dict[str, float]] | None = None) -> list[FileScore]:
        """Rank files by lexical match plus boosts from active consolidated habits (see patterns.py)."""
        boosts = boosts or {}
        symbol_boosts = symbol_boosts or {}
        text = clean_query(text)
        q = Counter(tokenize(text))
        # Title terms get double weight: in ~38% of tasks the title is the only real signal.
        q.update(tokenize(text.strip().splitlines()[0] if text.strip() else ""))
        idents = query_identifiers(text)

        sym_best: dict[str, dict[str, float]] = defaultdict(dict)
        for i, toks in enumerate(self.sym_tokens):
            common = toks & q.keys()
            if not common:
                continue
            s = sum(self.idf.get(t, 0) * (1 + math.log(q[t])) for t in common) / math.sqrt(len(toks))
            sym = self.symbols[i]
            sym_best[sym.file][sym.qualname] = max(s, sym_best[sym.file].get(sym.qualname, 0.0))

        exact_files = Counter()
        for name in idents:
            for i in self.by_name.get(name, ()):
                exact_files[self.symbols[i].file] += 1

        results = []
        for f in self.files:
            lexical = sorted(sym_best.get(f, {}).values(), reverse=True)
            sym_score = sum(lexical[:3]) / 3 if lexical else 0.0
            path_common = self.path_tokens[f] & q.keys()
            path_score = sum(self.idf.get(t, 0) * (1 + math.log(q[t])) for t in path_common)
            exact = math.log1p(exact_files[f]) * 3.0
            habit = boosts.get(f, 0.0)
            score = sym_score + 0.5 * path_score + exact + habit
            if score > 0:
                combined = dict(sym_best.get(f, {}))
                for qual, w in symbol_boosts.get(f, {}).items():
                    combined[qual] = combined.get(qual, 0.0) + w
                top = sorted(combined, key=lambda qn: -combined[qn])[:3]
                parts = {"sym": round(sym_score, 2), "path": round(path_score, 2), "exact": round(exact, 2),
                         "habit": round(habit, 2)}
                results.append(FileScore(f, score, parts, top))
        results.sort(key=lambda r: -r.score)
        return results[:k] if k else results
