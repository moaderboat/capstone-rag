"""Scrub credentials from text before it is indexed or sent to a model.

Runs at ingest time, so secrets never land in the index file or in a prompt.
Patterns err on the side of over-redacting: a lost IP or hostname is cheap,
a leaked root token is not.
"""
import re

MASK = "[REDACTED]"

# (name, pattern, replacement). Replacements keep the label so answers still
# make sense ("the Vault root token is [REDACTED]").
_RULES = [
    ("private_key_block",
     re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
     MASK),
    ("vault_token", re.compile(r"\bhv[sbr]\.[A-Za-z0-9_\-]{20,}"), MASK),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"), MASK),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), MASK),
    ("bearer", re.compile(r"(?i)(bearer\s+)[A-Za-z0-9_\-\.=]{16,}"), r"\1" + MASK),
    # 32+ hex chars: HMAC keys, unseal key shares, API keys. Git SHAs (40 hex)
    # get caught too; acceptable.
    ("long_hex", re.compile(r"\b[0-9a-fA-F]{32,}\b"), MASK),
    # base64-ish blobs of 40+ chars (Vault unseal keys, client secrets)
    ("long_b64", re.compile(r"(?<![A-Za-z0-9/+])[A-Za-z0-9+/]{40,}={0,2}(?![A-Za-z0-9/+])"), MASK),
    # key: value / key=value where the key names a secret
    ("labeled_secret",
     re.compile(r"(?i)\b((?:\w*_)?(?:pass(?:word|wd)?|pwd|secret|token|api[_-]?key|client[_-]?secret|"
                r"unseal[_ ]?key(?:\s*\d)?|root[_ ]token|psk|passphrase)\b[\"'`]?\s*[:=]\s*)"
                r"([\"'`]?)([^\s\"'`|,;)]+)\2"),
     r"\1\2" + MASK + r"\2"),
    # `user`/`password` pairs as written in the lab notes (e.g. admin/`secret`)
    ("backtick_cred_pair",
     re.compile(r"(`[\w.\-@]{2,40}`\s*/\s*`)([^`\s]{4,})(`)"),
     r"\1" + MASK + r"\3"),
    # plain user/password pairs after the word "credentials" or "login"
    ("login_pair",
     re.compile(r"(?i)((?:login|credentials?|creds)\s*[:=]?\s*[\w.\-@]{2,40}\s*/\s*)(\S{4,})"),
     r"\1" + MASK),
]


_LABELED = {"labeled_secret": 3, "backtick_cred_pair": 2, "login_pair": 2}  # rule -> value group


def find_secrets(text: str) -> set[str]:
    """Literal secret values found next to a label anywhere in the text."""
    found = set()
    for name, pattern, _ in _RULES:
        g = _LABELED.get(name)
        if g:
            for m in pattern.finditer(text):
                v = m.group(g).strip("`'\".,")
                if len(v) >= 6 and v != MASK and not v.startswith("<") and v.lower() not in _PLACEHOLDERS:
                    found.add(v)
    return found


_PLACEHOLDERS = {"password", "username", "changeme", "redacted", "required", "enabled",
                 "disabled", "true", "false", "none", "null", "your_password", "secret"}


def redact(text: str, known_secrets: set[str] = frozenset()) -> tuple[str, dict[str, int]]:
    """Return (clean_text, {rule_name: hits}).

    known_secrets: literal values learned from labeled lines in *other* files, so a
    password written once as "Password: x" is also scrubbed where it appears bare.
    """
    counts: dict[str, int] = {}
    for name, pattern, repl in _RULES:
        text, n = pattern.subn(repl, text)
        if n:
            counts[name] = counts.get(name, 0) + n
    for s in sorted(known_secrets, key=len, reverse=True):
        n = text.count(s)
        if n:
            text = text.replace(s, MASK)
            counts["known_literal"] = counts.get("known_literal", 0) + n
    return text, counts
