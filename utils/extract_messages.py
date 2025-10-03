'''
extract_messages.py

python extract_messages.py > imessage_input.txt

'''

# Newest N true text messages from one chat:
#  - excludes reactions (tapbacks)
#  - decodes attributedBody (NSArchiver and NSKeyedArchiver)
#  - maps phone/email -> human names via NAME_MAP

import sqlite3, re
from pathlib import Path

# --- Config ---
DB_PATH = ""                     # query a copy, not the live DB
CHAT_ID = 1237                                       # chat ROWID for chat of interest
TARGET_TEXT_MESSAGES = 60000                         # how many text messages to output
PAGE_SIZE = 400                                      # raw rows fetched per page
MAX_PAGES = 2000                                     # safety guard
MY_NAME = "Milan"                                    

# Map of normalized identifiers to display names.
# Use E.164 for phone numbers (keep leading '+'), lowercase for emails.
# We'll normalize incoming handles so 'tel:+1972…' and '+1972…' both match here.
NAME_MAP = {
    # "+19876543210": "Chris",
}
# ---------------

# PyObjC
from Foundation import (
    NSData, NSAttributedString, NSMutableAttributedString,
    NSKeyedUnarchiver, NSUnarchiver,
    NSSet, NSArray, NSDictionary, NSString, NSNumber, NSData as NSNSData
)

# --- Helpers: name mapping / normalization ---

def normalize_handle(h: str) -> str:
    """Normalize a handle id (phone/email) to map keys."""
    if not h:
        return ""
    s = h.strip()
    # Strip prefixes some databases include
    if s.startswith("tel:"):
        s = s[4:]
    elif s.startswith("mailto:"):
        s = s[7:]
    s = s.strip()
    # Phone-like: keep + and digits only
    if s.startswith("+") or s.replace(" ", "").replace("-", "").isdigit():
        s = "".join(ch for ch in s if ch.isdigit() or ch == "+")
        # If it doesn't start with + but looks like digits, leave as-is; mapping can include that if desired
        return s
    # Email-like: lowercase
    return s.lower()

def display_name(sender_raw: str) -> str:
    if sender_raw == "Me":
        return MY_NAME
    key = normalize_handle(sender_raw or "")
    return NAME_MAP.get(key, sender_raw or "Unknown")

# --- Helpers: reaction filtering ---

# Textual “safety net” in case any reaction slips through with assoc_type=0
REACTION_TEXT_RE = re.compile(
    r"^(Liked|Loved|Disliked|Laughed at|Emphasized|Questioned|Removed a)\b",
    re.IGNORECASE
)

def looks_like_reaction(text: str) -> bool:
    return bool(text) and bool(REACTION_TEXT_RE.match(text.strip()))

# --- Decoding attributedBody ---

def _gather_strings(obj, out):
    """Collect human strings from bridged Foundation containers."""
    if isinstance(obj, (str, NSString)):
        s = str(obj).strip()
        if s:
            out.append(s)
    elif isinstance(obj, (NSAttributedString, NSMutableAttributedString)):
        s = obj.string()
        if s:
            s = str(s).strip()
            if s:
                out.append(s)
    elif isinstance(obj, (list, tuple, set, NSArray)):
        for x in obj:
            _gather_strings(x, out)
    elif isinstance(obj, (dict, NSDictionary)):
        for k, v in obj.items():
            _gather_strings(k, out)
            _gather_strings(v, out)

def _decode_nonkeyed(nsdata):
    """Decode NSArchiver payloads (non-keyed) via NSUnarchiver."""
    try:
        obj = NSUnarchiver.unarchiveObjectWithData_(nsdata)
        return obj
    except Exception:
        try:
            ua = NSUnarchiver.alloc().initForReadingWithData_(nsdata)
            obj = ua.decodeObject()
            return obj
        except Exception:
            return None

def _decode_keyed_secure(nsdata):
    """Decode NSKeyedArchiver payloads via secure unarchiving with a broad allow-list."""
    obj = None
    if hasattr(NSKeyedUnarchiver, "unarchivedObjectOfClasses_fromData_error_"):
        allowed = NSSet.setWithArray_([
            NSAttributedString, NSMutableAttributedString,
            NSString, NSNumber, NSArray, NSDictionary, NSNSData
        ])
        obj, err = NSKeyedUnarchiver.unarchivedObjectOfClasses_fromData_error_(allowed, nsdata, None)
    return obj

def _decode_keyed_legacy(nsdata):
    """Legacy keyed decode."""
    try:
        return NSKeyedUnarchiver.unarchiveObjectWithData_(nsdata)
    except Exception:
        return None

def decode_attributed_body(blob: bytes) -> str:
    """Best-effort plain text from attributedBody supporting both archivers."""
    if not blob:
        return ""
    nsdata = NSData.dataWithBytes_length_(blob, len(blob))

    # Try NON-KEYED first (common on recent macOS)
    obj = _decode_nonkeyed(nsdata)
    if obj is None:
        obj = _decode_keyed_secure(nsdata)
    if obj is None:
        obj = _decode_keyed_legacy(nsdata)
    if obj is None:
        return ""

    strings = []
    _gather_strings(obj, strings)
    if not strings:
        return ""

    # Dedup preserve order; prefer last snippet; fallback to longest
    seen, ordered = set(), []
    for s in strings:
        if s not in seen:
            seen.add(s); ordered.append(s)
    body = ordered[-1].replace("\u2028", "\n").strip()
    if len(body) < 2:
        ordered.sort(key=len, reverse=True)
        body = ordered[0].strip()
    return body

# --- Main: page until we have N true text messages (no reactions) ---

def main():
    if not Path(DB_PATH).exists():
        raise SystemExit(f"DB not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    collected = []  # (date, sender_display, body)
    page = 0
    offset = 0

    while len(collected) < TARGET_TEXT_MESSAGES and page < MAX_PAGES:
        # Exclude tapbacks/reactions at the SQL level (associated_message_type != 0)
        rows = cur.execute(f"""
            SELECT
                m.ROWID,
                m.date,
                CASE WHEN m.is_from_me = 1 THEN 'Me' ELSE IFNULL(h.id, '') END AS sender,
                m.text,
                m.attributedBody,
                IFNULL(m.associated_message_type, 0) AS assoc_type
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            WHERE cmj.chat_id = ?
              AND IFNULL(m.associated_message_type, 0) = 0
            ORDER BY m.date DESC
            LIMIT {PAGE_SIZE} OFFSET {offset}
        """, (CHAT_ID,)).fetchall()

        if not rows:
            break

        for _, msg_date, sender_raw, plain_text, attr_blob, _ in rows:
            # Prefer plain text; else decode attributedBody
            body = (plain_text or "").strip()
            if not body and attr_blob:
                body = decode_attributed_body(attr_blob).strip()

            # Final filter: drop any residual reaction-looking strings
            if not body or looks_like_reaction(body):
                continue

            collected.append((msg_date, display_name(sender_raw), body))
            if len(collected) >= TARGET_TEXT_MESSAGES:
                break

        page += 1
        offset += PAGE_SIZE

    conn.close()

    if not collected:
        print("(No textual (non-reaction) messages found in the scanned range.)")
        return

    # Keep newest N; print oldest → newest
    collected = collected[:TARGET_TEXT_MESSAGES]
    for _, sender_name, body in reversed(collected):
        print(f"{sender_name}:\n{body}\n")

if __name__ == "__main__":
    main()