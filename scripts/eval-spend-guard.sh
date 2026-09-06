#!/usr/bin/env bash
# scripts/eval-spend-guard.sh — hard spend cap for paid eval runs.
#
# INVARIANT: a paid command never launches when (ledger total + estimate) would
# exceed the cap, and every launch is appended to the ledger whether or not it
# succeeded — so the running total can only ever UNDER-state spend if the
# command itself lies about its cost. The guard FAILS CLOSED on anything it
# cannot account for: a missing ledger, an unparseable ledger line, a signed or
# malformed amount. No jq / bun dependency at runtime.
#
# Usage:
#   scripts/eval-spend-guard.sh <cap_usd> <estimate_usd> -- <command...>
#
# Amounts are UNSIGNED decimals (`3`, `1.25`, `.5`, `2e-1`); a sign is a usage
# error — a negative estimate would drive the ledger backwards. Both are
# normalized to %.6f before they are compared or written.
#
# Ledger: $GBRAIN_EVAL_SPEND_LEDGER (default ~/gbrain-lme-receipts/spend.jsonl),
# one JSON object per line with a numeric `cost_usd` field. The file MUST
# already exist: the first run sets GBRAIN_EVAL_SPEND_LEDGER_INIT=1 (or
# pre-creates the file), which prints a loud NEW LEDGER line — a ledger that
# silently starts at $0 because a path was mistyped is how a cap gets blown.
# Every non-empty line must parse (`{…"cost_usd":<number>…}`); otherwise the
# guard names the offending line numbers and refuses to launch (exit 3).
#
# After the command exits, this appends:
#   {"ts":"<UTC ISO>","estimate_usd":E,"cost_usd":C,"exit_code":N,"command":"..."}
# where C comes from $GBRAIN_EVAL_ACTUAL_COST_FILE when the command wrote one
# (a bare unsigned number, or a JSON object carrying `cost_usd`) AND it is
# positive; a malformed, signed, or non-positive cost falls back to the
# estimate (over-stating spend is the safe direction). When
# GBRAIN_EVAL_ACTUAL_COST_FILE is unset, a temp path is exported to the child
# so harnesses can report usage-derived cost without operator setup.
#
# Exit codes: the wrapped command's exit code · 2 usage error · 3 refused
# (cap exceeded, ledger missing, or ledger unparseable).

set -u

usage() {
  echo "usage: $0 <cap_usd> <estimate_usd> -- <command...>" >&2
  exit 2
}

# Unsigned decimal / exponent only. `+1`, `-1`, `1.`-with-sign, `abc`, '' → 1.
is_number() {
  case "$1" in
    ''|*[!0-9.eE+-]*) return 1 ;;
  esac
  printf '%s' "$1" | grep -Eq '^([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?$'
}

# Canonical %.6f so every value written into the ledger is valid JSON
# (`1.` and `.5` are not) and every comparison uses the same precision.
norm6() {
  awk -v x="$1" 'BEGIN { printf "%.6f", x + 0 }'
}

[ $# -ge 4 ] || usage
CAP="$1"; EST="$2"; SEP="$3"; shift 3
[ "$SEP" = "--" ] || usage
is_number "$CAP" || { echo "eval-spend-guard: cap_usd '$CAP' is not an unsigned number" >&2; exit 2; }
is_number "$EST" || { echo "eval-spend-guard: estimate_usd '$EST' is not an unsigned number (signed / negative estimates are refused)" >&2; exit 2; }
[ $# -ge 1 ] || usage
CAP="$(norm6 "$CAP")"
EST="$(norm6 "$EST")"

LEDGER="${GBRAIN_EVAL_SPEND_LEDGER:-$HOME/gbrain-lme-receipts/spend.jsonl}"
if [ ! -f "$LEDGER" ]; then
  if [ "${GBRAIN_EVAL_SPEND_LEDGER_INIT:-}" = "1" ]; then
    mkdir -p "$(dirname "$LEDGER")" || { echo "eval-spend-guard: cannot create ledger dir for $LEDGER" >&2; exit 2; }
    : > "$LEDGER" || { echo "eval-spend-guard: cannot create ledger $LEDGER" >&2; exit 2; }
    echo "eval-spend-guard: NEW LEDGER — created $LEDGER (spend history starts at \$0.000000; GBRAIN_EVAL_SPEND_LEDGER_INIT=1)" >&2
  else
    echo "eval-spend-guard: REFUSED — ledger does not exist: $LEDGER" >&2
    echo "eval-spend-guard: a missing ledger is NOT a \$0 ledger. Pre-create the file, or set GBRAIN_EVAL_SPEND_LEDGER_INIT=1 for the first run only." >&2
    echo "eval-spend-guard: command NOT run: $*" >&2
    exit 3
  fi
fi

# Audit + sum the ledger in one awk pass (no jq). A line counts ONLY when it
# is a complete JSON object (`{…}`) carrying an UNSIGNED numeric cost_usd
# followed by `,` or `}` — a truncated tail, a string-typed cost, a signed
# cost, or junk all count as unparseable. Prints: <lines> <parsed> <sum> <bad-line-list>
ledger_audit() {
  awk '
    /^[[:space:]]*$/ { next }
    {
      n++
      ok = 0
      if ($0 ~ /^[[:space:]]*\{.*\}[[:space:]]*$/ &&
          match($0, /"cost_usd"[[:space:]]*:[[:space:]]*([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?[[:space:]]*[,}]/)) {
        tok = substr($0, RSTART, RLENGTH)
        sub(/^"cost_usd"[[:space:]]*:[[:space:]]*/, "", tok)
        sub(/[[:space:]]*[,}]$/, "", tok)
        s += tok + 0
        ok = 1
      }
      if (!ok) { badn++; bad = bad (bad == "" ? "" : ",") NR }
    }
    END { printf "%d %d %.6f %s\n", n + 0, n - badn, s + 0, bad }
  ' "$1"
}

AUDIT="$(ledger_audit "$LEDGER")"
LINES="${AUDIT%% *}"; REST="${AUDIT#* }"
PARSED="${REST%% *}"; REST="${REST#* }"
TOTAL="${REST%% *}"; BAD="${REST#* }"
[ "$BAD" = "$TOTAL" ] && BAD=""   # no fourth field → awk printed nothing after the sum
if [ "$LINES" != "$PARSED" ]; then
  echo "eval-spend-guard: REFUSED — ledger $LEDGER has $((LINES - PARSED)) unparseable line(s) out of $LINES (line numbers: $BAD)" >&2
  echo "eval-spend-guard: every line must be a complete JSON object with an unsigned numeric cost_usd; repair or remove the offending lines — never guess spend" >&2
  echo "eval-spend-guard: command NOT run: $*" >&2
  exit 3
fi

PROJECTED="$(awk -v a="$TOTAL" -v b="$EST" 'BEGIN { printf "%.6f", a + b }')"
OVER="$(awk -v p="$PROJECTED" -v c="$CAP" 'BEGIN { print (p > c) ? 1 : 0 }')"

if [ "$OVER" = "1" ]; then
  echo "eval-spend-guard: REFUSED — ledger \$${TOTAL} + estimate \$${EST} = \$${PROJECTED} exceeds cap \$${CAP} (ledger: $LEDGER)" >&2
  echo "eval-spend-guard: command NOT run: $*" >&2
  exit 3
fi
echo "eval-spend-guard: ledger \$${TOTAL} ($LINES row(s)) + estimate \$${EST} = \$${PROJECTED} <= cap \$${CAP}; launching" >&2

# Cost file: honor the operator's path or hand the child a scratch one.
CLEANUP_COST_FILE=0
if [ -z "${GBRAIN_EVAL_ACTUAL_COST_FILE:-}" ]; then
  GBRAIN_EVAL_ACTUAL_COST_FILE="$(mktemp "${TMPDIR:-/tmp}/gbrain-eval-cost.XXXXXX")"
  rm -f "$GBRAIN_EVAL_ACTUAL_COST_FILE"
  CLEANUP_COST_FILE=1
fi
export GBRAIN_EVAL_ACTUAL_COST_FILE

"$@"
CODE=$?

# Actual cost: bare unsigned number or JSON with unsigned cost_usd, and it
# must be POSITIVE; anything else (malformed, signed, zero) → the estimate.
COST="$EST"
if [ -f "$GBRAIN_EVAL_ACTUAL_COST_FILE" ]; then
  RAW="$(tr -d '[:space:]' < "$GBRAIN_EVAL_ACTUAL_COST_FILE")"
  CANDIDATE=""
  if is_number "$RAW"; then
    CANDIDATE="$RAW"
  else
    FROM_JSON="$(grep -oE '"cost_usd"[[:space:]]*:[[:space:]]*([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?[[:space:]]*[,}]' "$GBRAIN_EVAL_ACTUAL_COST_FILE" 2>/dev/null \
      | head -1 | sed -E 's/^"cost_usd"[[:space:]]*:[[:space:]]*//; s/[[:space:]]*[,}]$//')"
    if [ -n "$FROM_JSON" ] && is_number "$FROM_JSON"; then CANDIDATE="$FROM_JSON"; fi
  fi
  if [ -n "$CANDIDATE" ]; then
    CANDIDATE="$(norm6 "$CANDIDATE")"
    POSITIVE="$(awk -v c="$CANDIDATE" 'BEGIN { print (c > 0) ? 1 : 0 }')"
    if [ "$POSITIVE" = "1" ]; then
      COST="$CANDIDATE"
    else
      echo "eval-spend-guard: cost file $GBRAIN_EVAL_ACTUAL_COST_FILE reports non-positive cost \$${CANDIDATE}; recording the estimate instead" >&2
    fi
  else
    echo "eval-spend-guard: cost file $GBRAIN_EVAL_ACTUAL_COST_FILE unreadable (need an unsigned number or {\"cost_usd\":<number>}); recording the estimate" >&2
  fi
fi
[ "$CLEANUP_COST_FILE" = "1" ] && rm -f "$GBRAIN_EVAL_ACTUAL_COST_FILE"

# JSON-escape the command (backslash, quote, control chars) without jq.
CMD_JSON="$(printf '%s' "$*" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/\t/\\t/g' -e 's/\r/\\r/g' | awk 'NR > 1 { printf "\\n" } { printf "%s", $0 }')"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '{"ts":"%s","estimate_usd":%s,"cost_usd":%s,"exit_code":%s,"command":"%s"}\n' \
  "$TS" "$EST" "$COST" "$CODE" "$CMD_JSON" >> "$LEDGER"
AFTER="$(ledger_audit "$LEDGER")"; AFTER="${AFTER#* }"; AFTER="${AFTER#* }"; AFTER="${AFTER%% *}"
echo "eval-spend-guard: recorded cost \$${COST} (exit $CODE); ledger now \$${AFTER}" >&2

exit "$CODE"
