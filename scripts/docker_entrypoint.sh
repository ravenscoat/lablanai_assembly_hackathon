#!/bin/sh
set -e

is_true() {
  case "$1" in
    true|True|TRUE|1|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

if is_true "$POPULATE_KB" || is_true "$NAVAI_POPULATE_KB"; then
  python scripts/populate_qdrant.py --auto
else
  echo "Skipping KB population on startup (set POPULATE_KB=1 to enable)"
fi

exec "$@"
