#!/bin/sh
set -e

if [ -z "${LARA_API_KEY}" ]; then
    echo "FATAL: LARA_API_KEY is not set. The frontend cannot start without a backend API key." >&2
    exit 1
fi

exec /docker-entrypoint.sh "$@"
