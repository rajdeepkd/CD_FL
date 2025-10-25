#!/usr/bin/env bash
#
# This is an example helper script. It does NOT download anything.
#
# Usage:
#   bash scripts/prepare_data_example.sh
#
# What it does:
#   - Creates data/raw_data if missing
#   - Touches placeholder CSVs for two example clients
#

set -e

mkdir -p data/raw_data
touch data/raw_data/client1.csv
touch data/raw_data/client2.csv

echo "Place your real client CSVs in data/raw_data/ (one per client)."
echo "Each CSV should include features + a binary label column."
