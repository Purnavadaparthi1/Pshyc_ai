#!/usr/bin/env bash
set -e

# ─────────────────────────────────────────────
#  PSYCH.AI — Single-command startup script
#  Usage: ./start.sh
# ─────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║           PSYCH.AI  Startup              ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── 1. Check .env ────────────────────────────────────────────────────────
if [ ! -f "$BACKEND_DIR/.env" ]; then
  echo "⚠  No .env found. Creating from example…"
  cp "$BACKEND_DIR/.env.example" "$BACKEND_DIR/.env"
  echo "✗  Please edit backend/.env and set your GEMINI_API_KEY, then re-run."
  exit 1
fi

# Check key is set
if grep -q "your_gemini_api_key_here" "$BACKEND_DIR/.env"; then
  echo "✗  GEMINI_API_KEY is not set in backend/.env"
  echo "   Get your key at: https://aistudio.google.com/app/apikey"
  exit 1
fi

# ── 2. Python virtual env & deps ────────────────────────────────────────
echo "→ Setting up Python environment…"
cd "$BACKEND_DIR"

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✓ Python dependencies ready"

# ── 3. Frontend deps & build ─────────────────────────────────────────────
cd "$FRONTEND_DIR"

if [ ! -d "node_modules" ]; then
  echo "→ Installing frontend dependencies (npm install)…"
  npm install --silent
fi

if [ ! -d "dist" ] || [ -z "$(ls -A dist 2>/dev/null)" ]; then
  echo "→ Building frontend (npm run build)…"
  npm run build
fi
echo "✓ Frontend built"

# ── 4. Start backend (serves frontend) ───────────────────────────────────
cd "$BACKEND_DIR"
source venv/bin/activate

echo ""
echo "✓ Starting PSYCH.AI at http://localhost:8000"
echo "  API docs: http://localhost:8000/docs"
echo ""
echo "  TIP: To ingest your IGNOU PDFs:"
echo "       cp your_ignou.pdf backend/data/documents/"
echo "       python backend/scripts/ingest.py"
echo ""

python main.py
