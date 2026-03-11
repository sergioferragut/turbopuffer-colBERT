"""
Shared configuration: loads .env and creates the turbopuffer client.

All other modules import from here instead of reading env vars directly.
"""

import os
import sys

from dotenv import load_dotenv

# Load .env from the project root (the directory containing this file).
# Variables already set in the environment take precedence.
load_dotenv()

TURBOPUFFER_API_KEY = os.environ.get("TURBOPUFFER_API_KEY", "")
TURBOPUFFER_REGION = os.environ.get("TURBOPUFFER_REGION", "gcp-us-central1")

if not TURBOPUFFER_API_KEY:
    sys.exit(
        "TURBOPUFFER_API_KEY is not set.\n"
        "Add it to .env or export it as an environment variable."
    )

import turbopuffer as tpuf

client = tpuf.Turbopuffer(api_key=TURBOPUFFER_API_KEY, region=TURBOPUFFER_REGION)
