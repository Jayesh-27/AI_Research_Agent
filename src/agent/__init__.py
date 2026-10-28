"""Production-oriented AI Research Agent."""

import sys

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# Automatically use Windows native trust store if available
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

__version__ = "0.1.0"

