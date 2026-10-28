"""Production-oriented AI Research Agent."""

# Automatically use Windows native trust store if available
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

__version__ = "0.1.0"

