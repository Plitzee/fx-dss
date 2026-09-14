"""Bao dam import duoc `api.main` du chay pytest tu bat cu thu muc con nao."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
