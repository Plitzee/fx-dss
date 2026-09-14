"""/refresh — tinh lai cache tu du lieu tren dia (khong tai lai tu mang)."""
import datetime as dt

from fastapi import APIRouter, Query

from api.config import PAIRS
from api.cache import lay

router = APIRouter()


@router.post("/refresh")
def refresh(pair: str = Query(None)):
    """Tinh lai tu du lieu tren dia. KHONG tai lai tu mang — viec do do
    jobs/cap_nhat.py lam, de mot request cua nguoi dung khong the goi ra ngoai."""
    ds = [pair] if pair else list(PAIRS)
    for p in ds:
        lay(p, moi=True)
    return {"da_tinh_lai": ds, "luc": dt.datetime.utcnow().isoformat() + "Z"}
