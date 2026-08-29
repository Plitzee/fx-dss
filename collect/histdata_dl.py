#!/usr/bin/env python3
"""TAI M1 TU HISTDATA.COM — 1 request = 1 NAM du lieu (thay vi 1 thang nhu Dukascopy).

Chay:
    py histdata_dl.py --probe          # thu 1 file, bao cao chi tiet, KHONG tai gi them
    py histdata_dl.py                  # tai 5 cap con thieu, 2010-2025
    py histdata_dl.py --pairs EURUSD --from 2015 --to 2016
"""
import argparse,io,os,re,sys,time,urllib.parse,urllib.request,zipfile

BASE="https://www.histdata.com"
PAGE=BASE+"/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{pair}/{year}"
POST=BASE+"/get.php"
UA={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                 "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language":"en-US,en;q=0.9"}
# EURUSD da co tu Dukascopy -> mac dinh chi lay 5 cap con lai (trung bo 6 cap cua HuyH)
DEFAULT=["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF"]  # EURUSD = moc hieu chuan mui gio

def get(url,ref=None,data=None,timeout=60):
    h=dict(UA)
    if ref: h["Referer"]=ref
    if data is not None:
        h["Content-Type"]="application/x-www-form-urlencoded"
        h["Origin"]=BASE
        data=urllib.parse.urlencode(data).encode()
    r=urllib.request.urlopen(urllib.request.Request(url,headers=h,data=data),timeout=timeout)
    return r.read(),r.headers

def token_of(html):
    for pat in (r'id=["\']tk["\'][^>]*value=["\']([^"\']+)',
                r'name=["\']tk["\'][^>]*value=["\']([^"\']+)',
                r'value=["\']([^"\']+)["\'][^>]*id=["\']tk["\']'):
        m=re.search(pat,html,re.I)
        if m: return m.group(1)
    return None

def fetch_year(pair,year,verbose=False):
    ref=PAGE.format(pair=pair.lower(),year=year)
    html,_=get(ref)
    html=html.decode("utf-8","ignore")
    tk=token_of(html)
    if not tk: raise RuntimeError("khong tim thay token 'tk' trong trang")
    if verbose: print(f"     token = {tk[:32]}...")
    body={"tk":tk,"date":str(year),"datemonth":str(year),
          "platform":"ASCII","timeframe":"M1","fxpair":pair.upper()}
    raw,hdrs=get(POST,ref=ref,data=body)
    if verbose: print(f"     tra ve {len(raw):,} byte | {hdrs.get('Content-Type')}")
    if not raw[:2]==b"PK": raise RuntimeError(f"khong phai ZIP (dau file: {raw[:60]!r})")
    z=zipfile.ZipFile(io.BytesIO(raw))
    names=[n for n in z.namelist() if n.lower().endswith(".csv")]
    if not names: raise RuntimeError(f"ZIP khong co .csv: {z.namelist()}")
    return z.read(names[0]),names[0],len(raw)

def probe():
    print("="*74); print("DO THU — EURUSD 2020"); print("="*74)
    try:
        data,name,zb=fetch_year("EURUSD",2020,verbose=True)
    except Exception as e:
        print(f"  THAT BAI: {e}")
        print("\n  -> HistData khong dung duoc theo cach nay.")
        print("     Dung phuong an Forex Tester (tai thu cong 5 file) thay the.")
        return False
    txt=data.decode("utf-8","ignore")
    lines=[l for l in txt.split("\n") if l.strip()][:3]
    n=txt.count("\n")
    print(f"  File trong ZIP : {name}")
    print(f"  Kich thuoc     : nen {zb:,} B -> giai nen {len(data):,} B")
    print(f"  So dong        : {n:,}  (1 nam M1 ~ 370.000 dong)")
    print(f"  3 dong dau:")
    for l in lines: print(f"     {l}")
    d=";" if ";" in lines[0] else ("," if "," in lines[0] else None)
    print(f"  Dau phan cach  : {d!r} | so cot: {len(lines[0].split(d)) if d else '?'}")
    ok = n>100000 and d is not None
    print(f"\n  ==> {'HOAT DONG. Chay lai khong co --probe de tai.' if ok else 'DINH DANG LA — gui ket qua nay cho Claude.'}")
    return ok

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pairs",default=",".join(DEFAULT))
    ap.add_argument("--from",dest="y0",type=int,default=2010)
    ap.add_argument("--to",dest="y1",type=int,default=2025)
    ap.add_argument("--out",default="histdata_raw")
    ap.add_argument("--probe",action="store_true")
    ap.add_argument("--gap",type=float,default=1.0)
    a=ap.parse_args()
    if a.probe: sys.exit(0 if probe() else 1)
    if not probe():
        print("\nDung lai vi buoc do thu that bai."); sys.exit(1)

    pairs=[p.strip().upper() for p in a.pairs.split(",")]
    os.makedirs(a.out,exist_ok=True)
    tot=len(pairs)*(a.y1-a.y0+1); done=0; fail=[]; t0=time.time()
    print("\n"+"="*74); print(f"TAI {len(pairs)} cap x {a.y1-a.y0+1} nam = {tot} file"); print("="*74)
    for p in pairs:
        for y in range(a.y0,a.y1+1):
            out=os.path.join(a.out,f"{p}_{y}.csv")
            if os.path.exists(out) and os.path.getsize(out)>100000:
                done+=1; continue
            try:
                data,_,_=fetch_year(p,y)
                open(out,"wb").write(data); done+=1
                print(f"  {p} {y}: {len(data):,} B  ({done}/{tot}, {time.time()-t0:.0f}s)",flush=True)
            except Exception as e:
                fail.append((p,y)); print(f"  ! {p} {y}: {str(e)[:55]}",flush=True)
            time.sleep(a.gap)
    print(f"\nXong {done}/{tot} file trong {(time.time()-t0)/60:.1f} phut -> {os.path.abspath(a.out)}")
    if fail: print(f"Loi {len(fail)} file: {fail[:10]}{'...' if len(fail)>10 else ''}  — chay lai de tai bu")
    print("\nBuoc tiep: py prep_fx.py")

if __name__=="__main__": main()
