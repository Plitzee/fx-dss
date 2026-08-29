#!/usr/bin/env python3
"""XU LY DU LIEU FX TU NGUON NGOAI (HistData / Forex Tester / Forexite).

Lam 4 viec:
  1. TU DO DINH DANG  — dau phan cach, thu tu cot, kieu ngay gio (khong can khai bao)
  2. HIEU CHUAN MUI GIO — so voi du lieu EURUSD Dukascopy da co, tim do lech gio
     dung bang tuong quan, thay vi tin vao tai lieu
  3. GOP thanh H1 + D1 dung dinh dang pipeline
  4. BAO CAO CHAT LUONG tung cap

Chay:  py prep_fx.py                 (tu tim file trong histdata_raw/ va forextester_raw/)
       py prep_fx.py --src thu_muc_khac
"""
import argparse,glob,os,re,sys
import numpy as np, pandas as pd

REF_D1="dukas_h1_data/EURUSD_d1_direct.csv"      # moc chuan de hieu chuan (UTC)
OUT="fx_clean"
TZ="America/New_York"    # HistData / Forexite ghi theo gio New York, CO doi gio mua he

def sniff(path,nlines=5):
    with open(path,"r",errors="ignore") as f:
        head=[f.readline().strip() for _ in range(nlines)]
    head=[h for h in head if h]
    if not head: return None
    line=head[0]
    delim=max([";",",","\t"],key=lambda d:line.count(d))
    if line.count(delim)<3: return None
    cols=line.split(delim)
    return dict(delim=delim,ncol=len(cols),sample=head[:3],first=cols)

def parse_file(path,info):
    d=info["delim"]; nc=info["ncol"]
    df=pd.read_csv(path,sep=d,header=None,engine="c",
                   names=[f"c{i}" for i in range(nc)],dtype=str)
    c0=df.c0.astype(str)
    # Dang A: "YYYYMMDD HHMMSS" trong 1 cot  (HistData)
    if c0.str.match(r"^\d{8}\s+\d{6}$").iloc[0]:
        dt=pd.to_datetime(c0,format="%Y%m%d %H%M%S",errors="coerce"); k=1
    # Dang B: cot0=YYYYMMDD, cot1=HHMMSS  (Forexite / ForexTester)
    elif c0.str.match(r"^\d{8}$").iloc[0]:
        dt=pd.to_datetime(c0+" "+df.c1.astype(str).str.zfill(6),
                          format="%Y%m%d %H%M%S",errors="coerce"); k=2
    # Dang C: cot0 = ten cap, cot1=ngay, cot2=gio  (Forexite co ticker)
    elif len(c0.iloc[0])<=8 and df.c1.astype(str).str.match(r"^\d{8}$").iloc[0]:
        dt=pd.to_datetime(df.c1.astype(str)+" "+df.c2.astype(str).str.zfill(6),
                          format="%Y%m%d %H%M%S",errors="coerce"); k=3
    else:
        dt=pd.to_datetime(c0,errors="coerce"); k=1
    o=pd.to_numeric(df[f"c{k}"],errors="coerce")
    h=pd.to_numeric(df[f"c{k+1}"],errors="coerce")
    l=pd.to_numeric(df[f"c{k+2}"],errors="coerce")
    c=pd.to_numeric(df[f"c{k+3}"],errors="coerce")
    out=pd.DataFrame({"Date":dt,"open":o,"high":h,"low":l,"close":c}).dropna()
    return out[(out.low<=out.open)&(out.open<=out.high)&(out.low<=out.close)&(out.close<=out.high)]

def to_h1(df):
    g=df.assign(k=df.Date.dt.floor("h")).groupby("k")
    return pd.DataFrame({"open":g.open.first(),"high":g.high.max(),
        "low":g.low.min(),"close":g.close.last(),"n_bars":g.size()}
        ).reset_index().rename(columns={"k":"Date"})

def to_d1(h1):
    g=h1.assign(k=h1.Date.dt.normalize()).groupby("k")
    return pd.DataFrame({"open":g.open.first(),"high":g.high.max(),
        "low":g.low.min(),"close":g.close.last(),"n_bars":g.n_bars.sum(),
        "n_bars":g.size()}).reset_index().rename(columns={"k":"Date"})

def _scan(h1,ref,months=None,min_overlap=30):
    best=(None,None); sc=[]
    for sh in range(-12,13):
        d=to_d1(h1.assign(Date=h1.Date+pd.Timedelta(hours=sh)))
        if months is not None: d=d[d.Date.dt.month.isin(months)]
        m=d.merge(ref,on="Date",suffixes=("_n","_r"))
        if len(m)<min_overlap: continue
        err=float(np.median(np.abs(m.close_n-m.close_r))*1e4)
        sc.append((sh,err,len(m)))
        if best[1] is None or err<best[1]: best=(sh,err)
    return best[0],best[1],sc

def calibrate(h1_new,ref_d1,min_overlap=30):
    """Do do lech gio bang gia dong ngay. Do RIENG mua dong va mua he de phat hien DST."""
    sh_all,err_all,sc=_scan(h1_new,ref_d1,None,min_overlap)
    sh_w,err_w,_=_scan(h1_new,ref_d1,[12,1,2],min_overlap)      # mua dong (EST)
    sh_s,err_s,_=_scan(h1_new,ref_d1,[6,7,8],min_overlap)       # mua he  (EDT)
    return dict(all=(sh_all,err_all),winter=(sh_w,err_w),summer=(sh_s,err_s),scores=sc)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src",default=None)
    ap.add_argument("--ref",default=REF_D1)
    ap.add_argument("--out",default=OUT)
    ap.add_argument("--tz",default=TZ,help="mui gio nguon (mac dinh America/New_York; 'none' de giu nguyen)")
    a=ap.parse_args()

    srcs=[a.src] if a.src else [d for d in ("histdata_raw","forextester_raw",".") if os.path.isdir(d)]
    files=[]
    for s in srcs:
        files+=glob.glob(os.path.join(s,"*.csv"))+glob.glob(os.path.join(s,"*.txt"))
    files=[f for f in files if os.path.getsize(f)>50000 and "fx_clean" not in f]
    if not files:
        print("Khong tim thay file du lieu. Dat file vao histdata_raw/ hoac forextester_raw/"); sys.exit(1)

    print("="*78); print(f"BUOC 1 — TU DO DINH DANG  ({len(files)} file)"); print("="*78)
    info=sniff(files[0])
    if not info: print(f"Khong doc duoc {files[0]}"); sys.exit(1)
    print(f"  Mau: {os.path.basename(files[0])}")
    for s in info["sample"]: print(f"     {s[:90]}")
    print(f"  Dau phan cach {info['delim']!r} | {info['ncol']} cot")

    # gom file theo cap tien
    byp={}
    for f in files:
        m=re.search(r"([A-Z]{6})",os.path.basename(f).upper())
        if m: byp.setdefault(m.group(1),[]).append(f)
    print(f"  Cac cap phat hien: {', '.join(sorted(byp))}")

    ref=None
    if os.path.exists(a.ref):
        ref=pd.read_csv(a.ref,parse_dates=["Date"])[["Date","close"]].drop_duplicates("Date")
        print(f"  Moc chuan Dukascopy: {len(ref):,} ngay EURUSD (UTC)")
    else:
        print(f"  !! Khong thay {a.ref} — se BO QUA hieu chuan mui gio")

    os.makedirs(a.out,exist_ok=True)
    cal=None
    print("\n"+"="*78); print("BUOC 2 — XU LY TUNG CAP"); print("="*78)
    print(f"{'Cap':<9}{'file':>6}{'thanh M1':>12}{'mui gio':>16}{'sai so con':>12}{'thanh H1':>11}{'ngay D1':>10}")
    print("-"*78)
    shift_global=None
    for p in sorted(byp):
        parts=[]
        for f in sorted(byp[p]):
            i=sniff(f)
            if i:
                try: parts.append(parse_file(f,i))
                except Exception: pass
        if not parts: print(f"{p:<9}  khong doc duoc"); continue
        m1=pd.concat(parts).drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
        h1=to_h1(m1)
        # ── CHUYEN MUI GIO DUNG CHUAN (tu dong xu ly gio mua he) ──
        if a.tz and a.tz.lower()!="none":
            try:
                loc=m1.Date.dt.tz_localize(a.tz,ambiguous="NaT",nonexistent="NaT")
                m1=m1.assign(Date=loc.dt.tz_convert("UTC").dt.tz_localize(None)).dropna(subset=["Date"])
                h1=to_h1(m1)
            except Exception as e:
                print(f"  !! {p}: loi doi mui gio ({e}) — giu nguyen goc")
        err=None
        if ref is not None and p=="EURUSD" and cal is None:
            cal=calibrate(h1,ref)          # do sai so CON LAI sau khi da doi mui gio
        d1=to_d1(h1)
        h1[["Date","open","high","low","close","n_bars"]].to_csv(f"{a.out}/{p}_h1.csv",index=False)
        d1[["Date","open","high","low","close","n_bars"]].to_csv(f"{a.out}/{p}_d1.csv",index=False)
        se=f"{cal['all'][1]:.3f} pip" if (cal and p=="EURUSD" and cal['all'][1] is not None) else "—"
        print(f"{p:<9}{len(byp[p]):>6}{len(m1):>12,}{a.tz.split('/')[-1] if a.tz else 'nguyen goc':>16}"
              f"{se:>12}{len(h1):>11,}{len(d1):>10,}")

    print("-"*78)
    print("\n"+"="*78); print("BUOC 3 — KIEM CHUNG SAU KHI DOI MUI GIO"); print("="*78)
    if cal:
        sa,ea=cal["all"]; sw,ew=cal["winter"]; ss,es=cal["summer"]
        print(f"  Da doi tu {a.tz} sang UTC (tu dong xu ly gio mua he).")
        print(f"  Bang duoi do do lech CON LAI so voi EURUSD Dukascopy — dung phai la 0h.\n")
        print(f"  {'Giai doan':<22}{'lech con lai':>14}{'sai so trung vi':>18}")
        print("  "+"-"*54)
        for lab,(sx,ex) in (("Toan bo",cal["all"]),("Mua dong (T12,1,2)",cal["winter"]),
                            ("Mua he (T6,7,8)",cal["summer"])):
            print(f"  {lab:<22}{(f'{sx:+d}h' if sx is not None else '—'):>14}"
                  f"{(f'{ex:.3f} pip' if ex is not None else '—'):>18}")
        print("  "+"-"*54)
        okw = (sw==0) and (ss==0) and (sa==0)
        if okw:
            print(f"\n  ==> DUNG. Lech con lai = 0h o ca hai mua.")
            print(f"      Sai so {ea:.3f} pip la chenh lech THUC giua hai nha cung cap")
            print(f"      (Dukascopy vs HistData) — day la ket qua doi chieu cheo doc lap.")
        else:
            print(f"\n  ==> VAN CON LECH: toan bo {sa}h, dong {sw}h, he {ss}h.")
            print(f"      Mui gio '{a.tz}' co the khong dung. Bao Claude.")
    elif ref is not None:
        print(f"  !! CHUA hieu chuan duoc — thieu EURUSD trong du lieu moi tai.")
        print(f"     EURUSD la moc doi chieu duy nhat (vi ban da co no tu Dukascopy).")
        print(f"     Chay:  py histdata_dl.py --pairs EURUSD")
        print(f"     roi chay lai:  py prep_fx.py")
        print(f"     Du lieu hien tai da ghi NGUYEN GOC (chua chinh gio) — chua dung duoc.")
    print(f"\n  Da ghi vao {os.path.abspath(a.out)}/")
    print("  Gui ket qua man hinh nay cho Claude de chay phan tich 6 cap.")

if __name__=="__main__": main()
