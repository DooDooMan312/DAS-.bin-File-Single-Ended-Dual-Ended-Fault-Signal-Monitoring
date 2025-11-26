#  ------------------------------------------------------------------------------
#  Copyright (c) 2025 Chaos
#  All rights reserved.
#  #
#  This software is proprietary and confidential.
#  Licensed exclusively to Shineway Technologies, Inc for internal use only,
#  according to the NDA / agreement signed on 2025.11.26
#  Unauthorized redistribution or disclosure is prohibited.
#  ------------------------------------------------------------------------------
#
#

# app/handlers.py
from pathlib import Path
import pandas as pd
import base64, time
from typing import Dict, Iterable

def make_alerts(row: Dict, th: Dict[str, float]) -> Iterable[Dict]:
    alerts = []
    rp_final = float(row.get("rp_final(%)", "nan")) if str(row.get("rp_final(%)","")).strip() != "" else None
    coh_med  = float(row.get("coh_med","nan")) if str(row.get("coh_med","")).strip() != "" else None
    poc_peak = float(row.get("poc_peak","nan")) if str(row.get("poc_peak","")).strip() != "" else None
    if rp_final is not None and rp_final < th.get("rp_final_min", 50):
        alerts.append({"type": "low_rp_final", "rp_final": rp_final})
    if coh_med is not None and coh_med < th.get("coh_med_min", 0.3):
        alerts.append({"type": "low_coherence", "coh_med": coh_med})
    if poc_peak is not None and poc_peak < th.get("poc_peak_min", 0.2):
        alerts.append({"type": "weak_poc", "poc_peak": poc_peak})
    return alerts

def _b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("ascii")

def upload_dir_outputs(sender, settings, data_dir: Path):
    out_root = data_dir / "_out"
    if not out_root.exists():
        return

    ts_now = int(time.time()*1000)

    sum_csv = out_root / "all_similarity_summary.csv"
    if sum_csv.exists():
        df = pd.read_csv(sum_csv)
        for _, r in df.iterrows():
            payload = {k: (None if pd.isna(v) else v) for k, v in r.items()}
            payload.update({"dir": str(data_dir), "ts": ts_now})
            sender.send_json("similarity", payload, key=str(payload.get("file","")))
            for al in make_alerts(payload, settings.thresholds):
                al.update({"file": payload.get("file",""), "dir": str(data_dir), "ts": ts_now})
                sender.send_json("alert", al, key=str(payload.get("file","")))

    for fdir in out_root.iterdir():
        if not fdir.is_dir():
            continue
        spec_dir = fdir / "spectrum"
        rp_dir   = fdir / "rp"

        for hcsv in spec_dir.glob("*_harmonics_col*.csv"):
            col = 0 if "col0" in hcsv.name else 1
            dfh = pd.read_csv(hcsv)
            sender.send_json("peaks", {
                "file_base": fdir.name,
                "col": int(col),
                "harmonics": dfh.to_dict(orient="records"),
                "dir": str(data_dir),
                "ts": ts_now,
            }, key=f"{fdir.name}:{col}")

        if settings.push_images:
            for img in spec_dir.glob("*.png"):
                sender.send_json("graph", {"meta":{"file_base":fdir.name,"kind":"spectrum","name":img.name,
                                                   "dir":str(data_dir),"ts":ts_now},
                                           "b64": _b64(img)}, key=img.name)
            for img in rp_dir.glob("*.png"):
                sender.send_json("graph", {"meta":{"file_base":fdir.name,"kind":"rp","name":img.name,
                                                   "dir":str(data_dir),"ts":ts_now},
                                           "b64": _b64(img)}, key=img.name)

    sender.flush()
