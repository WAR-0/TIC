#!/usr/bin/env python3
"""Aggregate running statistics from v8 batch logs."""

import glob
import re
import pandas as pd

PATTERN = re.compile(
    r"^\s*(D0|lambda|kappa|gamma|rho)\s*\|\s*r = ([+-]?\d+\.\d+)"
    r".*bias = ([+-]?\d+\.\d+).*RMSE = ([0-9.]+).*MAE = ([0-9.]+)"
)

records = []
for logfile in glob.glob("v8_full_batch_*.log"):
    with open(logfile) as f:
        for line in f:
            match = PATTERN.match(line)
            if match:
                param, r, bias, rmse, mae = match.groups()
                records.append(
                    {
                        "param": param,
                        "r": float(r),
                        "bias": float(bias),
                        "rmse": float(rmse),
                        "mae": float(mae),
                        "log": logfile,
                    }
                )

if not records:
    print("No simulation results found yet.")
else:
    df = pd.DataFrame(records)
    agg = df.groupby("param").agg(["mean", "std", "count"])
    pd.set_option("display.max_columns", None)
    print(agg)
