# =============================================================================
# SCRIPT 01: DATA ACQUISITION AND PROCESSING  (4‑Asset Enhanced)
# =============================================================================
import os, yfinance as yf, pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from scipy.stats import jarque_bera, kurtosis

def data_processing_and_summary():
    # ---------- 1. Parameters ----------
    tickers     = ['^GSPC', '^NDX', 'EUR=X', 'JPY=X']       # S&P500 / NASDAQ100 / EURUSD / USDJPY
    rename_map  = {'^GSPC': 'SPX', '^NDX': 'NDX', 'EUR=X':'EURUSD', 'JPY=X':'USDJPY'}
    start_date  = '2007-01-01'
    end_date    = '2025-06-01'
    csv_out     = 'spx_ndx_eurusd_usdjpy_daily.csv'         # <-- downstream filename
    plot_dir    = 'data_quality_plots'
    os.makedirs(plot_dir, exist_ok=True)

    # ---------- 2. Download ----------
    print(f"Downloading {tickers}  ({start_date} → {end_date}) …")
    px = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Close']
    print(f"Rows fetched: {len(px)}")

    # ---------- 3. Basic cleaning ----------
    px = px.rename(columns=rename_map).dropna()
    print(f"Aligned rows (no NaNs): {len(px)}  (dropped {len(px.index)-len(px)} )")

    # log‑returns (%)
    ret = np.log(px / px.shift(1))
    ret.columns = [f"{c}_Return" for c in ret.columns]
    data = pd.concat([px, ret], axis=1).dropna()
    data.to_csv(csv_out)
    print(f"Saved cleaned data → {csv_out}")

    # ---------- 4. Descriptive stats (Table 4.1) ----------
    returns = data[[c for c in data.columns if c.endswith('_Return')]]
    desc = returns.describe().T
    desc['Skewness'] = returns.skew()
    desc['Kurtosis'] = returns.kurtosis()
    desc['Kurtosis_scipy'] = [kurtosis(returns[c], fisher=False) for c in returns]
    jb = [jarque_bera(returns[c]) for c in returns]
    desc['Jarque-Bera'] = [x[0] for x in jb]
    desc['JB p-value']  = [x[1] for x in jb]

    print("\n" + "="*80)
    print(">>> OUTPUT FOR DISSERTATION: TABLE 4.1 <<<")
    keep_cols = ['mean','std','min','max','Skewness','Kurtosis',
                 'Kurtosis_scipy','Jarque-Bera','JB p-value']
    print(desc[keep_cols].to_markdown(floatfmt=".4f"))
    print("="*80)

    # ---------- 5. Data‑quality checks ----------
    print("\n>>> ENHANCED DATA QUALITY CHECKS <<<")
    extremes = {
        'EURUSD': data[abs(data['EURUSD_Return']) > 0.03],
        'USDJPY': data[abs(data['USDJPY_Return']) > 0.03],
        'SPX'   : data[abs(data['SPX_Return' ]) > 0.05],
        'NDX'   : data[abs(data['NDX_Return' ]) > 0.06],
    }
    for k,v in extremes.items():
        print(f"{k} extreme returns : {len(v)}")

    # ---------- 6. Gaps ----------
    gaps = data.asfreq('B').index.to_series().diff().dt.days.gt(3)
    if gaps.any():
        print("Gaps >3 days detected!")
    else:
        print("\n--- Data Continuity Check ---\nNo significant gaps (beyond weekends).")

    # ---------- 7. Quick visualisations ----------
    print("\n--- Generating visualisations ---")
    # 7.1 price series
    fig, ax = plt.subplots(2,2, figsize=(11,7), sharex=True)
    for a, col, ttl in zip(ax.ravel(),
                           ['EURUSD','USDJPY','SPX','NDX'],
                           ['EURUSD Price','USDJPY Price','SPX Price','NDX Price']):
        a.plot(data[col]); a.set_title(ttl); a.grid(True)
    plt.tight_layout(); plt.savefig(f"{plot_dir}/price_series.png"); plt.close()

    # 7.2 return hist
    fig, ax = plt.subplots(2,2, figsize=(11,7))
    for a, col in zip(ax.ravel(),
                      ['EURUSD_Return','USDJPY_Return','SPX_Return','NDX_Return']):
        sns.histplot(data[col], bins=50, kde=True, ax=a)
        a.set_title(f"{col.split('_')[0]} Return Dist"); a.axvline(0, ls='--', c='r')
    plt.tight_layout(); plt.savefig(f"{plot_dir}/return_hist.png"); plt.close()

    print(f"Plots saved to “{plot_dir}/”")

    # ---------- 8. Report ----------
    print("\n>>> DATA QUALITY REPORT SUMMARY <<<")
    print(f"Observations : {len(data)}")
    print(f"Date range   : {data.index[0].date()} – {data.index[-1].date()}")
    for k,v in extremes.items():
        pct = len(v)/len(data)*100
        print(f"{k} extremes : {len(v)} ({pct:.2f}%)")
    print("="*80 + "\n")

if __name__ == "__main__":
    data_processing_and_summary()
