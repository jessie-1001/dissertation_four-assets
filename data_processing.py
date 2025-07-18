# =============================================================================
# SCRIPT 01: DATA ACQUISITION AND PROCESSING (ENHANCED – 4 assets)
# =============================================================================
import yfinance as yf
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import jarque_bera, kurtosis
import os, textwrap

def data_processing_and_summary():
    """
    Download, clean, summarise S&P500, Nasdaq‑100, EUR/USD, USD/JPY
    并执行多项数据质量检查与可视化。
    """

    # ------------------------------------------------------------------
    # 1 ‑ PARAMETERS
    # ------------------------------------------------------------------
    tickers = ['^GSPC', '^NDX', 'EUR=X', 'JPY=X']
    col_map = {'^GSPC': 'SPX',
               '^NDX':  'NDX',
               'EUR=X': 'EURUSD',
               'JPY=X': 'USDJPY'}

    start_date, end_date = '2007-01-01', '2025-06-01'
    output_file = 'spx_ndx_eurusd_usdjpy_daily.csv'
    plot_dir = 'data_quality_plots'
    os.makedirs(plot_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 2 ‑ DOWNLOAD
    # ------------------------------------------------------------------
    print(f"\nDownloading {tickers}  ({start_date} → {end_date}) …")
    price_raw = yf.download(tickers, start=start_date, end=end_date,
                            auto_adjust=False, progress=False)['Close']
    price_raw.rename(columns=col_map, inplace=True)
    print(f"Rows fetched: {len(price_raw)}")

    # ------------------------------------------------------------------
    # 3 ‑ CLEAN & RETURNS
    # ------------------------------------------------------------------
    price = price_raw.dropna()
    print(f"Aligned rows (no NaNs): {len(price)}  "
          f"(dropped {len(price_raw)-len(price)})")

    returns = np.log(price/price.shift(1)).rename(
        columns={c: f"{c}_Return" for c in price.columns})

    final = pd.concat([price, returns], axis=1).dropna()
    final.to_csv(output_file)
    print(f"Saved cleaned data → {output_file}\n")

    # ------------------------------------------------------------------
    # 4 ‑ DESCRIPTIVE STATISTICS
    # ------------------------------------------------------------------
    desc_stats = returns.describe().T

    # pandas 指标
    desc_stats['Skewness'] = returns.skew()
    desc_stats['Kurtosis'] = returns.kurtosis()

    def safe_kurt(arr):
        try:
            val = kurtosis(arr, fisher=False, nan_policy='omit')
            return np.nan if np.isinf(val) else val
        except Exception:
            return np.nan

    def safe_jb(arr):
        try:
            jb = jarque_bera(arr)
            # 若返回 inf -> nan
            return (np.nan if np.isinf(jb[0]) else jb[0],
                    np.nan if np.isinf(jb[1]) else jb[1])
        except Exception:
            return (np.nan, np.nan)

    desc_stats['Kurtosis_scipy'] = [safe_kurt(returns[c].dropna()) for c in returns]

    jb_vals = [safe_jb(returns[c].dropna()) for c in returns]
    desc_stats['Jarque-Bera'] = [v[0] for v in jb_vals]
    desc_stats['JB p-value']  = [v[1] for v in jb_vals]

    # ------------------------------------------------------------------
    # 5 ‑ DATA‑QUALITY CHECKS
    # ------------------------------------------------------------------
    print(">>> ENHANCED DATA QUALITY CHECKS <<<\n")

    # 5‑1  Extreme‑value detection (asset‑specific thresholds)
    thr = {'SPX':0.05, 'NDX':0.06, 'EURUSD':0.03, 'USDJPY':0.03}
    extremes = {}
    for asset in price.columns:
        rname = f"{asset}_Return"
        extremes[asset] = final[(final[rname] < -thr[asset]) |
                                (final[rname] >  thr[asset])]
        print(f"{asset} extreme returns (>|{thr[asset]*100:.0f}%|): "
              f"{len(extremes[asset])}")

    # 5‑2  Continuity (gap) check
    print("\n--- Data Continuity Check ---")
    gap_days = (final.asfreq('B').index.to_series().diff()
                           .dt.days.dropna())
    gaps = gap_days[gap_days > 3]
    if gaps.empty:
        print("No significant gaps (beyond weekends).")
    else:
        print(f"Detected {len(gaps)} gaps (>3 days). Top 5:")
        for i,(idx,days) in enumerate(gaps.items()):
            if i==5: break
            prev = final.index[final.index.get_loc(idx)-1]
            print(f"  {prev.date()} → {idx.date()}  ({days} days)")

    # ------------------------------------------------------------------
    # 6 ‑ VISUALISATION
    # ------------------------------------------------------------------
    print("\n--- Generating visualisations ---")

    # 6‑1  Price series
    fig, ax = plt.subplots(2,2, figsize=(14,10))
    for i,asset in enumerate(price.columns):
        price[asset].plot(ax=ax[i//2, i%2],
                          title=f"{asset} Price")
        ax[i//2, i%2].grid(True)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/price_series.png")
    plt.close()

    # 6‑2  Return histograms
    fig, ax = plt.subplots(2,2, figsize=(14,10))
    for i,asset in enumerate(price.columns):
        sns.histplot(final[f"{asset}_Return"], ax=ax[i//2, i%2],
                     bins=50, kde=True)
        ax[i//2, i%2].axvline(0, ls='--', c='red')
        ax[i//2, i%2].set_title(f"{asset} Return Dist")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/return_hist.png")
    plt.close()

    # 6‑3  Extreme returns marked (example: SPX & EURUSD)
    fig, ax = plt.subplots(2,1, figsize=(14,8), sharex=True)
    for j,(asset,color) in enumerate([('SPX','blue'), ('EURUSD','green')]):
        r = final[f"{asset}_Return"]
        ax[j].plot(r, color=color, alpha=.7, label=f"{asset} Daily Returns")
        ax[j].scatter(extremes[asset].index,
                      extremes[asset][f"{asset}_Return"],
                      color='red', s=30, label='Extreme')
        ax[j].axhline(thr[asset], ls='--', c='gray')
        ax[j].axhline(-thr[asset], ls='--', c='gray')
        ax[j].legend(); ax[j].grid(True)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/extreme_events.png")
    plt.close()
    print(f"Plots saved to “{plot_dir}/”")

    # ------------------------------------------------------------------
    # 7 ‑ QUALITY SUMMARY
    # ------------------------------------------------------------------
    print("\n>>> DATA QUALITY REPORT SUMMARY <<<")
    print(f"Observations : {len(final)}")
    print(f"Date range   : {final.index[0].date()} – {final.index[-1].date()}")
    for asset in price.columns:
        print(f"{asset} extremes : {len(extremes[asset])} "
              f"({len(extremes[asset])/len(final)*100:.2f}%)")
    print("="*80+"\n")

# ------------------------------------------------------------------
if __name__ == '__main__':
    data_processing_and_summary()
