# =============================================================================
# SCRIPT 04: BACKTESTING AND MODEL ANALYSIS (FINAL ENHANCED VERSION)
#
# This definitive version incorporates expert feedback:
# 1. Implements more robust statistical tests for VaR.
# 2. Adds advanced tail risk metrics (Tail Ratio, Max Deviation).
# 3. Integrates the Basel "Traffic Light" regulatory framework for context.
# 4. Produces publication-quality summary tables for the dissertation.
# =============================================================================

import pandas as pd
import numpy as np
from scipy.stats import chi2
import warnings

warnings.filterwarnings('ignore')

def kupiec_pof_test(hits, alpha=0.01):
    """Robust Kupiec's proportion of failures (POF) test."""
    n = len(hits)
    if n == 0: return 0, np.nan, 1.0
    
    n1 = hits.sum()
    p = alpha
    pi_hat = n1 / n
    
    # Handle cases with no breaches or perfect prediction
    if n1 == 0:
        log_likelihood_ratio = -2 * (n * np.log(1 - p))
    elif np.isclose(pi_hat, 1.0):
        log_likelihood_ratio = -2 * (n * np.log(p))
    else:
        log_likelihood_ratio = 2 * (n1 * np.log(pi_hat / p) + (n - n1) * np.log((1 - pi_hat) / (1 - p)))
        
    p_value = 1 - chi2.cdf(log_likelihood_ratio, df=1)
    return n1, log_likelihood_ratio, p_value

def christoffersen_cc_test(hits, alpha=0.01):
    """Robust Christoffersen's conditional coverage test."""
    n1, lr_uc, p_uc = kupiec_pof_test(hits, alpha)
    
    if n1 < 2 or pd.isna(lr_uc):
        return np.nan
    
    n00, n01, n10, n11 = 0, 0, 0, 0
    for i in range(1, len(hits)):
        if not hits.iloc[i-1] and not hits.iloc[i]: n00 += 1
        elif not hits.iloc[i-1] and hits.iloc[i]: n01 += 1
        elif hits.iloc[i-1] and not hits.iloc[i]: n10 += 1
        elif hits.iloc[i-1] and hits.iloc[i]: n11 += 1
        
    if (n01 + n11) == 0 or (n00 + n10) == 0: return np.nan
    
    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi_all = (n01 + n11) / (n00 + n01 + n10 + n11)
    
    if np.isclose(pi_all, 0) or np.isclose(pi_all, 1): return np.nan

    log_l_uncond = (n00 + n10) * np.log(1 - pi_all) + (n01 + n11) * np.log(pi_all)
    log_l_ind = (n00 * np.log(1 - pi0) if pi0 < 1 else 0) + (n01 * np.log(pi0) if pi0 > 0 else 0) + \
                (n10 * np.log(1 - pi1) if pi1 < 1 else 0) + (n11 * np.log(pi1) if pi1 > 0 else 0)
    
    lr_ind = 2 * (log_l_ind - log_l_uncond)
    lr_cc = lr_uc + lr_ind
    p_value = 1 - chi2.cdf(lr_cc, df=2)
    return p_value

def acerbi_szekely_es_test(returns, var_forecasts, es_forecasts):
    """
    Acerbi & Szekely (2014) ES back‑test + 3 个尾部指标
    返回值顺序：Z (or "N/A")，Pass/Fail，Tail Ratio，Max Deviation
    """
    hits = returns < var_forecasts       # True = VaR breach
    n1   = hits.sum()
    if n1 == 0:
        return "No Breaches", "Pass", np.nan, np.nan

    breach_returns = returns[hits]
    breach_es      = es_forecasts[hits]

    # ------------- 附加尾部指标 ----------------
    tail_ratio = breach_returns.mean() / breach_es.mean()
    worst_loss_day = breach_returns.idxmin()
    max_dev = breach_returns.min() - breach_es.loc[worst_loss_day]

    # ------------- Acerbi‑Szekely Z -------------
    ratio = breach_returns / breach_es     # R/ES; 若 ES 足够宽则 < 1
    stat  = ratio - 1                      # 理论均值 0
    std_  = stat.std(ddof=1)

    # n1 == 1 或所有超损恰等于 ES -> 无方差，直接视为通过
    if std_ == 0 or np.isnan(std_):
        return "N/A", "Pass", f"{tail_ratio:.2f}", f"{max_dev:.4f}"

    z_stat = np.sqrt(n1) * stat.mean() / std_
    pass_fail = "Pass" if abs(z_stat) <= 1.645 else "Reject"

    return f"{z_stat:.4f}", pass_fail, f"{tail_ratio:.2f}", f"{max_dev:.4f}"

def basel_traffic_light(n_breaches, n_obs, alpha=0.01):
    """Classifies VaR model based on Basel traffic light zones."""
    # Scale breaches to a 250-day equivalent for comparability
    if n_obs == 0: return "N/A"
    scaled_breaches = n_breaches * (250 / n_obs)
    
    if scaled_breaches <= 4: return "Green Zone"
    if scaled_breaches <= 9: return "Yellow Zone"
    return "Red Zone"

def run_analysis_for_period(returns, forecasts, period_name):
    """Runs all backtests for a given period and returns a DataFrame."""
    results = []
    alpha = 0.01

    for col in forecasts.columns:
        if 'VaR' in col:
            model_name = col.split('_')[0]
            var_series = forecasts[col]
            es_series = forecasts[f"{model_name}_ES_99"]
            
            hits = returns < var_series
            n1, _, p_uc = kupiec_pof_test(hits, alpha)
            p_cc = christoffersen_cc_test(hits, alpha)
            z_es, es_pass_fail, tail_ratio, max_dev = acerbi_szekely_es_test(returns, var_series, es_series)
            
            # Get Basel Zone classification
            basel_zone = basel_traffic_light(n1, len(returns))
            
            results.append({
                'Model': model_name,
                'VaR Breaches': f"{n1} (Exp: {len(returns)*alpha:.1f})",
                'Kupiec p-val': p_uc,
                'Christoffersen p-val': p_cc,
                'Basel Zone': basel_zone,
                'ES Z-stat': z_es,
                'ES Pass/Fail': es_pass_fail,
                'Tail Ratio': tail_ratio,
                'Max Deviation': max_dev
            })
    return pd.DataFrame(results)

if __name__ == '__main__':
    print("\n\n" + "="*80)
    print(">>> SCRIPT 04: GENERATING FINAL BACKTESTING RESULTS (TABLE 4.5) <<<")
    
    try:
        forecast_file = 'garch_copula_forecasts.csv'
        data_file = 'spx_eurusd_daily_data.csv'
        forecasts = pd.read_csv(forecast_file, index_col='Date', parse_dates=True)
        full_data = pd.read_csv(data_file, index_col='Date', parse_dates=True)
    except FileNotFoundError as e:
        print(f"Error: Could not find data file '{e.filename}'.")
        print("Please ensure scripts 01 and 03 have run successfully.")
        exit()

    aligned_data = full_data.join(forecasts, how='inner')

    aligned_data['Portfolio_Return'] = (
        aligned_data['SPX_Return']   * aligned_data['Gaussian_Weight_SPX'] +
        aligned_data['EURUSD_Return']* aligned_data['Gaussian_Weight_EURUSD']
    )
    
    periods = {
        "Full Out-of-Sample Period": aligned_data,
        "COVID-19 Shock (Mar-Apr 2020)": aligned_data.loc['2020-03-01':'2020-04-30'],
        "Geopolitical Shock (Feb-Jun 2022)": aligned_data.loc['2022-02-24':'2022-06-30']
    }
    
    for name, df in periods.items():
        if not df.empty:
            period_returns = df['Portfolio_Return']
            period_forecasts = df[forecasts.columns]
            results_table = run_analysis_for_period(period_returns, period_forecasts, name)
            
            print(f"\n--- Backtesting Results for: {name} ---")
            print(results_table.to_markdown(index=False, floatfmt=".4f"))
    
    print("="*80 + "\n")