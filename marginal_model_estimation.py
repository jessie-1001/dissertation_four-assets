# =============================================================================
# SCRIPT 02: MARGINAL MODEL ESTIMATION AND DIAGNOSTICS (4-ASSET VERSION)
# =============================================================================

import pandas as pd
import numpy as np
from arch import arch_model
import statsmodels.api as sm
from scipy.stats import t
from statsmodels.stats.diagnostic import het_arch
import warnings
import traceback

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ===== SKEW-T PIT APPROXIMATION FUNCTION =====
def skewt_pit(std_resid, eta, lam):
    """
    Approximate PIT for skewed-t distribution (Fernandez & Steel, 1998).
    std_resid: standardized residuals
    eta: degrees of freedom
    lam: skewness parameter
    """
    delta = lam / np.sqrt(1 + lam**2)
    z = std_resid
    u = np.where(
        z < 0,
        (1 + lam**2) * t.cdf(z * np.sqrt(1 + lam**2), df=eta),
        (1 + lam**2) * (1 - t.cdf(-z * np.sqrt(1 + lam**2), df=eta))
    )
    return u

# ===== GARCH FITTING + DIAGNOSTICS =====
def fit_and_diagnose_garch(return_series, asset_name):
    print(f"\n--- Fitting model for {asset_name} ---")

    # Determine model type based on asset class
    if asset_name in ['SPX', 'NDX']:  # Equity indices
        model = arch_model(return_series, mean='ARX', lags=1, vol='Garch', p=1, o=1, q=1, dist='t')
        model_desc = "ARMA(1,1)-GJR-GARCH(1,1)-t"
    else:  # FX pairs
        model = arch_model(return_series, mean='ARX', lags=1, vol='EGARCH', p=2, q=1, dist='skewt')
        model_desc = "ARMA(1,1)-EGARCH(2,1)-skewt"

    print(f"Model: {model_desc}")

    best_result = None
    best_aic = np.inf

    for i in range(3):
        try:
            result = model.fit(update_freq=0, disp='off')
            if result.aic < best_aic:
                best_aic = result.aic
                best_result = result
        except Exception as e:
            print(f"   [!] Optimization attempt {i+1} failed: {str(e)}")
            continue

    if best_result is None:
        print("   [!] All optimization attempts failed, using fallback method")
        best_result = model.fit(update_freq=0, disp='off')

    result = best_result

    # --- Parameter Table ---
    params = result.params
    pvalues = result.pvalues
    param_table = pd.DataFrame({'Coefficient': params, 'P-value': pvalues})
    param_table['P-value'] = param_table['P-value'].apply(lambda p: "<0.0001" if p < 0.0001 else f"{p:.4f}")

    print(f"\n--- Parameter Estimates for {asset_name} ---")
    print(param_table.to_markdown(floatfmt=".4f"))

    # --- Diagnostic Tests ---
    std_resid = pd.Series(result.std_resid).dropna()
    lags_to_test = [5, 10, 20]
    diag_rows = []

    for lag in lags_to_test:
        lb1 = sm.stats.acorr_ljungbox(std_resid, lags=[lag], return_df=True)['lb_pvalue'].iloc[0]
        lb2 = sm.stats.acorr_ljungbox(std_resid**2, lags=[lag], return_df=True)['lb_pvalue'].iloc[0]
        diag_rows.append({'Test': f'Ljung-Box on Std Residuals (Lags={lag})', 'P-value': lb1})
        diag_rows.append({'Test': f'Ljung-Box on Sq Std Residuals (Lags={lag})', 'P-value': lb2})

    diag_table = pd.DataFrame(diag_rows)
    print(f"\n--- Diagnostic Tests for {asset_name} ---")
    print(diag_table.to_markdown(index=False, floatfmt=".4f"))

    arch_test = het_arch(std_resid)
    arch_fstat, arch_pval = arch_test[0], arch_test[1]

    if any(p < 0.05 for p in diag_table['P-value']) or arch_pval < 0.05:
        print("\nWARNING: Model shows signs of misspecification (p-value < 0.05).")
    else:
        print("\nSUCCESS: Model appears well-specified for this asset.")

    print(f"\nARCH-LM Test: F-stat = {arch_fstat:.4f}, p-value = {arch_pval:.4f}")
    print("-" * 80)

    return result

# ===== MAIN SCRIPT =====
if __name__ == '__main__':
    print("\n" + "="*80)
    print(">>> MARGINAL MODEL ESTIMATION AND DIAGNOSTICS (4-ASSET) <<<")

    try:
        # Modified input file for 4 assets
        input_file = 'spx_ndx_eurusd_usdjpy_daily.csv'
        data = pd.read_csv(input_file, index_col='Date', parse_dates=True)
        in_sample_end = '2019-12-31'
        in_sample_data = data.loc[:in_sample_end]

        # Extract all 4 return series
        spx_returns = in_sample_data['SPX_Return'] * 100
        ndx_returns = in_sample_data['NDX_Return'] * 100
        eurusd_returns = in_sample_data['EURUSD_Return'] * 100
        usdjpy_returns = in_sample_data['USDJPY_Return'] * 100

        print(f"Sample size: {len(spx_returns)} observations")
        print(f"Date range: {spx_returns.index[0].date()} to {spx_returns.index[-1].date()}")

        # Fit models for all 4 assets
        results = {}
        assets = {
            'SPX': spx_returns,
            'NDX': ndx_returns,
            'EURUSD': eurusd_returns,
            'USDJPY': usdjpy_returns
        }
        
        for asset_name, return_series in assets.items():
            results[asset_name] = fit_and_diagnose_garch(return_series, asset_name)

        # --- PIT Transform for all assets ---
        pit_data = {}
        
        # SPX (t-distribution)
        std_resid_spx = pd.Series(results['SPX'].std_resid).dropna()
        nu_spx = results['SPX'].params['nu']
        pit_data['u_spx'] = t.cdf(std_resid_spx, df=nu_spx)
        
        # NDX (t-distribution)
        std_resid_ndx = pd.Series(results['NDX'].std_resid).dropna()
        nu_ndx = results['NDX'].params['nu']
        pit_data['u_ndx'] = t.cdf(std_resid_ndx, df=nu_ndx)
        
        # EURUSD (skewed-t)
        std_resid_eurusd = pd.Series(results['EURUSD'].std_resid).dropna()
        eta_eurusd = results['EURUSD'].params['eta']
        lambda_eurusd = results['EURUSD'].params['lambda']
        pit_data['u_eurusd'] = skewt_pit(std_resid_eurusd, eta_eurusd, lambda_eurusd)
        
        # USDJPY (skewed-t)
        std_resid_usdjpy = pd.Series(results['USDJPY'].std_resid).dropna()
        eta_usdjpy = results['USDJPY'].params['eta']
        lambda_usdjpy = results['USDJPY'].params['lambda']
        pit_data['u_usdjpy'] = skewt_pit(std_resid_usdjpy, eta_usdjpy, lambda_usdjpy)

       # 使用样本内数据从第二个日期开始的索引
        pit_index = in_sample_data.index[1:]
        
        # 创建DataFrame，使用正确的索引
        pit_df = pd.DataFrame(pit_data, index=pit_index).dropna().clip(1e-6, 1 - 1e-6)
        pit_df.index.name = 'Date'
        
        # 保存到CSV
        pit_df.to_csv("copula_input_data_4asset.csv")
        # ================

        print("\nPIT Validation:")
        for col in pit_df.columns:
            print(f"{col} range: [{pit_df[col].min():.6f}, {pit_df[col].max():.6f}]")

        print("\n4-asset PIT data ready for copula modeling saved to 'copula_input_data_4asset.csv'.")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found. Please run script 01 first.")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

    print("="*80 + "\n")