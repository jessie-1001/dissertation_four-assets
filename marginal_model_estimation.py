# =============================================================================
# SCRIPT 02: MARGINAL MODEL ESTIMATION AND DIAGNOSTICS (STABLE FIXED VERSION)
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
        2 * t.cdf(z / (1 - delta), df=eta),
        1 + (1 - 2 * t.cdf(-z / (1 + delta), df=eta))
    )
    return u

# ===== GARCH FITTING + DIAGNOSTICS =====
def fit_and_diagnose_garch(return_series, asset_name):
    print(f"\n--- Fitting model for {asset_name} ---")

    if "S&P" in asset_name:
        model = arch_model(return_series, mean='ARX', lags=1, vol='Garch', p=1, o=1, q=1, dist='t')
        model_desc = "ARMA(1,1)-GJR-GARCH(1,1)-t"
    else:
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
    print(">>> MARGINAL MODEL ESTIMATION AND DIAGNOSTICS <<<")

    try:
        input_file = 'spx_eurusd_daily_data.csv'
        data = pd.read_csv(input_file, index_col='Date', parse_dates=True)
        in_sample_end = '2019-12-31'
        in_sample_data = data.loc[:in_sample_end]

        spx_returns = in_sample_data['SPX_Return'] * 100
        eurusd_returns = in_sample_data['EURUSD_Return'] * 100

        print(f"Sample size: {len(spx_returns)} observations")
        print(f"Date range: {spx_returns.index[0].date()} to {spx_returns.index[-1].date()}")

        result_spx = fit_and_diagnose_garch(spx_returns, 'S&P 500')
        result_eurusd = fit_and_diagnose_garch(eurusd_returns, 'EUR/USD')

        # --- PIT Transform ---
        std_resid_spx = pd.Series(result_spx.std_resid).dropna()
        nu_spx = result_spx.params['nu']
        u_spx = t.cdf(std_resid_spx, df=nu_spx)

        std_resid_eurusd = pd.Series(result_eurusd.std_resid).dropna()
        eta_eurusd = result_eurusd.params['eta']
        lambda_eurusd = result_eurusd.params['lambda']
        u_eurusd = skewt_pit(std_resid_eurusd, eta_eurusd, lambda_eurusd)

        # --- Align indexes and clip extreme values ---
        pit_df = pd.DataFrame({
            'u_spx': u_spx,
            'u_eurusd': pd.Series(u_eurusd, index=std_resid_eurusd.index)
        }).dropna().clip(1e-6, 1 - 1e-6)

        print("\nPIT Validation:")
        print(f"SPX PIT range: [{pit_df['u_spx'].min():.6f}, {pit_df['u_spx'].max():.6f}]")
        print(f"EURUSD PIT range: [{pit_df['u_eurusd'].min():.6f}, {pit_df['u_eurusd'].max():.6f}]")

        pit_df.to_csv("copula_input_data.csv")
        print("\nData ready for copula modeling saved to 'copula_input_data.csv'.")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found. Please run script 01 first.")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

    print("="*80 + "\n")