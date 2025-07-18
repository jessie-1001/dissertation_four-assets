# =============================================================================
# SCRIPT 03: GARCH-COPULA MODEL WITH STABILITY ENHANCEMENTS
#
# PRESERVES GARCH MODELING WHILE ADDRESSING CONVERGENCE AND PERFORMANCE ISSUES
# =============================================================================

import pandas as pd
import numpy as np
from scipy.stats import norm, t
from arch import arch_model
from tqdm import tqdm
import warnings
from scipy.linalg import cholesky
from scipy.optimize import minimize
import pickle
from scipy.stats import t as student_t
from scipy.optimize import minimize
from scipy import stats
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# --- 1. STABLE COPULA SAMPLERS ---
def calc_min_var_weights(vol_spx, vol_eurusd, rho):
    var_spx, var_eu = vol_spx**2, vol_eurusd**2
    cov = rho * vol_spx * vol_eurusd
    w_spx = (var_eu - cov) / (var_spx + var_eu - 2*cov)
    w_spx = np.clip(w_spx, 0.2, 0.8)
    return w_spx, 1 - w_spx

def sample_gaussian_copula(n_samples, corr_matrix):
    """Robust Gaussian copula sampling using Cholesky decomposition."""
    try:
        L = cholesky(corr_matrix, lower=True)
        z = np.random.normal(0, 1, size=(n_samples, 2))
        z_correlated = z @ L.T
        return norm.cdf(z_correlated)
    except np.linalg.LinAlgError:
        # 当矩阵非正定时使用特征值修正
        eigvals, eigvecs = np.linalg.eigh(corr_matrix)
        eigvals = np.maximum(eigvals, 1e-6)
        reconstituted = eigvecs @ np.diag(eigvals) @ eigvecs.T
        D = np.diag(1 / np.sqrt(np.diag(reconstituted)))
        corr_matrix = D @ reconstituted @ D
        L = cholesky(corr_matrix, lower=True)
        z = np.random.normal(0, 1, size=(n_samples, 2))
        z_correlated = z @ L.T
        return norm.cdf(z_correlated)

def sample_t_copula(n_samples, corr_matrix, df):
    """Robust t-copula sampling with Cholesky decomposition."""
    try:
        L = cholesky(corr_matrix, lower=True)
        g = np.random.chisquare(df, n_samples)
        z = np.random.normal(0, 1, size=(n_samples, 2))
        z_correlated = z @ L.T
        x = np.sqrt(df / g)[:, np.newaxis] * z_correlated
        return t.cdf(x, df=df)
    except np.linalg.LinAlgError:
        # 回退到高斯Copula
        return sample_gaussian_copula(n_samples, corr_matrix)

def sample_clayton_copula(n_samples, theta):
    u1 = np.random.uniform(0, 1, n_samples)
    v = np.random.uniform(0, 1, n_samples)
    u2 = (u1**(-theta) * (v**(-theta / (1 + theta)) - 1) + 1)**(-1 / theta)
    return np.column_stack((u1, u2))

def sample_gumbel_copula(n_samples, theta):
    u1 = np.random.uniform(0, 1, n_samples)
    v = np.random.uniform(0, 1, n_samples)
    gamma_rv = np.random.gamma(1/theta, 1, n_samples)
    e1 = -np.log(u1) / gamma_rv
    e2 = -np.log(v) / gamma_rv
    u1_sim = np.exp(-e1**(1/theta))
    u2_sim = np.exp(-e2**(1/theta))
    return np.column_stack((u1_sim, u2_sim))

def sample_survival_gumbel_copula(n_samples, theta):
    """下尾 Gumbel：先采样上尾，再做 (1‑u, 1‑v) 旋转"""
    return 1 - sample_gumbel_copula(n_samples, theta)

def sample_survival_clayton_copula(n_samples, theta):
    """下尾 Clayton，同理旋转"""
    return 1 - sample_clayton_copula(n_samples, theta)
# --- 2. ROBUST PARAMETER ESTIMATION ---



def fit_t_copula_mle(data):
    """Full MLE for bivariate t‑Copula (ρ, ν).  ## MOD"""
    u = data.values

    def neg_ll(params):
        rho, df = params
        # 约束：|ρ|<0.99, ν>2
        if abs(rho) >= .99 or df <= 2.1:
            return np.inf
        c = np.array([[1, rho], [rho, 1]])
        inv_c = np.linalg.inv(c)
        det_c = np.linalg.det(c)
        z = student_t.ppf(u, df=df)
        quad = np.einsum('ij,jk,ik->i', z, inv_c, z)
        ll = -0.5*np.log(det_c) - (df+2)/2*np.log1p(quad/df) \
             + student_t.logpdf(z, df=df).sum(axis=1)
        return -ll.sum()

    kendall_tau = data.corr('kendall').iloc[0,1]
    rho0 = np.sin(np.pi*kendall_tau/2)
    res = minimize(neg_ll, [rho0, 8], bounds=[(-0.95,0.95),(2.1,20)])
    rho_hat, df_hat = res.x
    return {'corr_matrix': np.array([[1, rho_hat],[rho_hat,1]]), 'df': df_hat}

def get_copula_parameters(data, silent=False):
    """Estimates dependence parameters with improved stability."""
    if not silent:
        print("Estimating dependence parameters...")
    
    kendall_tau = data.corr(method='kendall').iloc[0, 1]
    pearson_corr = data.corr(method='pearson').values
    
    # Set reasonable bounds for Clayton and Gumbel parameters
    theta_clayton = max(0.01, 2 * kendall_tau / (1 - kendall_tau)) if (1 - kendall_tau) != 0 else 0.01
    theta_gumbel = max(1.01, 1 / (1 - kendall_tau)) if (1 - kendall_tau) != 0 else 1.01

    t_copula_params = fit_t_copula_mle(data)
    
    params = {
        'Gaussian': {'corr_matrix': pearson_corr},
        'StudentT': t_copula_params,
        'Gumbel': {'theta': theta_gumbel},
        'Clayton': {'theta': theta_clayton}
    }
    
    if not silent:
        param_data = {
            'Gaussian': {'ρ (Pearson)': pearson_corr[0,1]},
            'StudentT': {'ρ (MLE)': t_copula_params['corr_matrix'][0,1], 'ν (DoF, MLE)': t_copula_params['df']},
            'Gumbel': {'θ (from Kendall)': theta_gumbel},
            'Clayton': {'θ (from Kendall)': theta_clayton}
        }
        param_df = pd.DataFrame(param_data).T.fillna('')
        
        print("\n\n" + "="*80)
        print(">>> OUTPUT FOR DISSERTATION: TABLE 4.4 <<<")
        print(param_df.to_markdown(floatfmt=".4f"))
        print("="*80 + "\n")
        
    return params

# --- 3. STABLE GARCH MODELING ---

def fit_garch_model(returns, model_type, max_retries=3):
    """Robust GARCH fitting with enhanced settings."""
    for attempt in range(max_retries):
        try:
            if model_type == "SPX":
                model = arch_model(
                    returns, 
                    mean='Constant',  # 简化均值方程
                    vol='Garch', 
                    p=1, 
                    o=1, 
                    q=1, 
                    dist='t',
                    rescale=False  # 不重新缩放数据
                )
                # 优化器设置
                res = model.fit(
                    disp='off', 
                    options={'maxiter': 1000, 'ftol': 1e-5},
                    update_freq=0
                )
            else:  # EURUSD
                model = arch_model(
                    returns, 
                    mean='Constant',  # 简化均值方程
                    vol='EGARCH', 
                    p=1,  # 减少阶数
                    q=1, 
                    dist='t',
                    rescale=False
                )
                res = model.fit(
                    disp='off', 
                    options={'maxiter': 1000, 'ftol': 1e-5},
                    update_freq=0
                )
            return res
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"GARCH fitting failed after {max_retries} attempts: {e}. Using fallback volatility.")
                # Fallback to historical volatility
                hist_vol = returns.std()
                class FallbackResult:
                    def __init__(self, vol, nu=5):
                        self.params = {'Const': 0, 'nu': nu}
                        self.conditional_volatility = np.array([vol] * len(returns))
                    def forecast(self, *args, **kwargs):
                        class VarianceForecast:
                            def __init__(self, vol):
                                self.variance = pd.DataFrame([vol**2])
                        return VarianceForecast(hist_vol)
                return FallbackResult(hist_vol)
    
    return None

def calculate_dynamic_weights(vol_spx, vol_eurusd):
    """根据波动率动态调整组合权重"""
    # 添加平滑因子防止除零错误
    smoothing = 1e-6
    weight_spx = vol_eurusd / (vol_spx + vol_eurusd + smoothing)
    weight_eurusd = vol_spx / (vol_spx + vol_eurusd + smoothing)
    return weight_spx, weight_eurusd

# --- 4. OPTIMIZED SIMULATION FUNCTION WITH GARCH ---

def run_simulation_for_day(t_index, full_data, copula_params, n_simulations=50000):
    """GARCH-based Monte Carlo simulation with stability improvements."""
    window_data = full_data.loc[:t_index]
    
    # 拟合GARCH模型
    res_spx = fit_garch_model(window_data['SPX_Return'].values * 100, "SPX")
    res_eurusd = fit_garch_model(window_data['EURUSD_Return'].values * 100, "EURUSD")

    # 获取波动率预测
    try:
        sigma_t1_spx = np.sqrt(res_spx.forecast(horizon=1, reindex=False).variance.iloc[0, 0])
    except Exception as e:
        print(f"SPX volatility forecast failed: {e}. Using last conditional vol.")
        sigma_t1_spx = res_spx.conditional_volatility[-1]
    
    try:
        sigma_t1_eurusd = np.sqrt(res_eurusd.forecast(horizon=1, reindex=False).variance.iloc[0, 0])
    except Exception as e:
        print(f"EURUSD volatility forecast failed: {e}. Using last conditional vol.")
        sigma_t1_eurusd = res_eurusd.conditional_volatility[-1]
    
    # 计算动态权重
    rho_today = copula_params['Gaussian']['corr_matrix'][0,1]
    weight_spx, weight_eurusd = calc_min_var_weights(sigma_t1_spx,
                                                     sigma_t1_eurusd,
                                                     rho_today)
    
    # 简化均值预测 - 使用0均值假设提高稳定性
    mu_spx_next = 0
    mu_eurusd_next = 0
    
    daily_forecasts = {}
    samplers = {
        'Gaussian': lambda n, p: sample_gaussian_copula(n, p['corr_matrix']),
        'StudentT': lambda n, p: sample_t_copula(n, p['corr_matrix'], p['df']),
        'Gumbel': lambda n, p: sample_survival_gumbel_copula(n, p['theta']),
        'Clayton': lambda n, p: sample_survival_clayton_copula(n, p['theta'])
    }
    
    # 获取自由度参数 - 添加默认值
    nu_spx = res_spx.params.get('nu', 5) if hasattr(res_spx, 'params') else 5
    nu_eurusd = res_eurusd.params.get('nu', 5) if hasattr(res_eurusd, 'params') else 5
    
    for name, params in copula_params.items():
        try:
            simulated_uniforms = samplers[name](n_simulations, params)
            
            # 添加裁剪避免极端值
            u_spx = np.clip(simulated_uniforms[:, 0], 1e-4, 1-1e-4)
            u_eurusd = np.clip(simulated_uniforms[:, 1], 1e-4, 1-1e-4)
            
            # 使用更稳健的PPF计算
            z_spx = t.ppf(u_spx, df=nu_spx)
            z_eurusd = t.ppf(u_eurusd, df=nu_eurusd)
            
            r_spx_sim = (mu_spx_next + sigma_t1_spx * z_spx) / 100
            r_eurusd_sim = (mu_eurusd_next + sigma_t1_eurusd * z_eurusd) / 100
            
            # 使用动态权重计算组合收益
            r_portfolio_sim = weight_spx * r_spx_sim + weight_eurusd * r_eurusd_sim
            
            # 计算VaR和ES - 添加裁剪避免极端值
            r_portfolio_sim = np.clip(r_portfolio_sim, -0.5, 0.5)  # 限制在±50%范围内
            var_99 = np.percentile(r_portfolio_sim, 1)
            exceed = var_99 - r_portfolio_sim[r_portfolio_sim <= var_99]  # 正值超损
            if exceed.size > 30:                       ## FIX EVT
                shape, loc, scale = stats.genpareto.fit(exceed, floc=0)
                # 理论: ES = VaR + (scale + shape*VaR_exceed_mean)/(1‑shape)
                es_99 = var_99 - (scale + shape*exceed.mean()) / (1 - shape)
            else:
                es_99 = r_portfolio_sim[r_portfolio_sim <= var_99].mean()
            
            daily_forecasts[name] = {
                'VaR_99': var_99, 
                'ES_99': es_99,
                'Weight_SPX': weight_spx,
                'Weight_EURUSD': weight_eurusd,
                'Vol_SPX': sigma_t1_spx,
                'Vol_EURUSD': sigma_t1_eurusd
            }
        except Exception as e:
            print(f"Simulation for {name} failed: {e}. Using fallback values.")
            # 使用简单历史VaR作为回退值
            hist_returns = window_data['SPX_Return'] * weight_spx + window_data['EURUSD_Return'] * weight_eurusd
            var_99_fallback = np.percentile(hist_returns, 1)
            es_99_fallback = hist_returns[hist_returns <= var_99_fallback].mean()
            
            daily_forecasts[name] = {
                'VaR_99': var_99_fallback, 
                'ES_99': es_99_fallback,
                'Weight_SPX': weight_spx,
                'Weight_EURUSD': weight_eurusd,
                'Vol_SPX': sigma_t1_spx,
                'Vol_EURUSD': sigma_t1_eurusd
            }
        
    return daily_forecasts

# --- 5. MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    try:
        print("\n" + "="*80)
        print(">>> GARCH-COPULA MODEL ESTIMATION AND SIMULATION <<<")
        print("="*80 + "\n")
        
        # 加载数据
        copula_input_data = pd.read_csv('copula_input_data.csv', index_col='Date', parse_dates=True).dropna()
        full_data = pd.read_csv('spx_eurusd_daily_data.csv', index_col='Date', parse_dates=True)
        
        # 估计Copula参数（仅使用样本内数据）
        print("Estimating copula parameters using in-sample data...")
        copula_params = get_copula_parameters(copula_input_data)
        print("Copula parameters estimation complete.\n")

        # 设置样本外预测期
        out_of_sample_start = '2020-01-01'
        forecast_dates = full_data.loc[out_of_sample_start:].index
        print(f"Out-of-sample period: {out_of_sample_start} to {forecast_dates[-1].date()}")
        print(f"Number of forecast days: {len(forecast_dates)}\n")
        
        # 进行滚动预测
        all_forecasts = []
        
        def _one_day(day):
            t_idx = full_data.index[full_data.index.get_loc(day) - 1]
            forecasts = run_simulation_for_day(t_idx, full_data,
                                            copula_params,
                                            n_simulations=50000)
            flat = {'Date': day}
            for model, vals in forecasts.items():
                flat[f'{model}_VaR_99']       = vals['VaR_99']
                flat[f'{model}_ES_99']        = vals['ES_99']
                flat[f'{model}_Weight_SPX']   = vals['Weight_SPX']
                flat[f'{model}_Weight_EURUSD']= vals['Weight_EURUSD']
                flat[f'{model}_Vol_SPX']      = vals['Vol_SPX']
                flat[f'{model}_Vol_EURUSD']   = vals['Vol_EURUSD']
            return flat

        # 并行执行；prefer="threads" 可以避免多进程拷贝开销
        all_forecasts = Parallel(n_jobs=8, prefer="threads")(
            delayed(_one_day)(d) for d in tqdm(forecast_dates,
                                            desc="Forecasting VaR/ES"))

        # 保存预测结果
        forecasts_df = pd.DataFrame(all_forecasts).set_index('Date')
        forecast_output_file = 'garch_copula_forecasts.csv'
        forecasts_df.to_csv(forecast_output_file)
        
        print(f"\nAll forecasts saved to '{forecast_output_file}'.")
        print("Forecast summary:")
        print(forecasts_df.head())
        
        # 计算并显示基本统计
        print("\nForecast statistics:")
        print(forecasts_df.describe())
        
        # 保存Copula参数用于后续分析
        with open('copula_params.pkl', 'wb') as f:
            pickle.dump(copula_params, f)
        print("\nCopula parameters saved to 'copula_params.pkl'.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()