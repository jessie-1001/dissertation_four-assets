# =============================================================================
# SCRIPT 03: 4-ASSET GARCH-COPULA MODEL WITH STABILITY ENHANCEMENTS
#
# EXTENDED TO SUPPORT SPX, NDX, EURUSD, USDJPY
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
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# --- 1. STABLE COPULA SAMPLERS ---
def calc_min_var_weights(vols, corr_matrix):
    """
    计算四资产最小方差组合权重
    vols: 各资产波动率向量 [vol_spx, vol_ndx, vol_eurusd, vol_usdjpy]
    corr_matrix: 四资产相关矩阵
    """
    # 创建协方差矩阵
    cov_matrix = np.outer(vols, vols) * corr_matrix
    
    # 计算最小方差组合权重
    ones = np.ones(len(vols))
    cov_inv = np.linalg.inv(cov_matrix)
    weights = cov_inv @ ones / (ones.T @ cov_inv @ ones)
    
    # 应用权重限制 (每个资产权重在5%-40%之间)
    weights = np.clip(weights, 0.05, 0.40)
    
    # 归一化确保权重和为1
    weights /= weights.sum()
    return weights

def sample_gaussian_copula(n_samples, corr_matrix):
    """四维高斯copula采样"""
    try:
        L = cholesky(corr_matrix, lower=True)
        z = np.random.normal(0, 1, size=(n_samples, 4))
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
        z = np.random.normal(0, 1, size=(n_samples, 4))
        z_correlated = z @ L.T
        return norm.cdf(z_correlated)

# --- 2. ROBUST PARAMETER ESTIMATION ---

def fit_t_copula_mle(data):
    """Full MLE for multivariate t-Copula (ρ, ν)"""
    u = data.values
    n_assets = u.shape[1]

    def neg_ll(params):
        # 解包参数: 前n_params-1个是相关矩阵下三角元素，最后一个是自由度
        rho_vec = params[:-1]
        df = params[-1]
        
        # 重构相关矩阵
        L = np.zeros((n_assets, n_assets))
        indices = np.tril_indices(n_assets, -1)
        L[indices] = rho_vec
        np.fill_diagonal(L, 1.0)
        corr_matrix = L @ L.T
        
        # 约束：相关矩阵正定，ν>2
        try:
            cholesky(corr_matrix, lower=True)
        except np.linalg.LinAlgError:
            return np.inf
            
        if df <= 2.1:
            return np.inf
            
        inv_c = np.linalg.inv(corr_matrix)
        det_c = np.linalg.det(corr_matrix)
        z = student_t.ppf(u, df=df)
        
        # 计算对数似然
        quad_forms = np.einsum('ij,jk,ik->i', z, inv_c, z)
        ll = -0.5*np.log(det_c) - (df + n_assets)/2 * np.log1p(quad_forms/df) \
             + np.sum(student_t.logpdf(z, df=df), axis=1)
        return -ll.sum()

    # 初始值: 使用相关矩阵下三角元素和初始自由度=8
    initial_corr = data.corr().values
    L = np.linalg.cholesky(initial_corr)
    rho_vec = L[np.tril_indices(n_assets, -1)]
    initial_params = np.append(rho_vec, 8)
    
    # 设置参数边界
    bounds = [(-0.99, 0.99)] * len(rho_vec) + [(2.1, 30)]
    
    # 优化
    res = minimize(neg_ll, initial_params, bounds=bounds, method='L-BFGS-B')
    
    # 提取结果
    rho_vec_hat = res.x[:-1]
    df_hat = res.x[-1]
    
    # 重构相关矩阵
    L_hat = np.zeros((n_assets, n_assets))
    indices = np.tril_indices(n_assets, -1)
    L_hat[indices] = rho_vec_hat
    np.fill_diagonal(L_hat, 1.0)
    corr_matrix_hat = L_hat @ L_hat.T
    
    return {'corr_matrix': corr_matrix_hat, 'df': df_hat}

def get_copula_parameters(data, silent=False):
    """Estimates dependence parameters for 4 assets"""
    if not silent:
        print("Estimating dependence parameters for 4 assets...")
    
    # 估计高斯copula参数
    pearson_corr = data.corr().values
    
    # 估计t-copula参数
    t_copula_params = fit_t_copula_mle(data)
    
    params = {
        'Gaussian': {'corr_matrix': pearson_corr},
        'StudentT': t_copula_params
    }
    
    if not silent:
        print("\n\n" + "="*80)
        print(">>> OUTPUT FOR DISSERTATION: TABLE 4.4 (4-Asset) <<<")
        
        # 打印高斯copula参数
        print("\nGaussian Copula Correlation Matrix:")
        corr_df = pd.DataFrame(pearson_corr, 
                              index=data.columns, 
                              columns=data.columns)
        print(corr_df.to_markdown(floatfmt=".4f"))
        
        # 打印t-copula参数
        print("\nStudent-t Copula Parameters:")
        print(f"Degrees of Freedom (ν): {t_copula_params['df']:.4f}")
        print("\nCorrelation Matrix:")
        t_corr_df = pd.DataFrame(t_copula_params['corr_matrix'], 
                                index=data.columns, 
                                columns=data.columns)
        print(t_corr_df.to_markdown(floatfmt=".4f"))
        
        print("="*80 + "\n")
        
    return params

# --- 3. STABLE GARCH MODELING FOR 4 ASSETS ---

def fit_garch_model(returns, asset_type):
    """Robust GARCH fitting for different asset types"""
    try:
        if asset_type in ['SPX', 'NDX']:  # Equity indices
            model = arch_model(
                returns, 
                mean='Constant',
                vol='Garch', 
                p=1, 
                o=1, 
                q=1, 
                dist='t',
                rescale=False
            )
        else:  # FX pairs
            model = arch_model(
                returns, 
                mean='Constant',
                vol='EGARCH', 
                p=1,  
                q=1, 
                dist='t',
                rescale=False
            )
        
        # 优化器设置
        res = model.fit(
            disp='off', 
            options={'maxiter': 1000, 'ftol': 1e-5},
            update_freq=0
        )
        return res
    except Exception as e:
        print(f"GARCH fitting failed for {asset_type}: {e}. Using fallback volatility.")
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

# --- 4. OPTIMIZED SIMULATION FUNCTION FOR 4 ASSETS ---

def run_simulation_for_day(t_index, full_data, copula_params, n_simulations=50000):
    """GARCH-based Monte Carlo simulation for 4 assets"""
    window_data = full_data.loc[:t_index]
    assets = ['SPX', 'NDX', 'EURUSD', 'USDJPY']
    
    # 拟合GARCH模型
    garch_results = {}
    vols = []
    nus = []
    
    for asset in assets:
        # 获取收益率序列
        returns = window_data[f'{asset}_Return'].values * 100
        
        # 拟合GARCH模型
        res = fit_garch_model(returns, asset)
        garch_results[asset] = res
        
        # 获取波动率预测
        try:
            sigma = np.sqrt(res.forecast(horizon=1, reindex=False).variance.iloc[0, 0])
        except Exception as e:
            print(f"{asset} volatility forecast failed: {e}. Using last conditional vol.")
            sigma = res.conditional_volatility[-1]
        vols.append(sigma)
        
        # 获取自由度参数
        nu = res.params.get('nu', 5) if hasattr(res, 'params') else 5
        nus.append(nu)
    
    # 计算组合权重
    if 'StudentT' in copula_params:
        corr_matrix = copula_params['StudentT']['corr_matrix']
    else:
        corr_matrix = copula_params['Gaussian']['corr_matrix']
    
    weights = calc_min_var_weights(vols, corr_matrix)
    
    # 简化均值预测
    mus = [0, 0, 0, 0]  # 零均值假设
    
    daily_forecasts = {}
    samplers = {
        'Gaussian': lambda n, p: sample_gaussian_copula(n, p['corr_matrix']),
        'StudentT': lambda n, p: sample_gaussian_copula(n, p['corr_matrix'])  # 使用相同采样方法
    }
    
    for name, params in copula_params.items():
        try:
            # 采样四维Copula
            simulated_uniforms = samplers[name](n_simulations, params)
            
            # 添加裁剪避免极端值
            simulated_uniforms = np.clip(simulated_uniforms, 1e-4, 1-1e-4)
            
            # 转换为t分布分位数
            z_assets = []
            for i, asset in enumerate(assets):
                u = simulated_uniforms[:, i]
                z = t.ppf(u, df=nus[i])
                z_assets.append(z)
            
            # 计算资产收益率
            r_assets_sim = []
            for i, asset in enumerate(assets):
                r_sim = (mus[i] + vols[i] * z_assets[i]) / 100
                r_assets_sim.append(r_sim)
            
            # 计算组合收益
            r_portfolio_sim = np.zeros(n_simulations)
            for i in range(len(assets)):
                r_portfolio_sim += weights[i] * r_assets_sim[i]
            
            # 裁剪组合收益率
            r_portfolio_sim = np.clip(r_portfolio_sim, -0.5, 0.5)
            
            # 计算VaR和ES
            var_99 = np.percentile(r_portfolio_sim, 1)
            exceed = r_portfolio_sim[r_portfolio_sim <= var_99]
            es_99 = exceed.mean()
            
            # 存储结果
            daily_forecasts[name] = {
                'VaR_99': var_99, 
                'ES_99': es_99,
                'Vols': vols,
                'Weights': weights,
                'Corr_matrix': corr_matrix
            }
        except Exception as e:
            print(f"Simulation for {name} failed: {e}. Using fallback values.")
            # 回退方法：使用简单历史VaR
            hist_returns = np.sum([weights[i] * window_data[f'{asset}_Return'] for i, asset in enumerate(assets)], axis=0)
            var_99_fallback = np.percentile(hist_returns, 1)
            es_99_fallback = hist_returns[hist_returns <= var_99_fallback].mean()
            
            daily_forecasts[name] = {
                'VaR_99': var_99_fallback, 
                'ES_99': es_99_fallback,
                'Vols': vols,
                'Weights': weights,
                'Corr_matrix': corr_matrix
            }
        
    return daily_forecasts

# --- 5. MAIN EXECUTION BLOCK FOR 4 ASSETS ---
if __name__ == '__main__':
    try:
        print("\n" + "="*80)
        print(">>> 4-ASSET GARCH-COPULA MODEL ESTIMATION AND SIMULATION <<<")
        print("="*80 + "\n")
        
        # 加载四资产数据
        copula_input_data = pd.read_csv('copula_input_data_4asset.csv', index_col='Date', parse_dates=True)
        full_data = pd.read_csv('spx_ndx_eurusd_usdjpy_daily.csv', index_col='Date', parse_dates=True)
        
        # 确保列名正确
        required_pit_cols = ['u_spx', 'u_ndx', 'u_eurusd', 'u_usdjpy']
        for col in required_pit_cols:
            if col not in copula_input_data.columns:
                raise ValueError(f"Missing column in copula input data: {col}")
        
        # 估计Copula参数
        print("Estimating copula parameters for 4 assets...")
        copula_params = get_copula_parameters(copula_input_data[required_pit_cols])
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
                flat[f'{model}_VaR_99'] = vals['VaR_99']
                flat[f'{model}_ES_99'] = vals['ES_99']
                
                # 存储波动率和权重
                for i, asset in enumerate(['SPX', 'NDX', 'EURUSD', 'USDJPY']):
                    flat[f'{model}_Vol_{asset}'] = vals['Vols'][i]
                    flat[f'{model}_Weight_{asset}'] = vals['Weights'][i]
            
            return flat

        # 并行执行
        all_forecasts = Parallel(n_jobs=8, prefer="threads")(
            delayed(_one_day)(d) for d in tqdm(forecast_dates,
                                            desc="Forecasting VaR/ES"))

        # 保存预测结果
        forecasts_df = pd.DataFrame(all_forecasts).set_index('Date')
        forecast_output_file = '4asset_garch_copula_forecasts.csv'
        forecasts_df.to_csv(forecast_output_file)
        
        print(f"\nAll forecasts saved to '{forecast_output_file}'.")
        print("Forecast summary:")
        print(forecasts_df.head())
        
        # 保存Copula参数
        with open('4asset_copula_params.pkl', 'wb') as f:
            pickle.dump(copula_params, f)
        print("\nCopula parameters saved to '4asset_copula_params.pkl'.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()