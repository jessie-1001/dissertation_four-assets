# =============================================================================
# SCRIPT 03: 4-ASSET GARCH-COPULA MODEL WITH STABILITY ENHANCEMENTS
#
# MODIFIED TO MATCH PREVIOUS OUTPUT FORMAT AND FIX 'Date' ERROR
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
from scipy import stats
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# --- 1. STABLE COPULA SAMPLERS ---
def calc_min_var_weights(vols, corr_matrix):
    """计算四资产最小方差组合权重"""
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
        # 特征值修正
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
    """Full MLE for multivariate t-Copula"""
    u = data.values
    n_assets = u.shape[1]

    def neg_ll(params):
        rho_vec = params[:-1]
        df = params[-1]
        
        # 重构相关矩阵
        L = np.zeros((n_assets, n_assets))
        indices = np.tril_indices(n_assets, -1)
        L[indices] = rho_vec
        np.fill_diagonal(L, 1.0)
        corr_matrix = L @ L.T
        
        try:
            cholesky(corr_matrix, lower=True)
        except np.linalg.LinAlgError:
            return np.inf
            
        if df <= 2.1:
            return np.inf
            
        inv_c = np.linalg.inv(corr_matrix)
        det_c = np.linalg.det(corr_matrix)
        z = student_t.ppf(u, df=df)
        
        quad_forms = np.einsum('ij,jk,ik->i', z, inv_c, z)
        ll = -0.5*np.log(det_c) - (df + n_assets)/2 * np.log1p(quad_forms/df) \
             + np.sum(student_t.logpdf(z, df=df), axis=1)
        return -ll.sum()
    
    # 初始值
    initial_corr = data.corr().values
    L = np.linalg.cholesky(initial_corr)
    rho_vec = L[np.tril_indices(n_assets, -1)]
    initial_params = np.append(rho_vec, 8)
    
    bounds = [(-0.99, 0.99)] * len(rho_vec) + [(2.1, 30)]
    
    res = minimize(neg_ll, initial_params, bounds=bounds, method='L-BFGS-B')
    
    rho_vec_hat = res.x[:-1]
    df_hat = res.x[-1]
    
    L_hat = np.zeros((n_assets, n_assets))
    indices = np.tril_indices(n_assets, -1)
    L_hat[indices] = rho_vec_hat
    np.fill_diagonal(L_hat, 1.0)
    corr_matrix_hat = L_hat @ L_hat.T
    
    d = np.sqrt(np.diag(corr_matrix_hat))
    corr_matrix_hat = corr_matrix_hat / d[:, None] / d[None, :]

    return {'corr_matrix': corr_matrix_hat, 'df': df_hat}

def get_copula_parameters(data, silent=False):
    """Estimates dependence parameters for 4 assets"""
    if not silent:
        print("Estimating dependence parameters for 4 assets...")
    
    # 重命名列以匹配前序输出
    data = data.rename(columns={
        'u_spx': 'SPX', 
        'u_ndx': 'NDX',
        'u_eurusd': 'EURUSD',
        'u_usdjpy': 'USDJPY'
    })
    
    pearson_corr = data.corr().values
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
                             index=['SPX', 'NDX', 'EURUSD', 'USDJPY'], 
                             columns=['SPX', 'NDX', 'EURUSD', 'USDJPY'])
        print(corr_df.to_markdown(floatfmt=".4f"))
        
        # 打印t-copula参数
        print("\nStudent-t Copula Parameters:")
        print(f"Degrees of Freedom (ν): {t_copula_params['df']:.4f}")
        print("\nCorrelation Matrix:")
        t_corr_df = pd.DataFrame(t_copula_params['corr_matrix'], 
                               index=['SPX', 'NDX', 'EURUSD', 'USDJPY'], 
                               columns=['SPX', 'NDX', 'EURUSD', 'USDJPY'])
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
                mean='AR', lags=1,
                vol='Garch', 
                p=1, 
                o=1, 
                q=1, 
                dist='t'
            )
        else:  # FX pairs
            model = arch_model(
                returns, 
                mean='AR', lags=1,
                vol='EGARCH', 
                p=2,  
                q=1, 
                dist='skewt'
            )
        
        res = model.fit(disp='off', update_freq=0)
        return res
    except Exception as e:
        print(f"GARCH fitting failed for {asset_type}: {e}. Using fallback.")
        hist_vol = returns.std()
        class FallbackResult:
            def __init__(self, vol):
                self.conditional_volatility = np.array([vol] * len(returns))
                self.params = {'nu': 5} if asset_type in ['SPX', 'NDX'] else {'eta': 5, 'lambda': 0}
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
    dist_params = []
    
    for asset in assets:
        returns = window_data[f'{asset}_Return'].values * 100
        res = fit_garch_model(returns, asset)
        garch_results[asset] = res
        
        try:
            sigma = np.sqrt(res.forecast(horizon=1).variance.iloc[-1,0])
        except:
            sigma = res.conditional_volatility[-1]
        vols.append(sigma)
        
        if asset in ['SPX', 'NDX']:
            dist_params.append(res.params['nu'])
        else:
            dist_params.append((res.params['eta'], res.params['lambda']))
    
    # 计算组合权重
    corr_matrix = copula_params['StudentT']['corr_matrix']
    weights = calc_min_var_weights(vols, corr_matrix)
    
    daily_forecasts = {}
    
    for name, params in copula_params.items():
        try:
            # 采样四维Copula
            u = sample_gaussian_copula(n_simulations, params['corr_matrix'])
            u = np.clip(u, 1e-4, 1-1e-4)
            
            # 转换为分位数
            z = []
            for i, asset in enumerate(assets):
                if asset in ['SPX', 'NDX']:
                    z.append(t.ppf(u[:,i], df=dist_params[i]))
                else:
                    eta, lam = dist_params[i]
                    delta = lam / np.sqrt(1 + lam**2)
                    z_adj = np.where(
                        u[:,i] < 0.5,
                        (1 - delta) * t.ppf(u[:,i]/(1 + lam**2), df=eta),
                        (1 + delta) * t.ppf(1 - (1 - u[:,i])/(1 + lam**2), df=eta)
                    )
                    z.append(z_adj)
            
            # 计算收益率
            r_sim = []
            for i in range(4):
                r_sim.append(vols[i] * z[i] / 100)
            
            # 组合收益率
            r_port = np.sum([w*r for w,r in zip(weights, r_sim)], axis=0)
            r_port = np.clip(r_port, -0.5, 0.5)
            
            # 计算风险指标
            var_99 = np.percentile(r_port, 1)
            exceed = r_port[r_port <= var_99]
            es_99 = exceed.mean()
            
            daily_forecasts[name] = {
                'VaR_99': var_99,
                'ES_99': es_99,
                'Weights': weights,
                'Vols': vols
            }
        except Exception as e:
            print(f"Simulation failed for {name}: {e}")
            daily_forecasts[name] = {
                'VaR_99': np.nan,
                'ES_99': np.nan,
                'Weights': weights,
                'Vols': vols
            }
    
    return daily_forecasts

# --- 5. MAIN EXECUTION BLOCK FOR 4 ASSETS ---
if __name__ == '__main__':
    try:
        print("\n" + "="*80)
        print(">>> 4-ASSET GARCH-COPULA MODEL ESTIMATION AND SIMULATION <<<")
        print("="*80 + "\n")
        
        # 加载数据 - 修复Date列问题
        pit_data = pd.read_csv('copula_input_data_4asset.csv')
        # 检查并处理日期列
        date_col = None
        for col in pit_data.columns:
            if col.lower() == 'date':
                date_col = col
                break
        
        if date_col is None:
            # 如果没有找到日期列，尝试使用第一列作为索引
            pit_data = pit_data.set_index(pit_data.columns[0])
            print(f"Using first column '{pit_data.index.name}' as index")
        else:
            pit_data = pit_data.set_index(date_col)
        
        # 确保索引是日期时间类型
        try:
            pit_data.index = pd.to_datetime(pit_data.index)
        except Exception as e:
            print(f"Error converting index to datetime: {e}")
            # 如果转换失败，尝试解析列中的日期
            pit_data['Date'] = pd.to_datetime(pit_data.index)
            pit_data = pit_data.set_index('Date')
        
        # 加载完整数据（类似处理）
        full_data = pd.read_csv('spx_ndx_eurusd_usdjpy_daily.csv')
        date_col = None
        for col in full_data.columns:
            if col.lower() == 'date':
                date_col = col
                break
        
        if date_col is None:
            full_data = full_data.set_index(full_data.columns[0])
            print(f"Using first column '{full_data.index.name}' as index")
        else:
            full_data = full_data.set_index(date_col)
        
        try:
            full_data.index = pd.to_datetime(full_data.index)
        except Exception as e:
            print(f"Error converting index to datetime: {e}")
            full_data['Date'] = pd.to_datetime(full_data.index)
            full_data = full_data.set_index('Date')
        
        # 打印验证信息：
        print("\nData verification:")
        print("PIT data sample:")
        print(pit_data.head())
        print("\nFull data sample:")
        print(full_data.head())
        print(f"\nPIT data date range: {pit_data.index.min()} to {pit_data.index.max()}")
        print(f"Full data date range: {full_data.index.min()} to {full_data.index.max()}")

        # 估计Copula参数
        copula_params = get_copula_parameters(pit_data[['u_spx', 'u_ndx', 'u_eurusd', 'u_usdjpy']])
        
        # 样本外预测期
        out_of_sample_start = '2020-01-01'
        forecast_dates = pd.to_datetime(full_data.loc[out_of_sample_start:].index)
        
        # 并行计算
        def process_day(day):
            t_idx = full_data.index[full_data.index.get_loc(day) - 1]
            forecasts = run_simulation_for_day(t_idx, full_data, copula_params)
            result = {'Date': day}
            for model, vals in forecasts.items():
                result.update({
                    f'{model}_VaR_99': vals['VaR_99'],
                    f'{model}_ES_99': vals['ES_99'],
                    f'{model}_Weight_SPX': vals['Weights'][0],
                    f'{model}_Weight_NDX': vals['Weights'][1],
                    f'{model}_Weight_EURUSD': vals['Weights'][2],
                    f'{model}_Weight_USDJPY': vals['Weights'][3],
                    f'{model}_Vol_SPX': vals['Vols'][0],
                    f'{model}_Vol_NDX': vals['Vols'][1],
                    f'{model}_Vol_EURUSD': vals['Vols'][2],
                    f'{model}_Vol_USDJPY': vals['Vols'][3]
                })
            return result
        
        results = Parallel(n_jobs=4)(delayed(process_day)(day) for day in tqdm(forecast_dates))
        
        # 保存结果
        forecasts_df = pd.DataFrame(results).set_index('Date')
        forecasts_df.to_csv('4asset_garch_copula_forecasts.csv')
        
        print("\nForecast summary:")
        print(forecasts_df.describe())
        
        with open('4asset_copula_params.pkl', 'wb') as f:
            pickle.dump(copula_params, f)
            
        print("\nCompleted successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()