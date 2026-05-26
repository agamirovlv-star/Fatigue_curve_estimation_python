import numpy as np
from scipy.optimize import curve_fit
from typing import List, Tuple, Dict

# ===========================================================
# МОДЕЛИ КРИВОЙ УСТАЛОСТИ (со штрафами)
# ===========================================================

def sn_curve_model(sigma: np.ndarray, sigma_inf: float, C: float, m: float, 
                   ku: int, cx_min: float) -> np.ndarray:
    if sigma_inf < 0 or sigma_inf > cx_min - 1.0 or m < 0.0 or m > 10.0 or C < 0:
        return np.full_like(sigma, 1e20)
    sigma_adj = sigma - sigma_inf
    sigma_adj = np.maximum(sigma_adj, 1e-6)
    if ku == 0:
        # Модель для ku=0: возвращает N (долговечность в циклах)
        return 10 ** ((np.log10(C) - np.log10(sigma_adj)) / m)
    else:
        # Модель для ku=1: возвращает lgN
        return (np.log10(C) - np.log10(sigma_adj)) / m


def sigma_from_lgN(lgN: float, sigma_inf: float, C: float, m: float, ku: int) -> float:
    """вычисление напряжения по долговечности"""
    if ku == 0:
        return sigma_inf + C * (lgN) ** (-m)
    else:
        return sigma_inf + C * (10 ** lgN) ** (-m)


# ===========================================================
# НАЧАЛЬНЫЕ ПАРАМЕТРЫ
# ===========================================================

def compute_initial_parameters(cx: List[float], lgN: List[float], 
                                sigma_inf_guess: float, ku: int) -> Tuple[float, float]:
    s1 = np.log10(cx[0] - sigma_inf_guess)
    s2 = np.log10(cx[1] - sigma_inf_guess)
    if ku == 0:
        m_init = (s1 - s2) / (np.log10(lgN[1]) - np.log10(lgN[0]))
        logC_init = s1 + m_init * np.log10(lgN[0])
    else:
        m_init = (s1 - s2) / (lgN[1] - lgN[0])
        logC_init = s1 + m_init * lgN[0]
    return logC_init, m_init


# ====================== ТЕОРЕТИЧЕСКАЯ ОЦЕНКА ОШИБКИ ======================

def theoretical_std_sigma_R(sigma_inf: float, C: float, m: float, ku: int,
                            cx: List[float], lgN_levels: List[float], 
                            w: List[float], ni: List[int], N0: float) -> float:
    """
    Расчет теоретической стандартной ошибки предела выносливости sigma_R
    
    Параметры:
        sigma_inf, C, m - параметры кривой усталости
        ku - тип модели (0 или 1)
        cx - экспериментальные напряжения
        lgN_levels - экспериментальные логарифмы долговечностей
        slgN - среднеквадратичные отклонения lgN
        ni - количество образцов на каждом уровне
        N0 - базовая долговечность
    """
    n_i = np.array(ni)
    sigma_levels = np.array(cx)
    lgN_levels = np.array(lgN_levels)
    weights = np.array(w)
    
    sigma_adj = sigma_levels - sigma_inf
    sigma_adj = np.maximum(sigma_adj, 1e-6)
    ln10 = np.log(10)
    
    # Производные для матрицы чувствительности
    if ku == 0:
        # Для модели: lgN = (sigma - sigma_inf)/C)^(-1 / m)
        # Производная по sigma_inf
        d_dsigma_inf = (1.0 / (m * C)) * ((sigma_adj / C) ** (-1.0/m - 1.0))
        d_dC = (1.0 / (m * C)) * ((sigma_adj / C) ** (-1.0/m))
        d_dm = ((sigma_adj / C) ** (-1.0/m)) * np.log(sigma_adj / C) * (1.0 / m**2)
    else:
        # Для модели: lgN =(log10(C) - log10(sigma - sigma_inf)) / m
        # Производная по sigma_inf
        d_dsigma_inf = (1.0 / m) * (1.0 / (ln10 * sigma_adj))
        # Производная по C
        d_dC = (1.0 / m) * (1.0 / (ln10 * C))*np.ones_like(sigma_levels)
        # Производная по m
        d_dm = -(1.0 / m ** 2) * (np.log10(C) - np.log10(sigma_adj))
    
    J = np.column_stack([d_dsigma_inf, d_dC, d_dm])
    
    # Информационная матрица Фишера
    Fisher = J.T @ np.diag(weights) @ J
    
    if np.linalg.cond(Fisher) > 1e20: return 0.0
    
    try:
        Cov = np.linalg.inv(Fisher)
    except np.linalg.LinAlgError:
        return 0.0
    
    # Производные sigma_R по параметрам
    if ku == 0:
        # sigma_R = sigma_inf + C * N0^(-m)
        term = N0 ** (-m)
        dR_dsigma_inf = 1.0
        dR_dC = term
        dR_dm = -C * term * np.log(N0)
    else:
        # sigma_R = sigma_inf + C * (10^N0)^(-m) = sigma_inf + C * 10^(-m*N0)
        term = 10 ** (-m * N0)
        dR_dsigma_inf = 1.0
        dR_dC = term
        dR_dm = -C * term * np.log(10) * N0
    
    dR = np.array([dR_dsigma_inf, dR_dC, dR_dm])
    
    var_sigma_R = dR @ Cov @ dR
    return np.sqrt(max(var_sigma_R, 0.0))

# ===========================================================
# ФУНКЦИЯ ДЛЯ curve_fit (обертка с фиксированными ku и cx_min)
# ===========================================================

def make_model_func(ku: int, cx_min: float):
    def model_func(sigma, sigma_inf, C, m):
        return sn_curve_model(sigma, sigma_inf, C, m, ku, cx_min)
    return model_func


# ===========================================================
# ОСНОВНАЯ ФУНКЦИЯ ОЦЕНКИ (method='lm')
# ===========================================================

def estimate_fatigue_curve(cx: List[float], ni: List[int], 
                           lgN: List[float], w: List[float],
                           ku: int) -> Dict:
    
    cx = np.array(cx)
    lgN = np.array(lgN)
    w = np.array(w)

    sigma_inf_guess = 0.5 * cx[-1]
    logC_init, m_init = compute_initial_parameters(cx, lgN, sigma_inf_guess, ku)
    C_guess = 10 ** logC_init
    
    # Создаем модель с фиксированными ku
    cx_min = cx[-1]
    model_func = make_model_func(ku, cx_min)
    
    # Метод 'lm' не поддерживает bounds, используем только p0 и maxfev
    # Штрафы уже внутри model_func
    popt, pcov, infodict, mesg, ier = curve_fit(
        model_func, cx, lgN,
        p0=[sigma_inf_guess, C_guess, m_init],
        maxfev=10000,
        ftol=1e-12,
        xtol=1e-12,
        method='lm',
        full_output=True
    )
    
    sigma_inf_est, C_est, m_est = popt
    log10C_est = np.log10(C_est)
    
    # Предсказанные значения и невязки
    y_pred = model_func(cx, sigma_inf_est, C_est, m_est)
    residuals = y_pred - lgN
    Q = np.sum(w * residuals ** 2) / np.sum(w)
    
    # Стандартные ошибки
    se_sigma_inf = np.sqrt(max(pcov[0, 0], 0))
    se_C = np.sqrt(max(pcov[1, 1], 0))
    se_m = np.sqrt(max(pcov[2, 2], 0))
    
    # Расчетные напряжения для сравнения
    sigma_calc_list = []
    for i in range(len(cx)):
        sigma_calc_list.append(sigma_from_lgN(lgN[i], sigma_inf_est, C_est, m_est, ku))
    
    # Количество вызовов функции (nfev)
    nfev = infodict.get('nfev', 'N/A')
    
    return {
        'success': ier in [1, 2, 3, 4],
        'ku': ku,
        'sigma_inf': sigma_inf_est,
        'C': C_est,
        'm': m_est,
        'log10C': log10C_est,
        'Q': Q,
        'covariance': pcov,
        'std_errors': [se_sigma_inf, se_C, se_m],
        'residuals': residuals,
        'initial_params': {
            'sigma_inf': sigma_inf_guess,
            'C': C_guess,
            'm': m_init,
            'log10C': logC_init
        },
        'y_pred': y_pred,
        'sigma_calc': sigma_calc_list,
        'nfev': nfev,
        'ier': ier,
        'mesg': mesg
    }