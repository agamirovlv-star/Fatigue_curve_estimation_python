"""
Модуль для расчета параметров кривой усталости
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import List, Tuple, Dict

# ===========================================================
# МОДЕЛИ КРИВОЙ УСТАЛОСТИ (со штрафами)
# ===========================================================

def sn_curve_model(sigma: np.ndarray, sigma_inf: float, C: float, m: float, 
                   ku: int, cx_min: float) -> np.ndarray:
    if sigma_inf < 0 or sigma_inf > cx_min - 1.0 or m < 0.0 or m > 10.0 or C<0:
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
        return sigma_inf + C * (lgN)**(-m)
    else:
        return sigma_inf + C * (10**lgN)**(-m)

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
    
    cx=np.array(cx)
    lgN=np.array(lgN)
    w=np.array(w)

    sigma_inf_guess = 0.5 * cx[-1]
    logC_init, m_init = compute_initial_parameters(cx,lgN,sigma_inf_guess,ku)
    C_guess = 10**logC_init
    
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
    Q = np.sum(w*residuals**2)/np.sum(w)
    
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
    # Количество итераций для 'lm' можно оценить, но явного поля нет
    
    return {
        'success': ier in [1, 2, 3, 4],  # как в C++: 1-4 успех
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


def estimate_sigma_for_N0(N0_list: List[float], sigma_inf: float, C: float, m: float, ku: int) -> List[float]:
    results = []
    for N0 in N0_list:
        results.append(sigma_from_lgN(N0, sigma_inf, C, m, ku))
    return results