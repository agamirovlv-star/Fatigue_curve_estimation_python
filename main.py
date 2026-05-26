import numpy as np
import json
import sys
from datetime import datetime
from typing import List
from dataclasses import dataclass
from fatigue_calculator import estimate_fatigue_curve, theoretical_std_sigma_R

# ================================================================

@dataclass
class MaterialParams:
    cx: List[float]
    lgN: List[float]
    slgN: List[float]
    ni: List[int]
    N0: List[float]
    ku: int

    @classmethod
    def from_dict(cls, data: dict) -> 'MaterialParams':
        return cls(
            cx=data['cx'],
            lgN=data['lgN'],
            slgN=data['slgN'],
            ni=data['ni'],
            N0=data['N0'],
            ku=data['ku']
        )

# ================================================================

class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def __del__(self):
        self.file.close()

# ================================================================

def compute_weights(ni: List[int], lgN: List[float], slgN: List[float], ku: int) -> List[float]:
    w = []
    for i in range(len(ni)): w.append(ni[i] / (slgN[i]) ** 2)
        #if ku == 0:
        #    w.append(ni[i] * (lgN[i] / slgN[i]) ** 2)
        #else:
        #    w.append(ni[i] / (slgN[i]) ** 2)
    return w


def main():
    tee = Tee('fatigue.out')
    sys.stdout = tee

    print("=" * 80)
    print("ОЦЕНКА ПАРАМЕТРОВ КРИВОЙ УСТАЛОСТИ (method='lm')")
    print("=" * 80)

    with open('fatigue.json', 'r', encoding='utf-8') as f:
        params_dict = json.load(f)
    params = MaterialParams.from_dict(params_dict)
    
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Уровней: {len(params.cx)}")
    print(f"Образцов: {sum(params.ni)}")
    print()

    results = {}

    # ========== РАСЧЕТ ДЛЯ ku=0 ==========
    if params.ku == 0 or params.ku == 2:
        print("\n" + "=" * 80)
        print("РЕЗУЛЬТАТ ДЛЯ ku=0 (sigma = sigma_inf + C * (lgN)^(-m))")
        print("=" * 80)
        w0 = compute_weights(params.ni, params.lgN, params.slgN, 0)
        results[0] = estimate_fatigue_curve(params.cx, params.ni, params.lgN, w0, 0)
        res = results[0]
        print("ВЕСА:")
        for i, weight in enumerate(w0):
            print(f"  Weight[{i}] = {weight:.4f}")
        print()
        
        print("НАЧАЛЬНЫЕ ПАРАМЕТРЫ:")
        print(f"  sigma_inf = {res['initial_params']['sigma_inf']:.4f} МПа (0.5 * σ_min)")
        print(f"  log10(C)  = {res['initial_params']['log10C']:.4f}")
        print(f"  C         = {res['initial_params']['C']:.4f}")
        print(f"  m         = {res['initial_params']['m']:.6f}")
        print()
        
        print("СТАТУС ОПТИМИЗАЦИИ:")
        if res['success']:
            print(f"  УСПЕШНО (ier={res['ier']}: {res['mesg']})")
        else:
            print(f"  НЕ УСПЕШНО (ier={res['ier']}: {res['mesg']})")
        print(f"  Количество вызовов функции (nfev) = {res['nfev']}")
        print()
        
        print("ФИНАЛЬНЫЕ ПАРАМЕТРЫ:")
        print(f"  sigma_inf = {res['sigma_inf']:.7f} МПа")
        print(f"  log10(C)  = {res['log10C']:.7f}")
        print(f"  C         = {res['C']:.7f}")
        print(f"  m         = {res['m']:.7f}")
        print(f"  Q         = {res['Q']:.7f}")
        print()
        
        print("СТАНДАРТНЫЕ ОШИБКИ ПАРАМЕТРОВ:")
        print(f"  se(sigma_inf) = {res['std_errors'][0]:.6e}")
        print(f"  se(C)         = {res['std_errors'][1]:.6e}")
        print(f"  se(m)         = {res['std_errors'][2]:.6e}")
        print()
        
        print("КОВАРИАЦИОННАЯ МАТРИЦА:")
        print("       [sigma_inf        C            m     ]")
        for i in range(3):
            print(f"      [{res['covariance'][i,0]:12.6e} {res['covariance'][i,1]:12.6e} {res['covariance'][i,2]:12.6e}]")
        print()
        
        print("СРАВНЕНИЕ:")
        print(f"{'i':<3} {'σ(МПа)':>10} {'lgN_exp':>12} {'lgN_calc':>12} {'σ_calc(МПа)':>14} {'Невязка':>12}")
        print("-" * 70)
        for i in range(len(params.cx)):
            print(f"{i+1:<3} {params.cx[i]:10.1f} {params.lgN[i]:12.6f} {res['y_pred'][i]:12.6f} {res['sigma_calc'][i]:14.2f} {res['residuals'][i]:12.6e}")
        print()

    # ========== РАСЧЕТ ДЛЯ ku=1 ==========
    if params.ku == 1 or params.ku == 2:
        print("\n" + "=" * 80)
        print("РЕЗУЛЬТАТ ДЛЯ ku=1 (sigma = sigma_inf + C * N^(-m))")
        print("=" * 80)
        w1 = compute_weights(params.ni, params.lgN, params.slgN, 1)
        results[1] = estimate_fatigue_curve(params.cx, params.ni, params.lgN, w1, 1)
        res = results[1]
        print("ВЕСА:")
        for i, weight in enumerate(w1):
            print(f"  Weight[{i}] = {weight:.4f}")
        print()
        
        print("НАЧАЛЬНЫЕ ПАРАМЕТРЫ:")
        print(f"  sigma_inf = {res['initial_params']['sigma_inf']:.4f} МПа (0.5 * σ_min)")
        print(f"  log10(C)  = {res['initial_params']['log10C']:.4f}")
        print(f"  C         = {res['initial_params']['C']:.4f}")
        print(f"  m         = {res['initial_params']['m']:.6f}")
        print()
        
        print("СТАТУС ОПТИМИЗАЦИИ:")
        if res['success']:
            print(f"  УСПЕШНО (ier={res['ier']}: {res['mesg']})")
        else:
            print(f"  НЕ УСПЕШНО (ier={res['ier']}: {res['mesg']})")
        print(f"  Количество вызовов функции (nfev) = {res['nfev']}")
        print()
        
        print("ФИНАЛЬНЫЕ ПАРАМЕТРЫ:")
        print(f"  sigma_inf = {res['sigma_inf']:.7f} МПа")
        print(f"  log10(C)  = {res['log10C']:.7f}")
        print(f"  C         = {res['C']:.7f}")
        print(f"  m         = {res['m']:.7f}")
        print(f"  Q         = {res['Q']:.7f}")
        print()
        
        print("СТАНДАРТНЫЕ ОШИБКИ ПАРАМЕТРОВ:")
        print(f"  se(sigma_inf) = {res['std_errors'][0]:.6e}")
        print(f"  se(C)         = {res['std_errors'][1]:.6e}")
        print(f"  se(m)         = {res['std_errors'][2]:.6e}")
        print()
        
        print("КОВАРИАЦИОННАЯ МАТРИЦА:")
        print("       [sigma_inf        C            m     ]")
        for i in range(3):
            print(f"      [{res['covariance'][i,0]:12.6e} {res['covariance'][i,1]:12.6e} {res['covariance'][i,2]:12.6e}]")
        print()
        
        print("СРАВНЕНИЕ:")
        print(f"{'i':<3} {'σ(МПа)':>10} {'lgN_exp':>12} {'lgN_calc':>12} {'σ_calc(МПа)':>14} {'Невязка':>12}")
        print("-" * 70)
        for i in range(len(params.cx)):
            print(f"{i+1:<3} {params.cx[i]:10.1f} {params.lgN[i]:12.6f} {res['y_pred'][i]:12.6f} {res['sigma_calc'][i]:14.2f} {res['residuals'][i]:12.6e}")
        print()

    # ========== СРАВНЕНИЕ МОДЕЛЕЙ ==========
    if params.ku == 2:
        print("\n" + "=" * 80)
        print("СРАВНЕНИЕ ДВУХ МОДЕЛЕЙ")
        print("=" * 80)
        print(f"{'Параметр':<15} {'ku=0':>20} {'ku=1':>20}")
        print("-" * 55)
        print(f"{'sigma_inf (МПа)':<15} {results[0]['sigma_inf']:20.2f} {results[1]['sigma_inf']:20.2f}")
        print(f"{'C':<15} {results[0]['C']:20.2f} {results[1]['C']:20.2f}")
        print(f"{'m':<15} {results[0]['m']:20.4f} {results[1]['m']:20.4f}")
        print(f"{'Q':<15} {results[0]['Q']:20.7f} {results[1]['Q']:20.7f}")
        print()

    # ========== НАПРЯЖЕНИЯ И ОШИБКИ ДЛЯ БАЗОВЫХ ДОЛГОВЕЧНОСТЕЙ ==========
    print("\n" + "=" * 80)
    print("АМПЛИТУДЫ НАПРЯЖЕНИЙ И ОТНОСИТЕЛЬНЫЕ ОШИБКИ ДЛЯ БАЗОВЫХ ДОЛГОВЕЧНОСТЕЙ:")
    print("=" * 80)
    
    if params.ku == 0:
        print(f"{'i':<3} {'lg(N0)':>10} {'σ(lgN0), МПа':>18} {'σ_std, МПа':>14} {'δ = σ_std/σ':>14} {'δ,%':>10}")
        print("-" * 80)
        res = results[0]
        for i in range(len(params.N0)):
            sigma_R = res['sigma_inf'] + res['C'] * (params.N0[i]) ** (-res['m'])
            std_R = theoretical_std_sigma_R(
                res['sigma_inf'], res['C'], res['m'], 0,
                params.cx, params.lgN, w0, params.ni, params.N0[i]
            )
            delta = std_R / sigma_R if sigma_R > 0 else 0.0
            print(f"{i+1:<3} {params.N0[i]:10.3f} {sigma_R:18.7f} {std_R:14.7f} {delta:14.6f} {delta*100:9.2f}%")
    
    elif params.ku == 1:
        print(f"{'i':<3} {'lg(N0)':>10} {'σ(10^N0), МПа':>20} {'σ_std, МПа':>14} {'δ = σ_std/σ':>14} {'δ,%':>10}")
        print("-" * 80)
        res = results[1]
        for i in range(len(params.N0)):
            sigma_R = res['sigma_inf'] + res['C'] * (10 ** params.N0[i]) ** (-res['m'])
            std_R = theoretical_std_sigma_R(
                res['sigma_inf'], res['C'], res['m'], 1,
                params.cx, params.lgN, w1,params.ni, params.N0[i]
            )
            delta = std_R / sigma_R if sigma_R > 0 else 0.0
            print(f"{i+1:<3} {params.N0[i]:10.3f} {sigma_R:20.7f} {std_R:14.7f} {delta:14.6f} {delta*100:9.2f}%")
    
    elif params.ku == 2:
        # Вывод для обеих моделей
        print("\n--- МОДЕЛЬ ku=0 (sigma = sigma_inf + C * (lgN)^(-m)) ---")
        print(f"{'i':<3} {'lg(N0)':>10} {'σ(lgN0), МПа':>18} {'σ_std, МПа':>14} {'δ = σ_std/σ':>14} {'δ,%':>10}")
        print("-" * 80)
        res0 = results[0]
        for i in range(len(params.N0)):
            sigma_R = res0['sigma_inf'] + res0['C'] * (params.N0[i]) ** (-res0['m'])
            std_R = theoretical_std_sigma_R(
                res0['sigma_inf'], res0['C'], res0['m'], 0,
                params.cx, params.lgN, w0, params.ni, params.N0[i]
            )
            delta = std_R / sigma_R if sigma_R > 0 else 0.0
            print(f"{i+1:<3} {params.N0[i]:10.3f} {sigma_R:18.7f} {std_R:14.7f} {delta:14.6f} {delta*100:9.2f}%")
        
        print("\n--- МОДЕЛЬ ku=1 (sigma = sigma_inf + C * N^(-m)) ---")
        print(f"{'i':<3} {'lg(N0)':>10} {'σ(10^N0), МПа':>20} {'σ_std, МПа':>14} {'δ = σ_std/σ':>14} {'δ,%':>10}")
        print("-" * 80)
        res1 = results[1]
        for i in range(len(params.N0)):
            sigma_R = res1['sigma_inf'] + res1['C'] * (10 ** params.N0[i]) ** (-res1['m'])
            std_R = theoretical_std_sigma_R(
                res1['sigma_inf'], res1['C'], res1['m'], 1,
                params.cx, params.lgN, w1, params.ni, params.N0[i]
            )
            delta = std_R / sigma_R if sigma_R > 0 else 0.0
            print(f"{i+1:<3} {params.N0[i]:10.3f} {sigma_R:20.7f} {std_R:14.7f} {delta:14.6f} {delta*100:9.2f}%")

if __name__ == "__main__":
    main()