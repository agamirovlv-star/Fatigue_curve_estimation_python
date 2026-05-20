import numpy as np
import json
import sys
from datetime import datetime
from typing import List
from dataclasses import dataclass
from fatigue_calculator import estimate_fatigue_curve, estimate_sigma_for_N0

#================================================================

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

#================================================================

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

#================================================================

def compute_weights(ni: List[int], lgN: List[float], slgN: List[float], ku: int) -> List[float]:
    w = []
    for i in range(len(ni)):
        if ku == 0:
            w.append(ni[i] * (lgN[i] / slgN[i])**2)
        else:  # ku == 1
            w.append(ni[i] / (slgN[i])**2)
    return w


def main():
    tee = Tee('fatigue.out')
    sys.stdout = tee

    print("=" * 70)
    print("ОЦЕНКА ПАРАМЕТРОВ КРИВОЙ УСТАЛОСТИ (method='lm')")
    print("=" * 70)

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
        print("\n" + "=" * 70)
        print("РЕЗУЛЬТАТ ДЛЯ ku=0 (sigma = sigma_inf + C * (lgN)^(-m))")
        print("=" * 70)
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
        print(f"  sigma_inf = {res['sigma_inf']:.2f} МПа")
        print(f"  log10(C)  = {res['log10C']:.4f}")
        print(f"  C         = {res['C']:.4f}")
        print(f"  m         = {res['m']:.4f}")
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
        print("\n" + "=" * 70)
        print("РЕЗУЛЬТАТ ДЛЯ ku=1 (sigma = sigma_inf + C * N^(-m))")
        print("=" * 70)
        # Вычисляем веса только для ku=1
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
        print(f"  sigma_inf = {res['sigma_inf']:.2f} МПа")
        print(f"  log10(C)  = {res['log10C']:.4f}")
        print(f"  C         = {res['C']:.4f}")
        print(f"  m         = {res['m']:.4f}")
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
        print("\n" + "=" * 70)
        print("СРАВНЕНИЕ ДВУХ МОДЕЛЕЙ")
        print("=" * 70)
        print(f"{'Параметр':<15} {'ku=0':>20} {'ku=1':>20}")
        print("-" * 55)
        print(f"{'sigma_inf (МПа)':<15} {results[0]['sigma_inf']:20.2f} {results[1]['sigma_inf']:20.2f}")
        print(f"{'C':<15} {results[0]['C']:20.2f} {results[1]['C']:20.2f}")
        print(f"{'m':<15} {results[0]['m']:20.4f} {results[1]['m']:20.4f}")
        print(f"{'Q':<15} {results[0]['Q']:20.7f} {results[1]['Q']:20.7f}")
        print()

    # ========== НАПРЯЖЕНИЯ ДЛЯ БАЗОВЫХ ДОЛГОВЕЧНОСТЕЙ ==========
    print("\n" + "=" * 70)
    print("АМПЛИТУДЫ НАПРЯЖЕНИЙ ДЛЯ БАЗОВЫХ ДОЛГОВЕЧНОСТЕЙ:")
    print(f"{'i':<3} {'lg(N0)':>10} {'σ(lgN0) (МПа)':>18}")
    print("-" * 35)

    if params.ku == 0 or params.ku == 2:
        sigma_list = estimate_sigma_for_N0(params.N0, results[0]['sigma_inf'], results[0]['C'], results[0]['m'], 0)
        for i, (N0, sigma) in enumerate(zip(params.N0, sigma_list)):
            print(f"{i+1:<3} {N0:10.3f} {sigma:18.7f}")

    if params.ku == 1 or params.ku == 2:
        sigma_list = estimate_sigma_for_N0(params.N0, results[1]['sigma_inf'], results[1]['C'], results[1]['m'], 1)
        for i, (N0, sigma) in enumerate(zip(params.N0, sigma_list)):
            print(f"{i+1:<3} {N0:10.3f} {sigma:18.7f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()