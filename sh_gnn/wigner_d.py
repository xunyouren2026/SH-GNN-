"""Numerically stable Wigner small-d matrices."""

import torch
import math

_LOG_FACT_CACHE_D = {}


def _log_fact_d(n: int) -> float:
    if n < 0:
        return -float('inf')
    if n not in _LOG_FACT_CACHE_D:
        if n == 0:
            _LOG_FACT_CACHE_D[n] = 0.0
        else:
            _LOG_FACT_CACHE_D[n] = _LOG_FACT_CACHE_D.get(
                n-1, 0.0) + math.log(n)
    return _LOG_FACT_CACHE_D[n]


def _log_binom(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -float('inf')
    return _log_fact_d(n) - _log_fact_d(k) - _log_fact_d(n - k)


def stable_wigner_d(l: int, beta: torch.Tensor) -> torch.Tensor:
    size = 2 * l + 1
    device = beta.device
    beta_d = beta.double()
    cos_half = torch.cos(beta_d / 2)
    sin_half = torch.sin(beta_d / 2)

    d = torch.zeros(size, size, device=device, dtype=torch.float64)

    for m_idx, m in enumerate(range(-l, l + 1)):
        for mp_idx, mp in enumerate(range(-l, l + 1)):
            k_min = max(0, m - mp)
            k_max = min(l + m, l - mp)

            log_pos = -float('inf')
            log_neg = -float('inf')
            for k in range(int(k_min), int(k_max) + 1):
                log_term = (
                    _log_binom(l + m, k) +
                    _log_binom(l - m, l - mp - k) +
                    _log_binom(l + mp, l - m - k) +
                    (2 * l + mp - m - 2 * k) * torch.log(cos_half + 1e-30) +
                    (m - mp + 2 * k) * torch.log(sin_half + 1e-30)
                )
                sign = (-1) ** (k + mp - m)
                if sign > 0:
                    if log_term > log_pos:
                        log_pos = log_term
                    else:
                        max_val = max(log_pos, log_term)
                        log_pos = max_val + \
                            torch.log(torch.exp(log_pos - max_val) +
                                      torch.exp(log_term - max_val))
                else:
                    if log_term > log_neg:
                        log_neg = log_term
                    else:
                        max_val = max(log_neg, log_term)
                        log_neg = max_val + \
                            torch.log(torch.exp(log_neg - max_val) +
                                      torch.exp(log_term - max_val))

            if log_pos > -float('inf') and log_neg > -float('inf'):
                pos_val = torch.exp(log_pos)
                neg_val = torch.exp(log_neg)
                d[m_idx, mp_idx] = pos_val - neg_val
            elif log_pos > -float('inf'):
                d[m_idx, mp_idx] = torch.exp(log_pos)
            else:
                d[m_idx, mp_idx] = torch.exp(
                    log_neg) * (-1.0 if (log_neg > -float('inf')) else 0.0)

    return d.float()


def wigner_D_matrix_real(l: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    d = stable_wigner_d(l, beta)
    m_vals = torch.arange(-l, l + 1, device=beta.device, dtype=torch.float32)
    cos_alpha = torch.cos(m_vals * alpha)
    sin_alpha = -torch.sin(m_vals * alpha)
    cos_gamma = torch.cos(m_vals * gamma)
    sin_gamma = -torch.sin(m_vals * gamma)
    D_real = (torch.diag(cos_alpha) @ d @ torch.diag(cos_gamma) -
              torch.diag(sin_alpha) @ d @ torch.diag(sin_gamma))
    return D_real
