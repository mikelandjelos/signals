"""Convolve two discrete-time signals, in closed form when possible.

    y[n] = x[n] * h[n] = sum_k x[k] h[n-k]

A signal can be
  * a string in n, using u(.) (unit step) and delta(.) (unit impulse):
        "0.8**n * u(n)"   "u(n) - u(n-5)"   "2**n * u(-n)"   "n * u(n) * u(10-n)"
    Two strings give an exact piecewise closed form, worked out the way the book
    does it (overlap case by case), plus a numerical check.
  * a Python function of an integer array n                 -> numeric only
  * a list of values (starting at n=0), or (values, start)  -> numeric only

CLI
  python convolve.py "0.8**n*u(n)" "u(n)-u(n-5)"
  python convolve.py "2**n*u(-n)" "u(n)" --steps --range -10 10
Python
  from convolve import convolve
  r = convolve("0.8**n*u(n)", "u(n)-u(n-5)")
  print(r); r.plot(); r.closed_form   # sympy Piecewise
"""
import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

n, k = sp.symbols('n k', integer=True)
INF = sp.oo

C_X, C_H, C_Y, C_FORM = '#2a78d6', '#eb6834', '#0b0b0b', '#1baf7a'


class u(sp.Function):
  """unit step u[m]; left unevaluated so its support can be read off"""


class delta(sp.Function):
  """unit impulse delta[m]"""


PARSE_NAMES = {'n': n, 'u': u, 'step': u, 'delta': delta, 'd': delta}
NUMERIC = {'u': lambda m: np.where(m >= 0, 1.0, 0.0),
           'delta': lambda m: np.where(m == 0, 1.0, 0.0)}


class NoClosedForm(Exception):
  pass


# ---------------------------------------------------------------- parsing

def parse(text):
  # rational=True keeps 0.8 as 4/5, so the closed form stays exact
  return sp.sympify(text, locals=PARSE_NAMES, rational=True)


def _support_of(fn):
  """u(±n + c) or delta(±n + c)  ->  (lo, hi) in n"""
  arg = sp.expand(fn.args[0])
  p = sp.Poly(arg, n)
  s, c = p.coeff_monomial(n), p.coeff_monomial(1)
  if p.degree() != 1 or s not in (1, -1) or not c.is_integer:
    raise NoClosedForm(f'{fn.func.__name__} argument must be ±n + integer, got {arg}')
  if isinstance(fn, delta):
    return (-c * s, -c * s)
  return (-c, INF) if s == 1 else (-INF, c)


def terms(expr):
  """Split into terms  f(n) on lo <= n <= hi  (lo, hi may be ±oo)."""
  out = []
  for term in sp.Add.make_args(sp.expand(expr)):
    f, lo, hi = sp.Integer(1), -INF, INF
    for fac in sp.Mul.make_args(term):
      base, _ = fac.as_base_exp()       # u(n)**2 == u(n)
      if isinstance(base, (u, delta)):
        a, b = _support_of(base)
        lo, hi = sp.Max(lo, a), sp.Min(hi, b)
      else:
        f *= fac
    if f.has(u, delta):
      raise NoClosedForm(f'u/delta must appear as plain factors, got {term}')
    if lo <= hi:
      out.append((f, lo, hi))
  return out


# ---------------------------------------------------------- one term pair

def _sum(summand, L, U):
  S = sp.summation(summand, (k, L, U))
  if S.has(sp.Sum):                    # e.g. cos * geometric: try exponentials
    S = sp.summation(summand.rewrite(sp.exp), (k, L, U))
  if S.has(sp.Sum):
    raise NoClosedForm(f'sympy could not sum {summand} for k in [{L}, {U}]')
  if S.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
    raise NoClosedForm(f'sum of {summand} for k in [{L}, {U}] diverges')
  return sp.simplify(S)


def conv_pair(tx, th):
  """(f, a1, b1) * (g, a2, b2)  ->  [(n_lo, n_hi, L, U, y)]

  sum_k f(k) g(n-k)  with  a1 <= k <= b1  and  a2 <= n-k <= b2, i.e.
  k from L = max(a1, n-b2)  to  U = min(b1, n-a2).
  Which argument wins the max/min depends on n; that's the case analysis.
  """
  (f, a1, b1), (g, a2, b2) = tx, th
  summand = f.subs(n, k) * g.subs(n, n - k)
  start, end = a1 + a2, b1 + b2          # y is zero outside this

  cuts = []
  if a1.is_finite and b2.is_finite:
    cuts.append(a1 + b2)                 # L switches from a1 to n-b2 after this n
  if b1.is_finite and a2.is_finite:
    cuts.append(b1 + a2)                 # U switches from n-a2 to b1 after this n
  pieces = [(start, end)]
  for c in sorted(set(cuts)):
    pieces = [q for p0, p1 in pieces
              for q in (((p0, c), (c + 1, p1)) if p0 <= c < p1 else ((p0, p1),))]

  out = []
  for p0, p1 in pieces:
    if a1.is_finite and b2.is_finite:
      L = a1 if p1 <= a1 + b2 else n - b2
    else:
      L = a1 if a1.is_finite else (n - b2 if b2.is_finite else -INF)
    if b1.is_finite and a2.is_finite:
      U = n - a2 if p1 <= b1 + a2 else b1
    else:
      U = b1 if b1.is_finite else (n - a2 if a2.is_finite else INF)
    out.append((p0, p1, L, U, _sum(summand, L, U)))
  return out


# ---------------------------------------------------------- tidy formulas

def _regroup(e):
  """Collect every b**(c*n+d) in a term into one (b**c)**n, so 2**(2n)/5**n -> (4/5)**n."""
  groups = {}
  for t in sp.Add.make_args(sp.expand(sp.powdenest(e, force=True))):
    B, rest = sp.Integer(1), sp.Integer(1)
    for f in sp.Mul.make_args(t):
      b, x = f.as_base_exp()
      if b.is_number and b.is_positive and x.has(n) and sp.Poly(x, n).degree() == 1:
        p = sp.Poly(x, n)
        B *= b ** p.coeff_monomial(n)
        rest *= b ** p.coeff_monomial(1)
      else:
        rest *= f
    groups[B] = groups.get(B, 0) + rest
  return sp.Add(*(sp.factor(v) * (sp.Pow(B, n, evaluate=False) if B != 1 else 1)
                  for B, v in groups.items()))


def nicer(e):
  e = sp.simplify(e)
  cands = [e, _regroup(e)]
  if e.has(sp.I):   # real for integer n: fold exponentials back into cos/sin
    nr = sp.Symbol('nr', real=True)
    r = sp.expand_complex(sp.expand(sp.powsimp(e.subs(n, nr), force=True)))
    r = sp.simplify(r.rewrite(sp.cos)).subs(nr, n)
    cands += [r, sp.simplify(_regroup(r))]
  return min(cands, key=lambda z: (z.has(sp.I), sp.count_ops(z), len(str(z))))


# ------------------------------------------------------- the whole signal

def closed_form(x_expr, h_expr, steps=None):
  """Exact y[n] as [(lo, hi, expr)] over consecutive n intervals (zero elsewhere)."""
  pieces = []
  for tx in terms(x_expr):
    for th in terms(h_expr):
      for p0, p1, L, U, y in conv_pair(tx, th):
        pieces.append((p0, p1, y))
        if steps is not None:
          steps.append((tx, th, p0, p1, L, U, y))

  # cut n into elementary intervals at every piece boundary, add what's active
  cuts = sorted({p0 for p0, _, _ in pieces if p0.is_finite}
                | {p1 + 1 for _, p1, _ in pieces if p1.is_finite})
  bounds = [-INF] + cuts + [INF]
  out = []
  for lo, nxt in zip(bounds[:-1], bounds[1:]):
    hi = nxt - 1 if nxt.is_finite else INF
    y = sp.simplify(sum((e for p0, p1, e in pieces if p0 <= lo and hi <= p1), sp.Integer(0)))
    if lo == hi:
      y = sp.simplify(y.subs(n, lo))     # a single sample is just a number
    if out:
      plo, phi, py = out[-1]
      if sp.simplify(py - y) == 0 or (lo == hi and sp.simplify(py.subs(n, lo) - y) == 0):
        out[-1] = (plo, hi, py)          # same formula continues
        continue
      if plo == phi and sp.simplify(y.subs(n, plo) - py) == 0:
        out[-1] = (plo, hi, y)           # previous single sample fits this formula
        continue
    out.append((lo, hi, y))
  # a formula can hit exactly 0 at a finite end (cancelling steps): shrink to the real support
  trimmed = []
  for lo, hi, y in out:
    if y == 0:
      continue
    while lo.is_finite and lo <= hi and sp.simplify(y.subs(n, lo)) == 0:
      lo += 1
    while hi.is_finite and lo <= hi and sp.simplify(y.subs(n, hi)) == 0:
      hi -= 1
    if lo <= hi and y != 0:
      trimmed.append((lo, hi, nicer(y)))
  return trimmed


def interval_text(lo, hi):
  if not lo.is_finite and not hi.is_finite:
    return 'all n'
  if not lo.is_finite:
    return f'n ≤ {hi}'
  if not hi.is_finite:
    return f'n ≥ {lo}'
  return f'n = {lo}' if lo == hi else f'{lo} ≤ n ≤ {hi}'


# ----------------------------------------------------------- numeric side

def as_function(sig):
  """anything we accept -> (sympy expr or None, vectorized numpy function)"""
  if isinstance(sig, str):
    expr = parse(sig)
    lam = sp.lambdify(n, expr, modules=[NUMERIC, 'numpy'])

    def fn(m):
      m = np.asarray(m, float)
      with np.errstate(all='ignore'):
        v = np.broadcast_to(np.asarray(lam(m), float), m.shape).copy()
      v[~np.isfinite(v)] = 0.0           # inf * u = 0 where the step is off
      return v
    return expr, fn
  if callable(sig):
    return None, lambda m: np.broadcast_to(np.asarray(sig(np.asarray(m)), float), np.shape(m))
  vals, start = sig if isinstance(sig, tuple) else (sig, 0)
  vals = np.asarray(vals, float)

  def fn(m):
    m = np.asarray(m)
    out = np.zeros(m.shape)
    ok = (m >= start) & (m < start + vals.size)
    out[ok] = vals[(m[ok] - start).astype(int)]
    return out
  return None, fn


def numeric_conv(x, h, N, margin=400):
  """brute-force y[n] = sum_k x[k] h[n-k] on N, k over a wide finite window"""
  M = int(max(abs(N[0]), abs(N[-1]))) + margin
  K = np.arange(-M, M + 1)
  xk = x(K)
  prods = np.array([xk * h(m - K) for m in N])
  y = prods.sum(axis=1)
  # truncation only matters if the summand is still nonzero at the window edges
  edge = np.abs(prods[:, [0, -1]]).max()
  if edge > 1e-9 * max(np.abs(prods).max(), 1e-300):
    warnings.warn(f'x or h is still {edge:.3g} at |n| = {M}: the numeric sum is truncated '
                  'there and may be off (or the true sum diverges).')
  return y


# ----------------------------------------------------------------- result

class Convolution:
  def __init__(self, x, h, steps=False):
    self.x_src, self.h_src = x, h
    self.x_expr, self.x = as_function(x)
    self.h_expr, self.h = as_function(h)
    self.pieces, self.steps, self.why_not = None, [] if steps else None, None
    if self.x_expr is not None and self.h_expr is not None:
      try:
        self.pieces = closed_form(self.x_expr, self.h_expr, self.steps)
      except NoClosedForm as e:
        self.why_not = str(e)
    else:
      self.why_not = 'closed form needs both signals as formulas'

  @property
  def closed_form(self):
    if self.pieces is None:
      return None
    return sp.Piecewise(*[(y, sp.And(n >= lo, n <= hi)) for lo, hi, y in self.pieces],
                        (0, True))

  def default_range(self):
    if self.pieces:
      ends = [b for lo, hi, _ in self.pieces for b in (lo, hi) if b.is_finite]
      a, b = (int(min(ends)), int(max(ends))) if ends else (0, 0)
      return a - 5, max(b + 8, a + 20)
    return -5, 25

  def evaluate(self, N):
    """closed form at integer n (NaN where there is none)"""
    if self.pieces is None:
      return np.full(len(N), np.nan)
    y = np.zeros(len(N))
    for lo, hi, e in self.pieces:
      m = (N >= lo) & (N <= hi)
      if m.any():
        f = sp.lambdify(n, e, 'numpy')
        y[m] = np.broadcast_to(np.real(f(N[m].astype(float))), m.sum())
    return y

  def __str__(self):
    name = lambda src, e: sp.sstr(e) if e is not None else type(src).__name__
    lines = [f'x[n] = {name(self.x_src, self.x_expr)}',
             f'h[n] = {name(self.h_src, self.h_expr)}', '']
    if self.steps:
      lines.append('term by term (sum over the overlap of the two supports):')
      for (f, a1, b1), (g, a2, b2), p0, p1, L, U, y in self.steps:
        lines.append(f'  x-term {sp.sstr(f)} on {interval_text(a1, b1)}'
                     f'  *  h-term {sp.sstr(g)} on {interval_text(a2, b2)}')
        lines.append(f'      {interval_text(p0, p1):>16}:  sum_{{k={L}}}^{{{U}}} '
                     f'{sp.sstr(f.subs(n, k) * g.subs(n, n - k))}  =  {sp.sstr(nicer(y))}')
      lines.append('')
    if self.pieces is None:
      lines.append(f'no closed form ({self.why_not}); numeric only')
    elif not self.pieces:
      lines.append('y[n] = 0 for all n')
    else:
      lines.append('y[n] =')
      for lo, hi, y in self.pieces:
        lines.append(f'  {sp.sstr(y):<40}  {interval_text(lo, hi)}')
      lines.append(f'  {"0":<40}  otherwise')
    return '\n'.join(lines)

  def plot(self, n_range=None, show=True):
    a, b = n_range or self.default_range()
    N = np.arange(a, b + 1)
    y_num = numeric_conv(self.x, self.h, N)
    y_cf = self.evaluate(N)
    if self.pieces is not None:
      err = np.nanmax(np.abs(y_cf - y_num)) if len(N) else 0
      print(f'closed form vs numeric sum on [{a}, {b}]: max |difference| = {err:.2e}')

    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    for ax, vals, color, title in (
        (axes[0], self.x(N), C_X, f'x[n] = {self.x_src if isinstance(self.x_src, str) else "(given)"}'),
        (axes[1], self.h(N), C_H, f'h[n] = {self.h_src if isinstance(self.h_src, str) else "(given)"}'),
        (axes[2], y_num, C_Y, 'y[n] = x[n] * h[n]')):
      ax.axhline(0, color='#c3c2b7', lw=0.8)
      ax.vlines(N, 0, vals, color=color, lw=1.6)
      ax.plot(N, vals, 'o', color=color, ms=4.5)
      ax.set_title(title, loc='left', fontsize=10)
      ax.spines[['top', 'right']].set_visible(False)

    ax = axes[2]
    if self.pieces:
      ax.plot(N, y_cf, 'x', color=C_FORM, ms=7, mew=1.5, label='closed form')
      ax.legend(loc='upper right', frameon=False, fontsize=8)
      # shade each case of the piecewise answer and label it with its formula
      for i, (lo, hi, e) in enumerate(self.pieces):
        l, r = max(lo, a) - 0.5, min(hi, b) + 0.5
        if l < r:
          ax.axvspan(l, r, color=C_FORM, alpha=0.06 + 0.06 * (i % 2), lw=0)
          ax.text((l + r) / 2, 1.02, f'${sp.latex(e)}$', transform=ax.get_xaxis_transform(),
                  ha='center', va='bottom', fontsize=8, color='#52514e', clip_on=False)
      ax.set_title('y[n] = x[n] * h[n]', loc='left', fontsize=10, pad=22)
    ax.set_xlabel('n')
    fig.tight_layout()
    if show:
      plt.show()
    return fig


def convolve(x, h, steps=False):
  return Convolution(x, h, steps=steps)


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('x', help='x[n], e.g. "0.8**n*u(n)"')
  p.add_argument('h', help='h[n], e.g. "u(n)-u(n-5)"')
  p.add_argument('--range', nargs=2, type=int, metavar=('N0', 'N1'), help='n range to plot')
  p.add_argument('--steps', action='store_true', help='show the term-by-term case analysis')
  p.add_argument('--no-plot', action='store_true')
  p.add_argument('--save', metavar='FILE', help='save the plot instead of showing it')
  args = p.parse_args()

  r = convolve(args.x, args.h, steps=args.steps)
  print(r)
  if not args.no_plot:
    fig = r.plot(args.range, show=not args.save)
    if args.save:
      fig.savefig(args.save, dpi=130)
      print(f'saved {args.save}')


if __name__ == '__main__':
  main()
