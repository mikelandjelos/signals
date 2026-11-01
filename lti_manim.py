"""Manim animations for Oppenheim 2.1.1 and 2.1.2.

  ImpulseDecomposition  x[n] = sum_k x[k] delta[n-k], built one impulse at a time
  ConvolutionSum        LTI argument, then y[n] = sum_k x[k] h[n-k] by flip & slide

Render:  manim -pql lti_manim.py ImpulseDecomposition    (preview, low quality)
         manim -qh  lti_manim.py ConvolutionSum           (1080p)
"""
from manim import *
import numpy as np

# same roles as the notebook: blue = input, orange = kernel, green = term/product
C_X = '#3987e5'
C_H = '#eb6834'
C_TERM = '#1baf7a'
C_OUT = WHITE
C_GHOST = GREY_D


def stem(ax, k, v, color, dx=0.0, width=4, r=0.07):
  """Stem at index k with value v on axes ax; dx nudges it sideways."""
  base, top = ax.c2p(k + dx, 0), ax.c2p(k + dx, v)
  if abs(v) < 1e-9:
    return VGroup(Dot(base, radius=0.035, color=C_GHOST))
  return VGroup(Line(base, top, color=color, stroke_width=width),
                Dot(top, radius=r, color=color))


def stems(ax, seq, color, **kw):
  """seq: dict {k: value}"""
  return VGroup(*(stem(ax, k, v, color, **kw) for k, v in seq.items()))


def shift_tex(k, var='n'):
  # n-k for a concrete k:  -2 -> n+2,  0 -> n,  1 -> n-1
  return var if k == 0 else f'{var}{-k:+d}'


def axes(x_range, y_range, width, height, y_numbers=()):
  return Axes(
      x_range=x_range, y_range=y_range, x_length=width, y_length=height,
      tips=False,
      axis_config={'color': GREY_B, 'stroke_width': 2, 'font_size': 22},
      x_axis_config={'numbers_to_include': range(x_range[0], x_range[1] + 1)},
      y_axis_config={'numbers_to_include': y_numbers},
  )


class ImpulseDecomposition(Scene):
  X = {-2: 1.0, -1: -1.5, 0: 2.0, 1: 1.0, 2: -0.5, 3: 1.5}   # Fig. 2.1 style

  def construct(self):
    title = MathTex(r'x[n] = \sum_{k=-\infty}^{\infty} x[k]\,\delta[n-k]', font_size=40)
    title.to_edge(UP, buff=0.3)
    self.play(Write(title))

    top = axes([-5, 5, 1], [-2, 2.5, 1], 9.5, 2.3, y_numbers=[-1, 1, 2]).shift(UP * 1.0 + RIGHT * 0.9)
    bot = axes([-5, 5, 1], [-2, 2.5, 1], 9.5, 2.3, y_numbers=[-1, 1, 2]).shift(DOWN * 1.65 + RIGHT * 0.9)
    top_lbl = MathTex('x[n]', font_size=34).next_to(top, LEFT, buff=0.3)
    bot_lbl = Tex('building\\\\block', font_size=28, color=GREY_B).next_to(bot, LEFT, buff=0.3)
    self.play(Create(top), Create(bot), FadeIn(top_lbl), FadeIn(bot_lbl))

    # the target, as a ghost, so we can see the sum filling it in
    ghost = stems(top, self.X, C_GHOST, width=3, r=0.05)
    self.play(FadeIn(ghost))

    # running equation along the bottom, one term per sample
    parts = ['x[n]', '=']
    for i, k in enumerate(self.X):
      parts += (['+'] if i else []) + [f'x[{k}]\\,\\delta[{shift_tex(k)}]']
    eq = MathTex(*parts, font_size=30).to_edge(DOWN, buff=0.15)
    eq.scale_to_fit_width(min(eq.width, config.frame_width - 0.6)).to_edge(DOWN, buff=0.15)
    self.play(Write(eq[:2]))
    term_idx = 2

    for i, (k, v) in enumerate(self.X.items()):
      slow = i < 2
      rt = 1.0 if slow else 0.45

      # 1. a unit impulse at the origin
      imp = stem(bot, 0, 1, C_H)
      lbl = MathTex(r'\delta[n]', font_size=32, color=C_H).next_to(bot.c2p(0, 1), UP, buff=0.12)
      self.play(GrowFromPoint(imp, bot.c2p(0, 0)), FadeIn(lbl), run_time=rt)

      # 2. shift it to n = k
      lbl2 = MathTex(f'\\delta[{shift_tex(k)}]', font_size=32, color=C_H).next_to(bot.c2p(k, 1), UP, buff=0.12)
      self.play(imp.animate.shift(bot.c2p(k, 0) - bot.c2p(0, 0)), Transform(lbl, lbl2), run_time=rt)

      # 3. scale it by x[k]
      scaled = stem(bot, k, v, C_TERM)
      side = UP if v > 0 else DOWN
      lbl3 = MathTex(f'x[{k}]\\,\\delta[{shift_tex(k)}]', font_size=32, color=C_TERM)
      lbl3.next_to(bot.c2p(k, v), side, buff=0.12)
      if slow:
        val = MathTex(f'x[{k}] = {v:g}', font_size=28, color=GREY_A).next_to(lbl3, RIGHT, buff=0.3)
        self.play(Transform(imp, scaled), Transform(lbl, lbl3), FadeIn(val), run_time=rt)
      else:
        val = VMobject()
        self.play(Transform(imp, scaled), Transform(lbl, lbl3), run_time=rt)

      # 4. drop it into the sum
      placed = stem(top, k, v, C_OUT)
      term = eq[term_idx:term_idx + 2] if i else eq[term_idx:term_idx + 1]
      self.play(ReplacementTransform(imp, placed), ReplacementTransform(lbl, term),
                FadeOut(val), run_time=rt * 1.2)
      term_idx += len(term)
      if slow:
        self.wait(0.3)

    self.wait(0.5)
    summed = MathTex(r'x[n]', '=', r'\sum_k x[k]\,\delta[n-k]', font_size=34).to_edge(DOWN, buff=0.1)
    self.play(TransformMatchingTex(eq, summed), FadeOut(ghost))
    self.play(Circumscribe(summed, color=C_TERM))
    self.wait(1.5)


class ConvolutionSum(Scene):
  X = {k: 1.0 for k in range(0, 5)}                  # rectangle, 0..4
  H = {0: 1.0, 1: 0.75, 2: 0.5, 3: 0.25}             # decaying, asymmetric so the flip is visible

  def construct(self):
    self.lti_argument()
    self.flip_and_slide()

  def lti_argument(self):
    rows = [
        (r'\delta[n]', r'h[n]', 'definition of $h$'),
        (r'\delta[n-k]', r'h[n-k]', 'time invariance'),
        (r'x[k]\,\delta[n-k]', r'x[k]\,h[n-k]', 'homogeneity'),
        (r'\sum_k x[k]\,\delta[n-k]', r'\sum_k x[k]\,h[n-k]', 'additivity'),
    ]
    head = Tex('Why the impulse response is enough', font_size=40).to_edge(UP, buff=0.4)
    self.play(Write(head))
    lines = VGroup()
    for lhs, rhs, why in rows:
      l = MathTex(lhs, font_size=38, color=C_X)
      a = Arrow(LEFT, RIGHT, color=GREY_B).scale(0.8)
      sys = Tex('LTI', font_size=26).next_to(a, UP, buff=0.05)
      r = MathTex(rhs, font_size=38, color=C_H)
      w = Tex(why, font_size=26, color=GREY_B)
      lines.add(VGroup(l, VGroup(a, sys), r, w))
    for line in lines:
      line[0].move_to(LEFT * 3.6)
      line[1].move_to(LEFT * 0.9)
      line[2].move_to(RIGHT * 1.9)
      line[3].move_to(RIGHT * 5.0)
    for i, line in enumerate(lines):   # fixed columns, so arrows and labels line up
      line.shift(UP * (2.0 - 1.05 * i - line[1][0].get_center()[1]))
    for line in lines:
      self.play(FadeIn(line[0], shift=RIGHT * 0.2), GrowArrow(line[1][0]), FadeIn(line[1][1]), run_time=0.7)
      self.play(TransformFromCopy(line[0], line[2]), FadeIn(line[3]), run_time=0.9)
      self.wait(0.4)

    concl = MathTex('x[n]', r'\;\Rightarrow\;', 'y[n] = ', r'\sum_k x[k]\,h[n-k]', font_size=44)
    concl.next_to(lines, DOWN, buff=0.55)
    concl[0].set_color(C_X)
    self.play(Write(concl))
    self.play(Circumscribe(concl[2:], color=C_TERM))
    self.wait(1)
    self.play(FadeOut(head, lines, concl[:2]), concl[2:].animate.scale(0.85).to_edge(UP, buff=0.25))
    self.formula = concl[2:]

  def flip_and_slide(self):
    X, H = self.X, self.H
    hk = lambda m: H.get(m, 0.0)
    xk = lambda k: X.get(k, 0.0)
    kmin, kmax = -6, 11
    ys = {n: sum(xk(k) * hk(n - k) for k in X) for n in range(kmin, kmax + 1)}

    ax1 = axes([kmin, kmax, 1], [0, 1.4, 1], 11.5, 1.5, y_numbers=[1]).shift(UP * 1.75)
    ax2 = axes([kmin, kmax, 1], [0, 1.4, 1], 11.5, 1.5, y_numbers=[1]).shift(DOWN * 0.45)
    ax3 = axes([kmin, kmax, 1], [0, 3.0, 1], 11.5, 1.5, y_numbers=[1, 2, 3]).shift(DOWN * 2.75)
    for ax in (ax1, ax2, ax3):
      ax.shift(RIGHT * 0.9)
    lbl1 = MathTex('x[k]', font_size=30, color=C_X).next_to(ax1, LEFT, buff=0.2).shift(UP * 0.25)
    lbl2 = MathTex('x[k]\\,h[n-k]', font_size=28, color=C_TERM).next_to(ax2, LEFT, buff=0.15)
    lbl3 = MathTex('y[n]', font_size=30).next_to(ax3, LEFT, buff=0.2)
    klab = [MathTex('k', font_size=26, color=GREY_B).next_to(ax.x_axis, RIGHT, buff=0.1) for ax in (ax1, ax2)]
    nlab = MathTex('n', font_size=26, color=GREY_B).next_to(ax3.x_axis, RIGHT, buff=0.1)
    self.play(Create(ax1), FadeIn(lbl1), FadeIn(klab[0]))
    xs = stems(ax1, X, C_X, dx=-0.12)
    self.play(LaggedStartMap(GrowFromEdge, xs, edge=DOWN, lag_ratio=0.15))

    # h[k]  ->  flip  ->  h[-k]
    hs = stems(ax1, H, C_H, dx=0.12)
    hlbl = MathTex('h[k]', font_size=30, color=C_H).next_to(lbl1, DOWN, buff=0.15)
    self.play(LaggedStartMap(GrowFromEdge, hs, edge=DOWN, lag_ratio=0.15), FadeIn(hlbl))
    self.wait(0.5)
    flipped = stems(ax1, {-m: v for m, v in H.items()}, C_H, dx=0.12)
    hlbl2 = MathTex('h[-k]', font_size=30, color=C_H).move_to(hlbl)
    note = Tex('1. flip', font_size=30, color=C_H).next_to(ax1, UP, buff=0.1).align_to(ax1, RIGHT)
    self.play(Transform(hs, flipped), Transform(hlbl, hlbl2), FadeIn(note), run_time=1.5)
    self.wait(0.5)

    # from here on h[n-k] is driven by n
    n = ValueTracker(0)
    self.remove(hs)
    h_slide = always_redraw(lambda: stems(
        ax1, {n.get_value() - m: v for m, v in H.items()}, C_H, dx=0.12))
    hlbl_live = always_redraw(lambda: MathTex(
        f'h[{int(round(n.get_value()))}-k]' if round(n.get_value()) else 'h[-k]',
        font_size=30, color=C_H).move_to(hlbl))
    self.remove(hlbl)
    self.add(h_slide, hlbl_live)

    def overlap(ax):
      def make():
        nn = int(round(n.get_value()))
        lo, hi = max(min(X), nn - max(H)), min(max(X), nn - min(H))
        if lo > hi:
          return VMobject()
        x0, x1 = ax.c2p(lo - 0.45, 0)[0], ax.c2p(hi + 0.45, 0)[0]
        r = Rectangle(width=x1 - x0, height=ax.y_length + 0.1,
                      fill_color=C_TERM, fill_opacity=0.12, stroke_width=0)
        return r.move_to([(x0 + x1) / 2, ax.get_center()[1], 0])
      return always_redraw(make)

    note2 = Tex('2. slide to $n$, multiply, sum', font_size=30, color=C_TERM)
    note2.next_to(ax1, UP, buff=0.1).align_to(ax1, RIGHT)
    self.play(Create(ax2), Create(ax3), FadeIn(lbl2, lbl3, klab[1], nlab), Transform(note, note2))
    self.add(overlap(ax1), overlap(ax2))
    self.bring_to_front(xs, h_slide)
    prod = always_redraw(lambda: stems(
        ax2, {k: xk(k) * hk(int(round(n.get_value())) - k) for k in range(kmin, kmax + 1)}, C_TERM))
    self.add(prod)

    readout = always_redraw(lambda: MathTex(
        f'y[{int(round(n.get_value()))}] = {ys[int(round(n.get_value()))]:.3g}',
        font_size=32, color=C_TERM).move_to(ax3.c2p(9, 2.6)))
    self.add(readout)

    self.play(n.animate.set_value(-2), run_time=1.2)
    out = VGroup()
    for i, nn in enumerate(range(-2, 10)):
      slow = i < 5
      if i:
        self.play(n.animate.set_value(nn), run_time=0.9 if slow else 0.45, rate_func=smooth)
      ystem = stem(ax3, nn, ys[nn], C_OUT)
      self.play(ReplacementTransform(prod.copy().clear_updaters(), ystem),
                run_time=0.8 if slow else 0.35)
      out.add(ystem)
      if slow:
        self.wait(0.3)

    done = Tex('length $= L_x + L_h - 1 = 5 + 4 - 1 = 8$', font_size=28)
    done.move_to(ax3.c2p(0, 2.6)).align_to(ax3.c2p(-1.0, 0), RIGHT)
    self.play(FadeIn(done), Indicate(out, color=C_TERM))
    self.wait(2)
