# signals

Notes and visualizations for Oppenheim, *Signals & Systems*, ch. 2 (LTI systems).

- `lti.ipynb`: interactive views of §2.1.1 (signals as sums of shifted impulses) and §2.1.2 (the convolution sum, flip & slide). Needs `ipympl` and `ipywidgets`.
- `convolve.py`: convolve two signals given as formulas in `n` and get an exact piecewise closed form, checked numerically and plotted:
  ```
  python convolve.py "0.8**n*u(n)" "u(n)-u(n-5)" --steps
  ```
- `lti_manim.py`: manim animations `ImpulseDecomposition` and `ConvolutionSum`; rendered previews are in `media/`.
  ```
  conda create -n manim -c conda-forge python=3.12 manim
  conda activate manim
  manim -pql lti_manim.py ConvolutionSum
  ```
