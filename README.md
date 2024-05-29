# An Analysis of Linear Time Series Forecasting Models (ICML 2024)
Authors: [William Toner](https://github.com/WToner) and [Luke Darlow](https://github.com/lukedarlow)

This is the official repository to replicate the Ordinary Least Squares models from ["An Analysis of Linear Time Series Forecasting Models"](https://arxiv.org/abs/2403.14587).

We chose to keep this repository minimal for ease of use, thereby refraining from including any baseline comparitor methods. For the paper we ensured that all comparison methods were not affected by the 'drop_last' problem. See [the FITS repo](https://github.com/VEWOXIC/FITS), where we worked with those authors to highlight and remedy this issue. 

This repository requires only Pandas, Numpy and SKLearn (no deep learning frameworks), yet produces models that are comparable or better (see Table 3 in the paper) than state-of-the-art.

Commands to replicate the main performance results in Table 2 can be found in [run_main.sh](run_main.sh), while commands for regularised models in Table 3 can be found in [run_regularised.sh](run_regularised.sh).
