# FaST-LMM light

Lightweight implementation of the [FaST-LMM algorithm](https://europepmc.org/article/med/21892150) for estimating the parameters $\beta$, $\sigma^2$ and $\delta$ in a linear [mixed model](https://en.wikipedia.org/wiki/Mixed_model) or [Gaussian process](https://en.wikipedia.org/wiki/Gaussian_process) of the form

$$y \\sim N\\bigl(X\\beta, \\sigma^2(K + \\delta I) \\bigr)$$

[julia implementation](https://github.com/sens/FaSTLMM.jl)

[original](https://fastlmm.github.io/)

[![Build Status](https://github.com/tmichoel/FaSTLMMlight.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/tmichoel/FaSTLMMlight.jl/actions/workflows/CI.yml?query=branch%3Amaster)
