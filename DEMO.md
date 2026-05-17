# teach-once — Interactive Demo

**Live**: https://jaintle.github.io/teach-once/

## What it does

Show a robot a task once. TP-GPT generalizes it everywhere.

The interactive demo lets you reconfigure the scene in
your browser — no installation required. TP-GPT runs
entirely client-side in JavaScript.

## Three modes

| Mode | How to use | What you see |
|------|-----------|-------------|
| **Reshelving** | Move the box and shelf sliders | Arm picks up box and places it in the new location |
| **Cleaning** | Select a surface type, adjust tilt | Arm sweeps a cleaning path adapted to the new surface |
| **Arm-pose** | Move keypoint sliders or pick a preset | Arm traces a path through the new arm configuration |

## Quick demo

![TP-GPT highlight reel](reports/figures/final_highlight.gif)

## Run locally

No server needed — open `docs/index.html` directly in
Chrome or Firefox.

## Paper

Franzese, G., Prakash, R., Kober, J. (2024).
*Generalization of Task Parameterized Dynamical Systems
using Gaussian Process Transportation.*
arXiv:2404.13458

- [arXiv paper](https://arxiv.org/abs/2404.13458)
- [Paper video](https://youtu.be/bE6uOnAQBLo)
- [Original code](https://github.com/franzesegiovanni/policy_transportation)
