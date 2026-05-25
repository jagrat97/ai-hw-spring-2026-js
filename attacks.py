"""Untargeted L-infinity adversarial attacks on inputs in [0, 1]."""
import torch
import torch.nn.functional as F


def fgsm(model, x, y, epsilon):
    """Fast Gradient Sign Method (Goodfellow et al., 2014). Single step."""
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = x_adv.detach() + epsilon * grad.sign()
    return x_adv.clamp(0.0, 1.0)


def ifgsm(model, x, y, epsilon, n_iter=10, alpha=None):
    """Iterative FGSM / PGD with L-inf projection. No random start (deterministic I-FGSM)."""
    if alpha is None:
        alpha = epsilon / n_iter * 1.25  # slight overshoot, standard PGD setting
    x_orig = x.clone().detach()
    x_adv = x.clone().detach()
    for _ in range(n_iter):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        # project into epsilon ball around x_orig
        x_adv = torch.max(torch.min(x_adv, x_orig + epsilon), x_orig - epsilon)
        x_adv = x_adv.clamp(0.0, 1.0)
    return x_adv


def mi_fgsm(model, x, y, epsilon, n_iter=10, mu=1.0, alpha=None):
    """Momentum Iterative FGSM (Dong et al., 2018)."""
    if alpha is None:
        alpha = epsilon / n_iter
    x_orig = x.clone().detach()
    x_adv = x.clone().detach()
    g = torch.zeros_like(x)
    for _ in range(n_iter):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        # normalize by L1 norm per-sample
        grad_norm = grad.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        g = mu * g + grad / grad_norm
        x_adv = x_adv.detach() + alpha * g.sign()
        x_adv = torch.max(torch.min(x_adv, x_orig + epsilon), x_orig - epsilon)
        x_adv = x_adv.clamp(0.0, 1.0)
    return x_adv


ATTACKS = {
    "FGSM": fgsm,
    "I-FGSM": ifgsm,
    "MI-FGSM": mi_fgsm,
}
