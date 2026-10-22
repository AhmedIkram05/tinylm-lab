"""Model correctness tests: shapes, causality, tied weights, PE values, param budget."""

import dataclasses
import math

import pytest
import torch

from tinylm.model import TinyLM, sinusoidal_pe
from tinylm.utils import Config, load_config

VOCAB = 8
CEILING = 2_000_000

# capacity variants are built with dataclasses.replace over the real loader, never as
# config files - the capacity *series* (configs/copy_d064.yaml etc.) is experiment-phase
# work. Each entry is (base config, overrides, quoted parameter count).
CAPACITY_VARIANTS = {
    "copy_d064": (
        "configs/copy.yaml",
        {"d_model": 64, "n_heads": 4, "d_ff": 256, "n_layers": 2},
        100_608,
    ),
    "copy_n512": (
        "configs/copy.yaml",
        {"d_model": 128, "n_heads": 4, "d_ff": 512, "n_layers": 2},
        397_824,
    ),
    "copy_d256": (
        "configs/copy.yaml",
        {"d_model": 256, "n_heads": 4, "d_ff": 1024, "n_layers": 2},
        1_582_080,
    ),
    "copy_l1": ("configs/copy.yaml", {"n_layers": 1}, 199_552),
    "copy_l4": ("configs/copy.yaml", {"n_layers": 4}, 794_368),
    # modadd carries the wider vocab (modulus + PLUS + EQ), so every count is higher
    "modadd_d064": (
        "configs/modadd.yaml",
        {"d_model": 64, "n_heads": 4, "d_ff": 256},
        100_928,
    ),
    "modadd_d256": (
        "configs/modadd.yaml",
        {"d_model": 256, "n_heads": 4, "d_ff": 1024},
        1_583_360,
    ),
    "modadd_l1": ("configs/modadd.yaml", {"n_layers": 1}, 200_192),
    "modadd_l4": ("configs/modadd.yaml", {"n_layers": 4}, 795_008),
}


def demo_cfg() -> Config:
    return load_config("configs/copy.yaml")


def test_output_shape() -> None:
    cfg = demo_cfg()
    m = TinyLM(cfg)
    m.eval()
    x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    assert m(x).shape == (2, cfg.seq_len, cfg.vocab_size)


def test_causal_mask_exact_zeros() -> None:
    cfg = demo_cfg()
    m = TinyLM(cfg)
    m.eval()
    x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    _, attentions = m(x, return_attention=True)
    assert len(attentions) == cfg.n_layers  # every layer, not just the last
    assert len({id(a) for a in attentions}) == cfg.n_layers  # distinct tensors
    causal = torch.ones(cfg.seq_len, cfg.seq_len, dtype=torch.bool).tril()

    for att in attentions:
        assert att.shape == (2, cfg.n_heads, cfg.seq_len, cfg.seq_len)
        assert bool((att[:, :, ~causal] == 0.0).all())
        assert bool(
            torch.allclose(att.sum(-1), torch.ones_like(att.sum(-1)), atol=1e-5)
        )


def test_logit_invariance_to_future() -> None:
    cfg = demo_cfg()
    m = TinyLM(cfg)
    m.eval()
    torch.manual_seed(1)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    y = x.clone()
    y[:, -3:] = (y[:, -3:] + 1) % cfg.vocab_size  # only the future changes
    assert torch.equal(m(x)[:, :-3], m(y)[:, :-3])
    assert not torch.equal(m(x), m(y))  # sanity: the change did something


def test_tied_weights() -> None:
    m = TinyLM(demo_cfg())
    assert m.head.weight is m.token_emb.weight
    assert m.head.bias is None


def test_param_count() -> None:
    n = TinyLM(demo_cfg()).num_parameters()
    assert 350_000 <= n <= 450_000  # spec: ~0.4M at defaults
    assert n < CEILING  # hard ceiling


@pytest.mark.parametrize(
    ("config", "overrides", "quoted"),
    CAPACITY_VARIANTS.values(),
    ids=list(CAPACITY_VARIANTS),
)
def test_param_count_capacity_variants(
    config: str, overrides: dict, quoted: int
) -> None:
    """Every capacity variant stays under the ceiling and matches its quoted size."""
    n = TinyLM(dataclasses.replace(load_config(config), **overrides)).num_parameters()
    assert n < CEILING
    assert n == quoted, f"{n} vs quoted {quoted}"  # deterministic integer count


def test_pe_exact_values() -> None:
    pe = sinusoidal_pe(32, 128)

    ref = torch.zeros(32, 128)
    for pos in range(32):
        for i in range(64):
            ref[pos, 2 * i] = math.sin(pos * 10000 ** (-2 * i / 128))
            ref[pos, 2 * i + 1] = math.cos(pos * 10000 ** (-2 * i / 128))

    assert torch.allclose(pe, ref, atol=1e-5)  # float32 vs float64 needs 1e-5, not 1e-6
    assert abs(pe[0, 0].item()) < 1e-7 and abs(pe[0, 1].item() - 1.0) < 1e-7


def test_demo_config_builds_and_learns() -> None:
    cfg = demo_cfg()
    m = TinyLM(cfg)
    m.train()
    x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    loss = m(x).float().mean()
    loss.backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
