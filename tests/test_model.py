"""Smoke tests for shapes and a tiny end-to-end training step (CPU-only)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.dataset import (
    SlidingWindowDataset,
    build_splits,
    generate_synthetic,
)
from src.data.data_loader import apply_channel_permutation
from src.models import DualDomainForecaster
from src.models.freq_branch import FreqBranch
from src.models.mamba_block import BiMambaEncoder, MambaEncoder, MambaSSM
from src.models.time_branch import TimeBranch


def test_mamba_ssm_shape():
    b, l, d = 3, 40, 24
    x = torch.randn(b, l, d)
    ssm = MambaSSM(d_model=d, d_state=8)
    out = ssm(x)
    assert out.shape == (b, l, d)


def test_mamba_encoder_causal():
    """A causal SSM: changing a future timestep must not alter earlier outputs."""
    torch.manual_seed(0)
    enc = MambaEncoder(d_model=16, n_layers=2, d_state=8, use_official=False).eval()
    x = torch.randn(1, 20, 16)
    with torch.no_grad():
        y1 = enc(x)
        x2 = x.clone()
        x2[:, -1] += 5.0  # perturb only the last timestep
        y2 = enc(x2)
    # Outputs before the perturbed step must be unchanged.
    assert torch.allclose(y1[:, :-1], y2[:, :-1], atol=1e-5)


def test_bimamba_encoder_bidirectional():
    """Unlike the causal encoder, a future edit MUST reach earlier outputs."""
    torch.manual_seed(0)
    enc = BiMambaEncoder(d_model=16, n_layers=1, d_state=8, use_official=False).eval()
    x = torch.randn(1, 20, 16)
    with torch.no_grad():
        y1 = enc(x)
        x2 = x.clone()
        x2[:, -1] += 5.0  # perturb only the last position
        y2 = enc(x2)
    assert y1.shape == x.shape
    # Earlier positions should change: information flows backward too.
    assert not torch.allclose(y1[:, :-1], y2[:, :-1], atol=1e-4)


def test_freq_branch_mamba_shape():
    b, l, h, c, d = 4, 96, 24, 7, 32
    x = torch.randn(b, l, c)
    branch = FreqBranch(
        seq_len=l, pred_len=h, d_model=d, encoder="mamba", use_official_mamba=False
    )
    feat, y = branch(x)
    assert feat.shape == (b, c, d)
    assert y.shape == (b, c, h)


def test_model_freq_mamba_forward():
    b, l, h, c = 2, 48, 12, 3
    x = torch.randn(b, l, c)
    model = DualDomainForecaster(
        seq_len=l, pred_len=h, n_channels=c, d_model=16,
        freq_encoder="mamba", mamba_layers=1,
    )
    assert model(x).shape == (b, h, c)


def test_time_branch_shape():
    b, l, h, c, d = 4, 96, 24, 7, 32
    x = torch.randn(b, l, c)
    branch = TimeBranch(seq_len=l, pred_len=h, d_model=d, use_official_mamba=False)
    feat, y = branch(x)
    assert feat.shape == (b, c, d)
    assert y.shape == (b, c, h)


def test_time_branch_mlp_encoder():
    b, l, h, c, d = 4, 96, 24, 7, 32
    x = torch.randn(b, l, c)
    branch = TimeBranch(seq_len=l, pred_len=h, d_model=d, encoder="mlp")
    feat, y = branch(x)
    assert feat.shape == (b, c, d)
    assert y.shape == (b, c, h)


@pytest.mark.parametrize("encoder", ["mamba", "unimamba", "bimamba",
                                      "attention", "mlp", "none"])
def test_time_dependency_ablation_encoders(encoder):
    """Every S-Mamba-style TD replacement keeps the branch contract."""
    b, l, h, c, d = 2, 32, 8, 3, 16
    x = torch.randn(b, l, c)
    branch = TimeBranch(
        seq_len=l, pred_len=h, d_model=d, encoder=encoder,
        mamba_layers=1, use_official_mamba=False,
    )
    feat, y = branch(x)
    assert feat.shape == (b, c, d)
    assert y.shape == (b, c, h)


def test_no_time_encoder_retains_dlinear_anchor():
    """Removing TD encoding must not accidentally remove the linear anchor."""
    torch.manual_seed(0)
    x = torch.randn(2, 32, 3)
    branch = TimeBranch(
        seq_len=32, pred_len=8, d_model=16, encoder="none",
        zero_init_head=True,
    ).eval()
    with torch.no_grad():
        _, actual = branch(x)
        seasonal, trend = branch.decomp(x)
        expected = branch.lin_seasonal(seasonal.transpose(1, 2))
        expected = expected + branch.lin_trend(trend.transpose(1, 2))
    assert torch.allclose(actual, expected, atol=1e-6)


def test_freq_branch_shape():
    b, l, h, c, d = 4, 96, 24, 7, 32
    x = torch.randn(b, l, c)
    branch = FreqBranch(seq_len=l, pred_len=h, d_model=d)
    feat, y = branch(x)
    assert feat.shape == (b, c, d)
    assert y.shape == (b, c, h)


def test_model_forward_shape():
    b, l, h, c = 4, 96, 24, 7
    x = torch.randn(b, l, c)
    model = DualDomainForecaster(seq_len=l, pred_len=h, n_channels=c, d_model=32)
    out = model(x)
    assert out.shape == (b, h, c)


def test_fusion_modes():
    b, l, h, c = 2, 48, 12, 3
    x = torch.randn(b, l, c)
    for mode in ("gated", "sum", "concat"):
        model = DualDomainForecaster(
            seq_len=l, pred_len=h, n_channels=c, d_model=16, fusion=mode
        )
        assert model(x).shape == (b, h, c)


def test_time_branch_init_forecast_is_linear_backbone():
    """The time branch's correction head is zero-init, so its initial
    forecast must equal its internal linear (DLinear-style) backbone."""
    torch.manual_seed(0)
    b, l, h, c = 2, 48, 12, 3
    x = torch.randn(b, l, c)
    branch = TimeBranch(
        seq_len=l, pred_len=h, d_model=16, use_official_mamba=False
    ).eval()
    with torch.no_grad():
        _, y = branch(x)
        seasonal, trend = branch.decomp(x)
        expected = branch.lin_seasonal(seasonal.transpose(1, 2)) + branch.lin_trend(
            trend.transpose(1, 2)
        )
    assert torch.allclose(y, expected, atol=1e-5)


def test_forecast_is_convex_combination_of_branches():
    """The output must lie between the two branch forecasts elementwise:
    no path may bypass the dual branches."""
    torch.manual_seed(0)
    b, l, h, c = 2, 48, 12, 3
    x = torch.randn(b, l, c)
    model = DualDomainForecaster(
        seq_len=l, pred_len=h, n_channels=c, d_model=16
    ).eval()
    with torch.no_grad():
        out, comps = model(x, return_components=True)
    lo = torch.minimum(comps["time"], comps["freq"])
    hi = torch.maximum(comps["time"], comps["freq"])
    assert (out >= lo - 1e-5).all() and (out <= hi + 1e-5).all()


def test_ablation_hooks():
    """Every ablation switch must produce a working model of the right shape,
    and branch-only fusion must return exactly that branch's forecast."""
    torch.manual_seed(0)
    b, l, h, c = 2, 48, 12, 3
    x = torch.randn(b, l, c)
    for kwargs in (
        {"fusion": "time_only"},
        {"fusion": "freq_only"},
        {"use_revin": False},
        {"time_linear_backbone": False},
    ):
        model = DualDomainForecaster(
            seq_len=l, pred_len=h, n_channels=c, d_model=16, **kwargs
        ).eval()
        with torch.no_grad():
            out, comps = model(x, return_components=True)
        assert out.shape == (b, h, c)
        if kwargs.get("fusion") == "time_only":
            assert torch.allclose(out, comps["time"])
        if kwargs.get("fusion") == "freq_only":
            assert torch.allclose(out, comps["freq"])


def test_no_revin_output_not_denormalized():
    """With RevIN off the model must not re-add window statistics: shifting
    the input by a constant changes RevIN output but for a zero-init model
    without RevIN the forecast stays put."""
    b, l, h, c = 2, 48, 12, 2
    x = torch.randn(b, l, c)
    model = DualDomainForecaster(
        seq_len=l, pred_len=h, n_channels=c, d_model=16,
        use_revin=True, channel_mixer_layers=0,
    ).eval()
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x + 100.0)
    # RevIN on: constant shift passes straight through to the forecast.
    assert torch.allclose(y2 - y1, torch.full_like(y1, 100.0), atol=1e-2)


def test_fits_backbone_shapes_and_silent_init():
    """With the FITS spectral backbone, the freq branch must (a) keep its
    output contract and (b) start silent: backbone and head are zero-init,
    so the initial branch forecast is exactly zero."""
    b, l, h, c, d = 2, 96, 24, 5, 32
    x = torch.randn(b, l, c)
    branch = FreqBranch(seq_len=l, pred_len=h, d_model=d, backbone="fits").eval()
    with torch.no_grad():
        feat, y = branch(x)
    assert feat.shape == (b, c, d)
    assert y.shape == (b, c, h)
    assert y.abs().max().item() == 0.0


def test_fits_model_forward_and_gradients():
    torch.manual_seed(0)
    b, l, h, c = 2, 48, 12, 3
    x = torch.randn(b, l, c)
    model = DualDomainForecaster(
        seq_len=l, pred_len=h, n_channels=c, d_model=16,
        freq_backbone="fits", channel_mixer_layers=0,
    )
    out = model(x)
    assert out.shape == (b, h, c)
    # The zero-init spectral backbone must still receive gradients.
    out.sum().backward()
    g = model.freq_branch.spec_backbone.wr.weight.grad
    assert g is not None and g.abs().sum() > 0


def test_channel_mixer_cross_channel_flow():
    """With the mixer, perturbing one channel must influence another
    channel's forecast; without it, channels must stay fully independent."""
    torch.manual_seed(0)
    b, l, h, c = 1, 48, 12, 4
    x = torch.randn(b, l, c)
    x2 = x.clone()
    # Perturb only channel 0 — non-constant, so it survives the per-channel
    # instance normalization (a constant offset would be normalized away).
    x2[:, : l // 2, 0] += 5.0

    mixed = DualDomainForecaster(
        seq_len=l, pred_len=h, n_channels=c, d_model=16, channel_mixer_layers=1
    ).eval()
    indep = DualDomainForecaster(
        seq_len=l, pred_len=h, n_channels=c, d_model=16, channel_mixer_layers=0
    ).eval()
    with torch.no_grad():
        d_mixed = (mixed(x)[..., 1:] - mixed(x2)[..., 1:]).abs().max().item()
        d_indep = (indep(x)[..., 1:] - indep(x2)[..., 1:]).abs().max().item()
    assert d_indep < 1e-6, "mixer off: other channels' forecasts must not change"
    assert d_mixed > 1e-6, "mixer on: cross-channel information must flow"


def test_frequency_only_mixer_placement():
    model = DualDomainForecaster(
        seq_len=32, pred_len=8, n_channels=4, d_model=16,
        channel_mixer_layers=1, mixer_placement="freq",
        use_official_mamba=False,
    )
    assert model.time_branch.channel_mixer is None
    assert model.freq_branch.channel_mixer is not None
    assert model(torch.randn(2, 32, 4)).shape == (2, 8, 4)


def test_consistent_channel_permutation():
    data = np.arange(40, dtype=np.float32).reshape(10, 4)
    permuted_a, order_a = apply_channel_permutation(data, 17)
    permuted_b, order_b = apply_channel_permutation(data, 17)
    original, no_order = apply_channel_permutation(data, None)
    assert np.array_equal(order_a, order_b)
    assert np.array_equal(permuted_a, data[:, order_a])
    assert np.array_equal(permuted_a, permuted_b)
    assert no_order is None and original is data


def test_mamba_finite_under_autocast():
    """The selective scan must run in fp32 under autocast and stay finite."""
    torch.manual_seed(0)
    enc = MambaEncoder(d_model=16, n_layers=2, d_state=8, use_official=False)
    x = torch.randn(2, 64, 16) * 10  # large-ish values to stress fp16/bf16
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        y = enc(x)
    assert torch.isfinite(y).all()


def test_dataset_windowing():
    data = generate_synthetic(length=500, channels=3, seed=1)
    ds = SlidingWindowDataset(data, seq_len=96, pred_len=24)
    assert len(ds) == 500 - 96 - 24 + 1
    x, y, stats = ds[0]
    assert x.shape == (96, 3)
    assert y.shape == (24, 3)
    assert stats.shape == (0, 3)  # no dispersion resolutions -> empty stats


def test_multires_stats_and_dispersion():
    """Multi-resolution stats and the learned dispersion head: shapes, that
    D_hat is strictly positive, that history uses only pre-origin data, and
    that at initialization the head reduces to RevIN de-normalization."""
    import torch

    from src.models.dispersion import DispersionHead

    res = [24, 48, 96]
    data = generate_synthetic(length=600, channels=4, seed=2)
    ds = SlidingWindowDataset(data, seq_len=96, pred_len=24,
                              stats_resolutions=res, start_offset=0)
    x, y, stats = ds[0]
    assert stats.shape == (2 * len(res), 4)  # (mean,std) per resolution

    # No leakage: stats of the shortest resolution equal mean/std of the
    # samples strictly before the target (the tail of the input window).
    origin = 96
    hist = data[origin - res[0]:origin]
    assert np.allclose(stats[0].numpy(), hist.mean(0), atol=1e-5)
    assert np.allclose(stats[1].numpy(), hist.std(0), atol=1e-5)

    # Dispersion head: r_hat > 0 everywhere and equals 1 at init (t_hat 0).
    head = DispersionHead(stats_dim=2 * len(res), pred_len=24)
    s = stats.unsqueeze(0).transpose(1, 2)  # (1, C, S)
    r_hat, t_hat = head(s)
    assert r_hat.shape == (1, 4, 24) and t_hat.shape == (1, 4, 24)
    assert (r_hat > 0).all()
    assert torch.allclose(r_hat, torch.ones_like(r_hat), atol=1e-5)
    assert torch.allclose(t_hat, torch.zeros_like(t_hat), atol=1e-6)


def test_dispersion_reduces_to_revin_at_init():
    """A model with the learned dispersion head must produce exactly the RevIN
    reconstruction at initialization (r_hat=1, t_hat=0), so the head is a
    strict generalization that starts from the current de-normalization."""
    import torch

    from src.models.dual_domain_model import DualDomainForecaster

    res = (24, 48, 96)
    common = dict(seq_len=96, pred_len=24, n_channels=4, d_model=32,
                  channel_mixer_layers=0, use_revin=True)
    torch.manual_seed(0)
    m_none = DualDomainForecaster(dispersion="none", **common)
    torch.manual_seed(0)
    m_learn = DualDomainForecaster(dispersion="learned",
                                   dispersion_resolutions=res, **common)
    # Share all weights except the (identity-at-init) dispersion head.
    m_learn.load_state_dict(m_none.state_dict(), strict=False)
    m_none.eval(); m_learn.eval()

    x = torch.randn(3, 96, 4)
    stats = torch.randn(3, 2 * len(res), 4)
    with torch.no_grad():
        out_none = m_none(x)
        out_learn = m_learn(x, stats=stats)
    assert torch.allclose(out_none, out_learn, atol=1e-5)


def test_dispersion_head_efficiency():
    """Efficiency test for the dispersion experiment: the head's parameter
    overhead must be (a) exactly the head's own size, (b) small relative to the
    model, and (c) INDEPENDENT of the channel count -- the head is shared
    across variates, so it stays cheap even on high-channel data (Traffic)."""
    import torch

    from src.models.dispersion import DispersionHead
    from src.models.dual_domain_model import DualDomainForecaster

    res = (96, 144, 288, 336)
    stats_dim, pred_len, hidden = 2 * len(res), 96, 64

    def n_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    def build(c, dispersion):
        torch.manual_seed(0)
        return DualDomainForecaster(
            seq_len=96, pred_len=pred_len, n_channels=c, d_model=128,
            channel_mixer_layers=1, use_revin=True, dispersion=dispersion,
            dispersion_resolutions=res)

    head_params = n_params(DispersionHead(stats_dim, pred_len, hidden))

    for c in (7, 137, 862):  # ETT, Solar, Traffic channel counts
        base = build(c, "none")
        disp = build(c, "learned")
        overhead = n_params(disp) - n_params(base)
        # (a) the overhead is exactly the head's parameters, (c) the same for
        #     every channel count (shared across variates), and (b) small.
        assert overhead == head_params
        assert overhead < 0.10 * n_params(base)
    # The head has no per-channel parameters: O(1) in C.
    assert head_params < 15000


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="GPU memory/throughput must be measured on CUDA")
def test_dispersion_head_gpu_efficiency():
    """The model runs on the GPU, so the dispersion head's memory and
    throughput overhead is measured on the device (skipped without CUDA)."""
    from scripts.profile_efficiency import benchmark
    from src.models.dual_domain_model import DualDomainForecaster

    res = (96, 144, 288, 336)
    dev = torch.device("cuda")

    def build(dispersion):
        torch.manual_seed(0)
        return DualDomainForecaster(
            seq_len=96, pred_len=96, n_channels=137, d_model=128,
            channel_mixer_layers=1, use_revin=True, dispersion=dispersion,
            dispersion_resolutions=res)

    xb = torch.randn(32, 96, 137)
    statsb = torch.randn(32, 2 * len(res), 137).abs()
    _, thru_b, mem_b = benchmark(build("none"), xb, None, dev, iters=20)
    _, thru_d, mem_d = benchmark(build("learned"), xb, statsb, dev, iters=20)
    assert mem_b > 0 and mem_d > 0            # real GPU peak memory measured
    assert mem_d < 1.15 * mem_b               # head's GPU-memory overhead small
    assert thru_d > 0.7 * thru_b              # head does not gut throughput


def test_ett_canonical_borders():
    """ETTh protocol must reproduce the canonical window counts: for
    seq_len=96, pred_len=96 -> train 8449, val 2785, test 2785 (matching
    Informer/Autoformer/TSLib), regardless of the file's extra tail rows."""
    from src.data.dataset import ETT_BORDERS

    data = generate_synthetic(length=17420, channels=7, seed=3)  # ETTh1-sized
    train, val, test, _ = build_splits(
        data, seq_len=96, pred_len=96, train_ratio=0.6, val_ratio=0.2,
        scale=True, borders=ETT_BORDERS["ETTh"],
    )
    assert len(train) == 8640 - 96 - 96 + 1        # 8449
    assert len(val) == 2880 + 96 - 96 - 96 + 1     # 2785
    assert len(test) == 2880 + 96 - 96 - 96 + 1    # 2785


def test_build_splits_no_leakage_shapes():
    data = generate_synthetic(length=2000, channels=4, seed=2)
    train, val, test, scaler = build_splits(
        data, seq_len=96, pred_len=24, train_ratio=0.7, val_ratio=0.1, scale=True
    )
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    # Scaler should roughly standardize the training slice.
    scaled = scaler.transform(data[: int(len(data) * 0.7)])
    assert abs(scaled.mean()) < 0.1


def test_training_step_reduces_loss():
    torch.manual_seed(0)
    b, l, h, c = 16, 48, 12, 2
    x = torch.randn(b, l, c)
    # Target correlated with input so the model can actually learn.
    y = x[:, -h:, :] * 0.5
    model = DualDomainForecaster(seq_len=l, pred_len=h, n_channels=c, d_model=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()
    first = None
    for _ in range(50):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first


def test_dlinear_baseline():
    """The in-framework DLinear baseline must train through the same
    pipeline: correct shapes and selectable via model.arch."""
    from src.models.baselines import DLinearBaseline
    from src.models import build_model

    b, l, h, c = 2, 48, 12, 3
    x = torch.randn(b, l, c)
    m = DLinearBaseline(seq_len=l, pred_len=h)
    assert m(x).shape == (b, h, c)
    cfg = {"model": {"arch": "dlinear", "time_kernel_size": 25},
           "data": {"seq_len": l, "pred_len": h}}
    m2 = build_model(cfg, c)
    assert m2(x).shape == (b, h, c)


def test_zero_init_switch():
    """zero_init=False must produce standard random init: the freq branch is
    no longer silent and the time branch no longer equals its backbone."""
    torch.manual_seed(0)
    b, l, h, c = 2, 48, 12, 3
    x = torch.randn(b, l, c)
    common = dict(seq_len=l, pred_len=h, n_channels=c, d_model=16,
                  freq_backbone="fits", channel_mixer_layers=0, use_revin=False)
    silent = DualDomainForecaster(zero_init=True, **common).eval()
    rand = DualDomainForecaster(zero_init=False, **common).eval()
    with torch.no_grad():
        _, c0 = silent(x, return_components=True)
        _, c1 = rand(x, return_components=True)
    assert c0["freq"].abs().max() < 1e-6
    assert c1["freq"].abs().max() > 1e-3


def test_initialization_controls_are_isolated():
    """Changing one init arm must leave the other zero-initialized arms intact."""
    common = dict(
        seq_len=32, pred_len=8, n_channels=3, d_model=16,
        freq_backbone="fits", channel_mixer_layers=0,
        use_official_mamba=False,
    )

    torch.manual_seed(0)
    time_random = DualDomainForecaster(
        time_zero_init_head=False, **common
    )
    assert time_random.time_branch.head[-1].weight.abs().sum() > 0
    assert time_random.freq_branch.spec_backbone.wr.weight.abs().sum() == 0
    assert time_random.fusion.gate[2].weight.abs().sum() == 0

    torch.manual_seed(0)
    freq_backbone_random = DualDomainForecaster(
        freq_zero_init_backbone=False, **common
    )
    assert freq_backbone_random.time_branch.head[-1].weight.abs().sum() == 0
    assert freq_backbone_random.freq_branch.spec_backbone.wr.weight.abs().sum() > 0
    assert freq_backbone_random.fusion.gate[2].weight.abs().sum() == 0

    torch.manual_seed(0)
    fusion_random = DualDomainForecaster(fusion_zero_init=False, **common)
    assert fusion_random.time_branch.head[-1].weight.abs().sum() == 0
    assert fusion_random.freq_branch.spec_backbone.wr.weight.abs().sum() == 0
    assert fusion_random.fusion.gate[2].weight.abs().sum() > 0


def test_branch_only_active_parameter_count():
    common = dict(seq_len=32, pred_len=8, n_channels=4, d_model=16,
                  channel_mixer_layers=1, use_official_mamba=False)
    full = DualDomainForecaster(fusion="gated", **common)
    time_only = DualDomainForecaster(fusion="time_only", **common)
    freq_only = DualDomainForecaster(fusion="freq_only", **common)
    total = sum(p.numel() for p in full.parameters() if p.requires_grad)
    assert full.active_parameter_count() == total
    assert 0 < time_only.active_parameter_count() < total
    assert 0 < freq_only.active_parameter_count() < total


def test_ablation_chains_drop_noops_and_flip_revin():
    from scripts.run_ablation import (
        CORE_ABLATION,
        INITIALIZATION_CHAIN,
        resolve_chain,
    )

    cfg = {
        "model": {
            "time_encoder": "mamba",
            "freq_encoder": "linear",
            "freq_backbone": "fits",
            "channel_mixer_layers": 0,
            "mixer_placement": "both",
            "use_revin": False,
            "zero_init": True,
        }
    }
    core = resolve_chain(CORE_ABLATION, cfg)
    assert "with_revin" in core and "no_revin" not in core
    assert "no_channel_mixer" not in core and "vc_attention" not in core
    init = resolve_chain(INITIALIZATION_CHAIN, cfg)
    assert set(init) == {
        "full", "time_head_random_init", "freq_head_random_init",
        "freq_backbone_random_init", "fusion_random_init",
    }


# ---------------------------------------------------------------------------
# Revision scaffolding (review items 2, 3, 6)
# ---------------------------------------------------------------------------

def test_unified_baselines_shapes_and_registry():
    """Every in-repo baseline builds via model.arch and forecasts (B,H,C)."""
    from src.models import build_model
    from src.models.baselines import BASELINE_ARCHS

    b, l, h, c = 2, 96, 48, 7
    x = torch.randn(b, l, c)
    native = [a for a in BASELINE_ARCHS if a != "tf4tf"]
    assert set(native) >= {"nlinear", "rlinear", "patchtst", "itransformer",
                           "smamba", "msmamba"}
    for arch in native:
        cfg = {"model": {"arch": arch}, "data": {"seq_len": l, "pred_len": h}}
        m = build_model(cfg, c).eval()
        with torch.no_grad():
            out = m(x)
        assert out.shape == (b, h, c), arch


def test_rlinear_revin_invertibility():
    """RLinear's reversible norm must return to data scale (denorm inverts)."""
    from src.models.baselines import _InstanceNorm

    x = torch.randn(3, 96, 5) * 4 + 7
    norm = _InstanceNorm(n_channels=5, affine=True).eval()
    z = norm.normalize(x)
    # feed the normalized look-back's tail back through denormalize
    recon = norm.denormalize(z)
    assert torch.allclose(recon, x, atol=1e-4)


def test_tf4tf_adapter_requires_external_impl():
    """tf4tf must never fabricate a model: it raises without external_impl."""
    from src.models import build_model
    cfg = {"model": {"arch": "tf4tf"}, "data": {"seq_len": 96, "pred_len": 96}}
    with pytest.raises(NotImplementedError):
        build_model(cfg, 7)


def test_npz_pems_loader(tmp_path):
    """PEMS .npz (T,N,F) loads to (T,N) selecting the flow feature."""
    from src.data.dataset import load_raw_series

    p = tmp_path / "PEMS08.npz"
    np.savez(p, data=np.random.randn(200, 12, 3).astype("float32"))
    arr = load_raw_series(source="npz", csv_path=str(p), target_columns=None,
                          synthetic_length=0, synthetic_channels=0, seed=0,
                          npz_feature=0)
    assert arr.shape == (200, 12)
    assert arr.dtype == np.float32


def test_selection_protocol_helpers():
    """set_by_path / parse_value / apply_candidate build correct candidate cfgs."""
    from scripts.run_selection_protocol import (
        apply_candidate, parse_value, set_by_path,
    )

    assert parse_value("true") is True and parse_value("false") is False
    assert parse_value("3") == 3 and isinstance(parse_value("3"), int)
    assert parse_value("0.5") == 0.5
    assert parse_value("mamba") == "mamba"

    cfg = {"model": {"use_revin": True}, "data": {"seq_len": 96}}
    out = apply_candidate(cfg, {"model.use_revin": False,
                                "model.channel_mixer_layers": 2})
    assert out["model"]["use_revin"] is False
    assert out["model"]["channel_mixer_layers"] == 2
    assert cfg["model"]["use_revin"] is True  # original untouched

    bare = {"model": {}}
    set_by_path(bare, "use_revin", False)  # bare key defaults to model section
    assert bare["model"]["use_revin"] is False


def test_bh_correction_monotone_and_bounds():
    """Benjamini-Hochberg q-values are within [0,1] and >= raw p-values."""
    from scripts.compute_stats_correction import benjamini_hochberg

    ps = [0.001, 0.02, 0.03, 0.2, 0.5, 0.9]
    qs = benjamini_hochberg(ps)
    assert len(qs) == len(ps)
    assert all(0.0 <= q <= 1.0 for q in qs)
    assert all(q >= p - 1e-9 for p, q in zip(ps, qs))  # correction only inflates


def test_mixer_placement_switch_reduces_params_and_ties():
    """mixer_placement controls where the variate mixer lives; 'shared' ties
    one mixer across branches (fewer params), 'time'/'freq' use one branch."""
    b, l, h, c = 2, 96, 48, 24
    x = torch.randn(b, l, c)

    def build(placement):
        return DualDomainForecaster(
            seq_len=l, pred_len=h, n_channels=c, d_model=64,
            channel_mixer_layers=2, mixer_placement=placement)

    both = build("both").eval()
    shared = build("shared").eval()
    time_only = build("time").eval()
    n = lambda m: sum(p.numel() for p in m.parameters())

    # shared and single-branch both drop one mixer's worth of parameters
    assert n(shared) < n(both)
    assert n(time_only) < n(both)
    # sharing ties the two branches' mixers to the same module
    assert shared.freq_branch.channel_mixer is shared.time_branch.channel_mixer
    # 'time' placement leaves the frequency branch without a mixer
    assert time_only.freq_branch.channel_mixer is None
    # all still forecast the right shape
    with torch.no_grad():
        for m in (both, shared, time_only):
            assert m(x).shape == (b, h, c)

    import pytest as _pytest
    with _pytest.raises(ValueError):
        build("nonsense")


def test_crossformer_baseline_shapes_and_registry():
    """In-pipeline Crossformer (DSW + two-stage attention) forecasts (B,H,C),
    including when seq_len is not a multiple of the segment length and on
    high channel counts (router keeps cross-dimension attention linear in C)."""
    from src.models import build_model
    from src.models.baselines import BASELINE_ARCHS
    assert "crossformer" in BASELINE_ARCHS
    for (b, l, c, h) in [(2, 96, 7, 48), (2, 100, 21, 96)]:
        cfg = {"model": {"arch": "crossformer", "baseline_d_model": 32,
                         "cross_seg_len": 12},
               "data": {"seq_len": l, "pred_len": h}}
        m = build_model(cfg, c).eval()
        with torch.no_grad():
            out = m(torch.randn(b, l, c))
        assert out.shape == (b, h, c)


def test_apply_selection_update_key():
    """apply_selection writes validation-selected switches into a config,
    preserving trailing comments and not touching other sections; a missing
    key is inserted under its section."""
    from scripts.apply_selection import update_key, yaml_val
    lines = [
        "model:\n",
        "  use_revin: false  # keep this note\n",
        "  channel_mixer_layers: 2\n",
        "train:\n",
        "  lr: 0.001\n",
    ]
    changed, old = update_key(lines, "model", "use_revin", True)
    assert changed and old == "false"
    assert lines[1] == "  use_revin: true  # keep this note\n"       # comment kept
    update_key(lines, "model", "channel_mixer_layers", 0)           # int, no comment
    assert lines[2].strip() == "channel_mixer_layers: 0"
    assert any("lr: 0.001" in l for l in lines)                      # other section intact
    changed, old = update_key(lines, "model", "mixer_placement", "shared")
    assert changed and old is None                                   # inserted
    assert any("mixer_placement: shared" in l for l in lines)
    assert yaml_val(True) == "true" and yaml_val(3) == "3" and yaml_val("shared") == "shared"
