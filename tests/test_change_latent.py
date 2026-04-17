"""
Tests for the Variational Change Latent implementation.

Covers:
  1. change_latent.py — ChangeLatentEncoder, ChangePrior, FiLMConditioner, kl_divergence
  2. CrossSessionMemoryAttention — z_film parameter
  3. CrossPatientBank — z_posterior storage and retrieval
  4. CSMSAM — end-to-end forward with use_change_latent=True/False
  5. CombinedLoss — KL term with annealing
  6. Gradient flow — KL loss backpropagates through latent encoder

Run with:  python -m pytest tests/test_change_latent.py -v
       or: python tests/test_change_latent.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import pytest

from csmsam.modeling.change_latent import (
    ChangeLatentEncoder,
    ChangePrior,
    FiLMConditioner,
    SinusoidalTimeEncoder,
    kl_divergence,
    kl_beta,
)
from csmsam.modeling.cross_session_memory_attention import (
    CrossSessionMemoryAttention,
)
from csmsam.modeling.retrieval import CrossPatientBank
from csmsam.modeling.csm_sam import CSMSAM
from csmsam.losses.combined_loss import CombinedLoss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, C, D_Z = 4, 256, 64


def make_weeks(B=B):
    return torch.tensor([2.0, 3.0, 2.5, 4.0][:B])


# ---------------------------------------------------------------------------
# 1. SinusoidalTimeEncoder
# ---------------------------------------------------------------------------

class TestSinusoidalTimeEncoder:
    def test_output_shape(self):
        enc = SinusoidalTimeEncoder(d_out=C)
        w = make_weeks()
        out = enc(w)
        assert out.shape == (B, C), out.shape

    def test_long_input(self):
        enc = SinusoidalTimeEncoder(d_out=C)
        w = torch.tensor([2, 3], dtype=torch.long)
        out = enc(w)
        assert out.shape == (2, C)

    def test_zero_weeks_finite(self):
        enc = SinusoidalTimeEncoder(d_out=C)
        out = enc(torch.zeros(3))
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 2. ChangeLatentEncoder
# ---------------------------------------------------------------------------

class TestChangeLatentEncoder:
    def setup_method(self):
        self.enc = ChangeLatentEncoder(d_model=C, d_z=D_Z)
        self.f_pre = torch.randn(B, C)
        self.f_mid = torch.randn(B, C)
        self.weeks = make_weeks()

    def test_output_shapes(self):
        mu, lv = self.enc(self.f_pre, self.f_mid, self.weeks)
        assert mu.shape == (B, D_Z), mu.shape
        assert lv.shape == (B, D_Z), lv.shape

    def test_zero_init(self):
        mu, lv = self.enc(self.f_pre, self.f_mid, self.weeks)
        # Zero-initialised heads → outputs should be near 0
        assert mu.abs().max().item() < 0.1, f"mu not near 0 at init: {mu.abs().max()}"
        assert lv.abs().max().item() < 0.1, f"lv not near 0 at init: {lv.abs().max()}"

    def test_reparameterize(self):
        mu, lv = self.enc(self.f_pre, self.f_mid, self.weeks)
        z = ChangeLatentEncoder.reparameterize(mu, lv)
        assert z.shape == (B, D_Z)
        assert torch.isfinite(z).all()

    def test_gradients_flow(self):
        f_pre = self.f_pre.requires_grad_(True)
        f_mid = self.f_mid.requires_grad_(True)
        mu, lv = self.enc(f_pre, f_mid, self.weeks)
        loss = mu.sum() + lv.sum()
        loss.backward()
        assert f_pre.grad is not None
        assert f_mid.grad is not None


# ---------------------------------------------------------------------------
# 3. ChangePrior
# ---------------------------------------------------------------------------

class TestChangePrior:
    def setup_method(self):
        self.prior = ChangePrior(d_model=C, d_z=D_Z)
        self.f_pre = torch.randn(B, C)
        self.weeks = make_weeks()

    def test_output_shapes(self):
        mu, lv = self.prior(self.f_pre, self.weeks)
        assert mu.shape == (B, D_Z)
        assert lv.shape == (B, D_Z)

    def test_zero_init(self):
        mu, lv = self.prior(self.f_pre, self.weeks)
        assert mu.abs().max().item() < 0.1
        assert lv.abs().max().item() < 0.1

    def test_gradients_flow(self):
        f_pre = self.f_pre.requires_grad_(True)
        mu, lv = self.prior(f_pre, self.weeks)
        (mu.sum() + lv.sum()).backward()
        assert f_pre.grad is not None


# ---------------------------------------------------------------------------
# 4. FiLMConditioner
# ---------------------------------------------------------------------------

class TestFiLMConditioner:
    def setup_method(self):
        self.film = FiLMConditioner(d_z=D_Z, d_model=C)
        self.z = torch.randn(B, D_Z)

    def test_output_shapes(self):
        gamma, beta = self.film(self.z)
        assert gamma.shape == (B, C)
        assert beta.shape  == (B, C)

    def test_zero_init_identity(self):
        # At init (zero weights), gamma≈0 and beta≈0, so query is unchanged
        gamma, beta = self.film(self.z)
        query = torch.randn(B, 16, C)
        query_film = query * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        # With zero init, query_film ≈ query
        assert (query_film - query).abs().max().item() < 0.1

    def test_gradients_flow(self):
        z = self.z.requires_grad_(True)
        gamma, beta = self.film(z)
        (gamma.sum() + beta.sum()).backward()
        assert z.grad is not None


# ---------------------------------------------------------------------------
# 5. kl_divergence
# ---------------------------------------------------------------------------

class TestKLDivergence:
    def test_zero_at_identity(self):
        mu = torch.zeros(B, D_Z)
        lv = torch.zeros(B, D_Z)  # σ=1
        kl = kl_divergence(mu, lv, mu.clone(), lv.clone())
        assert kl.item() < 1e-5, f"KL should be 0 when q==p, got {kl.item()}"

    def test_positive(self):
        mu_q = torch.randn(B, D_Z)
        lv_q = torch.randn(B, D_Z)
        mu_p = torch.zeros(B, D_Z)
        lv_p = torch.zeros(B, D_Z)
        kl = kl_divergence(mu_q, lv_q, mu_p, lv_p)
        assert kl.item() >= 0, f"KL must be non-negative, got {kl.item()}"

    def test_scalar_output(self):
        mu = torch.randn(B, D_Z)
        lv = torch.zeros(B, D_Z)
        kl = kl_divergence(mu, lv, torch.zeros(B, D_Z), torch.zeros(B, D_Z))
        assert kl.ndim == 0

    def test_gradients_through_posterior(self):
        mu_q = torch.randn(B, D_Z, requires_grad=True)
        lv_q = torch.zeros(B, D_Z, requires_grad=True)
        mu_p = torch.zeros(B, D_Z)
        lv_p = torch.zeros(B, D_Z)
        kl = kl_divergence(mu_q, lv_q, mu_p, lv_p)
        kl.backward()
        assert mu_q.grad is not None
        assert lv_q.grad is not None


# ---------------------------------------------------------------------------
# 6. kl_beta schedule
# ---------------------------------------------------------------------------

class TestKLBeta:
    def test_zero_at_start(self):
        assert kl_beta(0, 0.1, 10) == 0.0

    def test_ramps_linearly(self):
        assert abs(kl_beta(5, 0.1, 10) - 0.05) < 1e-6

    def test_clamps_at_max(self):
        assert kl_beta(10, 0.1, 10) == 0.1
        assert kl_beta(100, 0.1, 10) == 0.1

    def test_zero_warmup(self):
        assert kl_beta(0, 0.1, 0) == 0.1


# ---------------------------------------------------------------------------
# 7. CrossSessionMemoryAttention with z_film
# ---------------------------------------------------------------------------

class TestCrossSessionAttentionZFilm:
    def setup_method(self):
        self.attn = CrossSessionMemoryAttention(d_model=C, num_heads=4)
        B_, HW_, N_pre_ = 2, 64, 128
        self.curr = torch.randn(B_, HW_, C)
        self.M_pre = torch.randn(B_, N_pre_, C)
        self.weeks = torch.tensor([3, 3], dtype=torch.long)

    def test_without_z_film(self):
        out, attn_w, gate = self.attn(self.curr, self.M_pre, None, self.weeks)
        assert out.shape == self.curr.shape
        assert torch.isfinite(out).all()

    def test_with_z_film(self):
        B_ = self.curr.shape[0]
        gamma = torch.zeros(B_, C)
        beta  = torch.zeros(B_, C)
        out_z, _, _ = self.attn(self.curr, self.M_pre, None, self.weeks,
                                z_film=(gamma, beta))
        out_n, _, _ = self.attn(self.curr, self.M_pre, None, self.weeks,
                                z_film=None)
        # With zero gamma/beta, FiLM is identity → outputs should be identical
        assert torch.allclose(out_z, out_n, atol=1e-5), \
            "Zero FiLM should not change output"

    def test_nonzero_z_film_changes_output(self):
        B_ = self.curr.shape[0]
        gamma = torch.randn(B_, C) * 0.5
        beta  = torch.randn(B_, C) * 0.5
        out_z, _, _ = self.attn(self.curr, self.M_pre, None, self.weeks,
                                z_film=(gamma, beta))
        out_n, _, _ = self.attn(self.curr, self.M_pre, None, self.weeks,
                                z_film=None)
        assert not torch.allclose(out_z, out_n, atol=1e-4), \
            "Non-zero FiLM should change output"

    def test_z_film_gradient_flows(self):
        B_ = self.curr.shape[0]
        gamma = torch.randn(B_, C, requires_grad=True)
        beta  = torch.randn(B_, C, requires_grad=True)
        curr  = self.curr.requires_grad_(True)
        out, _, _ = self.attn(curr, self.M_pre, None, self.weeks,
                              z_film=(gamma, beta))
        out.sum().backward()
        assert gamma.grad is not None
        assert beta.grad is not None
        assert curr.grad is not None


# ---------------------------------------------------------------------------
# 8. CrossPatientBank z_posterior
# ---------------------------------------------------------------------------

class TestCrossPPatientBankZ:
    def _make_bank(self, n=10, d_z=D_Z):
        bank = CrossPatientBank()
        for i in range(n):
            bank.add(
                patient_id=f"p{i}",
                pre_summary=torch.randn(C),
                change_template=torch.randn(16, C),
                weeks_elapsed=3,
                z_posterior=torch.randn(d_z),
            )
        return bank

    def test_has_z_true(self):
        bank = self._make_bank()
        assert bank.has_z()

    def test_has_z_false_without_z(self):
        bank = CrossPatientBank()
        bank.add("p0", torch.randn(C), torch.randn(16, C), 3, z_posterior=None)
        assert not bank.has_z()

    def test_topk_z_shape(self):
        bank = self._make_bank(10)
        query = torch.randn(2, C)
        z_ret = bank.topk_z(query, k=3)
        assert z_ret is not None
        assert z_ret.shape == (2, D_Z), z_ret.shape

    def test_topk_z_returns_none_without_z(self):
        bank = CrossPatientBank()
        bank.add("p0", torch.randn(C), torch.randn(16, C), 3, z_posterior=None)
        query = torch.randn(1, C)
        result = bank.topk_z(query, k=1)
        assert result is None

    def test_topk_z_finite(self):
        bank = self._make_bank(5)
        query = torch.randn(3, C)
        z = bank.topk_z(query, k=3)
        assert torch.isfinite(z).all()

    def test_save_load_preserves_z(self, tmp_path):
        bank = self._make_bank(5)
        path = tmp_path / "bank.pt"
        bank.save(str(path))
        bank2 = CrossPatientBank.load(str(path))
        assert bank2.has_z()
        q = torch.randn(1, C)
        z1 = bank.topk_z(q, k=3)
        z2 = bank2.topk_z(q, k=3)
        assert torch.allclose(z1, z2, atol=1e-5)

    def test_backward_compat_no_z_in_file(self, tmp_path):
        """Load a bank saved without z_posteriors_t — should not crash."""
        bank = CrossPatientBank()
        for i in range(3):
            bank.add(f"p{i}", torch.randn(C), torch.randn(16, C), 3)
        path = str(tmp_path / "old_bank.pt")
        # Save without z field (simulate old format)
        import torch as _t
        bank._rebuild()
        _t.save({
            "ids": bank.ids,
            "summaries_t": bank._summaries_t,
            "templates_t": bank._templates_t,
            "weeks": bank.weeks,
            # no z_posteriors_t key
        }, path)
        bank2 = CrossPatientBank.load(path)
        assert not bank2.has_z()   # gracefully no z


# ---------------------------------------------------------------------------
# 9. CSMSAM end-to-end (no SAM2, fallback encoder)
# ---------------------------------------------------------------------------

def _make_csmsam(use_change_latent=True, use_retrieval=True):
    """Build a CSMSAM with no real SAM2 (uses internal fallback encoder)."""
    return CSMSAM(
        sam2_model=None,
        d_model=C,
        num_heads=4,
        n_memory_frames=2,
        spatial_pool_size=4,
        use_cross_patient_retrieval=use_retrieval,
        retrieval_k=3,
        retrieval_n_tokens=4,
        use_change_latent=use_change_latent,
        d_z=D_Z,
        latent_alpha=0.5,
    )


class TestCSMSAMEndToEnd:
    B_ = 2
    H_, W_ = 64, 64

    def _make_inputs(self):
        B, H, W = self.B_, self.H_, self.W_
        mid_imgs   = torch.randn(B, 3, H, W)
        pre_imgs_5d = torch.randn(B, 2, 3, H, W)
        pre_masks_5d = torch.rand(B, 2, 1, H, W)
        weeks      = torch.tensor([3, 2], dtype=torch.long)
        pre_gtvp   = (torch.rand(B, 1, H, W) > 0.7).float()
        pre_gtvn   = (torch.rand(B, 1, H, W) > 0.7).float()
        return mid_imgs, pre_imgs_5d, pre_masks_5d, weeks, pre_gtvp, pre_gtvn

    def test_without_change_latent(self):
        model = _make_csmsam(use_change_latent=False)
        mid_imgs, pre_imgs_5d, pre_masks_5d, weeks, pgp, pgn = self._make_inputs()
        M_pre = model.encode_pre_rt(pre_imgs_5d, pre_masks_5d)
        model.reset_mid_session_memory()
        out = model(
            mid_images=mid_imgs,
            M_pre=M_pre,
            pre_images=mid_imgs,
            weeks_elapsed=weeks,
            pre_gtvp_mask=pgp,
            pre_gtvn_mask=pgn,
        )
        assert out["masks"].shape == (self.B_, 2, self.H_, self.W_)
        assert out["z_posterior"] is None
        assert out["kl_loss"] is None

    def test_with_change_latent_training(self):
        model = _make_csmsam(use_change_latent=True)
        model.train()
        mid_imgs, pre_imgs_5d, pre_masks_5d, weeks, pgp, pgn = self._make_inputs()
        M_pre = model.encode_pre_rt(pre_imgs_5d, pre_masks_5d)
        model.reset_mid_session_memory()
        out = model(
            mid_images=mid_imgs,
            M_pre=M_pre,
            pre_images=mid_imgs,
            weeks_elapsed=weeks,
            pre_gtvp_mask=pgp,
            pre_gtvn_mask=pgn,
        )
        assert out["masks"].shape == (self.B_, 2, self.H_, self.W_)
        assert out["z_posterior"] is not None
        assert out["z_posterior"].shape == (self.B_, D_Z)
        assert out["kl_loss"] is not None
        assert out["kl_loss"].ndim == 0
        assert torch.isfinite(out["kl_loss"])

    def test_with_change_latent_inference(self):
        model = _make_csmsam(use_change_latent=True)
        model.eval()
        mid_imgs, pre_imgs_5d, pre_masks_5d, weeks, pgp, pgn = self._make_inputs()
        M_pre = model.encode_pre_rt(pre_imgs_5d, pre_masks_5d)
        model.reset_mid_session_memory()
        with torch.no_grad():
            out = model(
                mid_images=mid_imgs,
                M_pre=M_pre,
                pre_images=None,   # no pre_images → inference mode → prior z
                weeks_elapsed=weeks,
                pre_gtvp_mask=pgp,
                pre_gtvn_mask=pgn,
            )
        assert out["masks"].shape == (self.B_, 2, self.H_, self.W_)
        assert out["z_posterior"] is not None
        assert out["kl_loss"] is None  # no KL at inference

    def test_kl_loss_backpropagates(self):
        model = _make_csmsam(use_change_latent=True)
        model.train()
        mid_imgs, pre_imgs_5d, pre_masks_5d, weeks, pgp, pgn = self._make_inputs()
        M_pre = model.encode_pre_rt(pre_imgs_5d, pre_masks_5d)
        model.reset_mid_session_memory()
        out = model(
            mid_images=mid_imgs,
            M_pre=M_pre,
            pre_images=mid_imgs,
            weeks_elapsed=weeks,
            pre_gtvp_mask=pgp,
            pre_gtvn_mask=pgn,
        )
        loss = out["masks"].sum() + out["kl_loss"]
        loss.backward()
        # Check latent encoder parameters have gradients
        for name, p in model.latent_encoder.named_parameters():
            assert p.grad is not None, f"No grad for latent_encoder.{name}"

    def test_trainable_param_count_increases(self):
        base  = _make_csmsam(use_change_latent=False)
        full  = _make_csmsam(use_change_latent=True)
        c_base = base.count_trainable_params()["total"]
        c_full = full.count_trainable_params()["total"]
        added = c_full - c_base
        assert added > 200_000, f"Expected >200k new params, got {added}"
        assert added < 1_000_000, f"Expected <1M new params, got {added}"

    def test_inference_with_bank_z(self):
        model = _make_csmsam(use_change_latent=True, use_retrieval=True)
        model.eval()

        # Build a small fake bank with z values
        bank = CrossPatientBank()
        for i in range(5):
            bank.add(
                f"p{i}",
                torch.randn(C),
                torch.randn(4, C),
                weeks_elapsed=3,
                z_posterior=torch.randn(D_Z),
            )
        model.set_patient_bank(bank)

        mid_imgs, pre_imgs_5d, pre_masks_5d, weeks, pgp, pgn = self._make_inputs()
        M_pre = model.encode_pre_rt(pre_imgs_5d, pre_masks_5d)
        model.reset_mid_session_memory()
        with torch.no_grad():
            out = model(
                mid_images=mid_imgs,
                M_pre=M_pre,
                pre_images=None,
                weeks_elapsed=weeks,
                pre_gtvp_mask=pgp,
                pre_gtvn_mask=pgn,
            )
        assert out["z_posterior"] is not None
        assert torch.isfinite(out["masks"]).all()


# ---------------------------------------------------------------------------
# 10. CombinedLoss with KL term
# ---------------------------------------------------------------------------

class TestCombinedLossKL:
    B_, H_, W_ = 2, 32, 32

    def test_kl_added_when_provided(self):
        loss_fn = CombinedLoss(lambda_dice=1.0, lambda_bce=1.0, lambda_kl=0.1)
        pred    = torch.zeros(self.B_, 2, self.H_, self.W_)
        target  = torch.zeros(self.B_, 2, self.H_, self.W_)
        kl      = torch.tensor(1.0)

        losses_no_kl = loss_fn(pred, target)
        losses_kl    = loss_fn(pred, target, kl_loss=kl, kl_beta=1.0)

        assert "kl" in losses_kl
        diff = losses_kl["total"].item() - losses_no_kl["total"].item()
        assert abs(diff - 0.1) < 1e-5, f"Expected diff 0.1, got {diff}"

    def test_kl_zero_at_beta_zero(self):
        loss_fn = CombinedLoss(lambda_dice=1.0, lambda_bce=1.0, lambda_kl=0.1)
        pred    = torch.zeros(self.B_, 2, self.H_, self.W_)
        target  = torch.zeros(self.B_, 2, self.H_, self.W_)
        kl      = torch.tensor(5.0)

        losses_kl0 = loss_fn(pred, target, kl_loss=kl, kl_beta=0.0)
        assert "kl" not in losses_kl0

    def test_kl_scales_with_beta(self):
        loss_fn = CombinedLoss(lambda_dice=1.0, lambda_bce=1.0, lambda_kl=1.0)
        pred    = torch.zeros(self.B_, 2, self.H_, self.W_)
        target  = torch.zeros(self.B_, 2, self.H_, self.W_)
        kl      = torch.tensor(2.0)

        losses_half = loss_fn(pred, target, kl_loss=kl, kl_beta=0.5)
        base        = loss_fn(pred, target)
        diff = losses_half["total"].item() - base["total"].item()
        assert abs(diff - 1.0) < 1e-5  # lambda_kl=1.0 * beta=0.5 * kl=2.0 = 1.0

    def test_kl_none_no_effect(self):
        loss_fn = CombinedLoss(lambda_dice=1.0, lambda_bce=1.0, lambda_kl=0.1)
        pred    = torch.zeros(self.B_, 2, self.H_, self.W_)
        target  = torch.zeros(self.B_, 2, self.H_, self.W_)

        losses_no_kl = loss_fn(pred, target, kl_loss=None, kl_beta=1.0)
        losses_base  = loss_fn(pred, target)
        assert abs(losses_no_kl["total"].item() - losses_base["total"].item()) < 1e-6


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    suites = [
        TestSinusoidalTimeEncoder,
        TestChangeLatentEncoder,
        TestChangePrior,
        TestFiLMConditioner,
        TestKLDivergence,
        TestKLBeta,
        TestCrossSessionAttentionZFilm,
        TestCrossPPatientBankZ,
        TestCSMSAMEndToEnd,
        TestCombinedLossKL,
    ]

    total = 0
    failed = 0

    for suite_cls in suites:
        suite = suite_cls()
        methods = [m for m in dir(suite) if m.startswith("test_")]
        for method_name in methods:
            total += 1
            # Provide a simple tmp_path stand-in for tests that need it
            import tempfile, pathlib
            with tempfile.TemporaryDirectory() as td:
                try:
                    if "setup_method" in dir(suite):
                        suite.setup_method()
                    method = getattr(suite, method_name)
                    import inspect
                    sig = inspect.signature(method)
                    if "tmp_path" in sig.parameters:
                        method(tmp_path=pathlib.Path(td))
                    else:
                        method()
                    print(f"  PASS  {suite_cls.__name__}::{method_name}")
                except Exception as e:
                    failed += 1
                    print(f"  FAIL  {suite_cls.__name__}::{method_name}")
                    traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Results: {total - failed}/{total} passed, {failed} failed")
    if failed:
        sys.exit(1)
    else:
        print("All tests passed.")
