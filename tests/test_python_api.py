import math

import pytest
import torch
import torch.nn.functional as F

import flash_kda


def make_inputs(T=33, H=1):
    B, D = 1, 128
    torch.manual_seed(0)
    q = F.normalize(
        torch.randn((B, T, H, D), dtype=torch.float32, device="cuda"),
        p=2,
        dim=-1,
    ).to(torch.bfloat16)
    return {
        "q": q,
        "k": q.clone(),
        "v": torch.randn_like(q),
        "g": torch.randn_like(q),
        "beta": torch.randn((B, T, H), dtype=torch.bfloat16, device="cuda"),
        "scale": 1.0 / math.sqrt(D),
        "A_log": torch.rand(H, dtype=torch.float32, device="cuda"),
        "dt_bias": torch.rand(H, D, dtype=torch.float32, device="cuda"),
        "lower_bound": -5.0,
    }


def test_pythonic_api_allocates_and_returns_out():
    args = make_inputs()

    out = flash_kda.fwd(**args)

    assert out.shape == args["q"].shape
    assert out.dtype == args["q"].dtype
    assert out.device == args["q"].device


def test_provided_out_is_returned():
    args = make_inputs()
    provided_out = torch.empty_like(args["q"])

    returned_out = flash_kda.fwd(**args, out=provided_out)

    assert returned_out is provided_out


def test_workspace_can_be_reused_for_sequential_calls():
    args = make_inputs(T=33)
    workspace = flash_kda.allocate_workspace(args["q"])
    data_ptr = workspace.data_ptr()

    first = flash_kda.fwd(**args, workspace=workspace).clone()
    second = flash_kda.fwd(**args, workspace=workspace).clone()
    smaller_args = make_inputs(T=17)
    smaller_with_reuse = flash_kda.fwd(**smaller_args, workspace=workspace).clone()
    smaller_automatic = flash_kda.fwd(**smaller_args).clone()

    assert workspace.data_ptr() == data_ptr
    assert torch.equal(first, second)
    assert torch.equal(smaller_with_reuse, smaller_automatic)


@pytest.mark.parametrize(
    "workspace_factory, error",
    [
        (lambda required: torch.empty(required, dtype=torch.float32, device="cuda"), "dtype uint8"),
        (lambda required: torch.empty(required - 1, dtype=torch.uint8, device="cuda"), "too small"),
        (lambda required: torch.empty(required, dtype=torch.uint8, device="cpu"), "CUDA tensor"),
    ],
)
def test_invalid_workspace_is_rejected(workspace_factory, error):
    args = make_inputs(T=17)
    required = flash_kda.allocate_workspace(args["q"]).numel()
    workspace = workspace_factory(required)

    with pytest.raises(RuntimeError, match=error):
        flash_kda.fwd(**args, workspace=workspace)


def test_public_exports_are_explicit():
    assert set(flash_kda.__all__) == {"allocate_workspace", "fwd", "get_workspace_size"}
