"""Two-GPU NCCL smoke test for the distributed KDA prefix prototype.

The rented A100 host used during development requires NCCL_P2P_DISABLE=1 even
for a one-element all-reduce; that is an instance transport workaround rather
than a requirement of the CP implementation.
"""

import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F


_CP_PATH = Path(__file__).parents[1] / "flash_kda" / "cp.py"
_SPEC = importlib.util.spec_from_file_location("flash_kda_cp_distributed", _CP_PATH)
assert _SPEC is not None and _SPEC.loader is not None
cp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cp
_SPEC.loader.exec_module(cp)


def make_segment(seed: int, tokens: int, device: torch.device):
    generator = torch.Generator().manual_seed(seed)
    heads, key_dim, value_dim = 2, 128, 128
    key = F.normalize(
        torch.randn((tokens, heads, key_dim), generator=generator), p=2, dim=-1
    ).to(device=device, dtype=torch.bfloat16)
    value = torch.randn(
        (tokens, heads, value_dim), generator=generator
    ).to(device=device, dtype=torch.bfloat16)
    log_decay = (
        -0.1 * torch.rand((tokens, heads, key_dim), generator=generator)
    ).to(device)
    beta = (
        0.1 + 0.8 * torch.rand((tokens, heads), generator=generator)
    ).to(device)
    return key, value, log_decay, beta


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError("this smoke test expects exactly two ranks")

    # Build the independent oracle before the first NCCL operation. CUDA work
    # after rank-divergent P2P can otherwise be ordered behind a collective on
    # the peer and turn a test-only synchronization pattern into a deadlock.
    if rank == 0:
        expected = cp.identity_transition(
            2, 128, 128, device=device, dtype=torch.float32
        )
    else:
        expected = cp.kda_segment_transition(*make_segment(20260820, 32, device))
    local = cp.kda_segment_transition(*make_segment(20260820 + rank, 32, device))
    prefix = cp.distributed_exclusive_prefix_transition(local)

    torch.testing.assert_close(prefix.matrix, expected.matrix, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(prefix.bias, expected.bias, rtol=2e-5, atol=2e-5)

    initial_state = torch.zeros((2, 128, 128), device=device, dtype=torch.bfloat16)
    actual_start = cp.apply_transition(initial_state, prefix)
    expected_start = cp.apply_transition(initial_state, expected)
    torch.testing.assert_close(actual_start, expected_start, rtol=2e-5, atol=2e-5)
    dist.barrier()
    if rank == 0:
        print("distributed CP prefix smoke test passed on 2 GPUs")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
