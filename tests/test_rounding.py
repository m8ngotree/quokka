import ml_dtypes
import numpy as np
import pytest
import torch

from quokka.reference.rounding import E2M1, E4M3, _ulp, round_to_grid

ORACLE = {"e2m1": ml_dtypes.float4_e2m1fn, "e4m3": ml_dtypes.float8_e4m3fn}


def _grid_values(grid):
    dtype = ORACLE[grid.name]
    bits = 16 if grid.name == "e2m1" else 256
    vals = np.arange(bits, dtype=np.uint8).view(dtype).astype(np.float32)
    vals = vals[np.isfinite(vals)]
    return np.unique(np.abs(vals))


def _test_points(grid):
    """Every grid point, every exact midpoint, and points just either side of each."""
    g = _grid_values(grid)
    mids = (g[:-1] + g[1:]) / 2
    pts = np.concatenate([g, mids, np.nextafter(mids, 0), np.nextafter(mids, np.inf)])
    rng = np.random.default_rng(0)
    pts = np.concatenate([pts, rng.uniform(0, grid.max_val, 4096).astype(np.float32)])
    return np.concatenate([pts, -pts]).astype(np.float32)


@pytest.mark.parametrize("grid", [E2M1, E4M3], ids=lambda g: g.name)
def test_nearest_matches_hardware_cast(grid):
    x = _test_points(grid)
    want = x.astype(ORACLE[grid.name]).astype(np.float32)
    got = round_to_grid(torch.from_numpy(x), grid).numpy()
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("grid", [E2M1, E4M3], ids=lambda g: g.name)
def test_saturates(grid):
    x = torch.tensor([grid.max_val * 1.01, -1e9, 1e30])
    got = round_to_grid(x, grid)
    assert got.tolist() == [grid.max_val, -grid.max_val, grid.max_val]


@pytest.mark.parametrize("grid", [E2M1, E4M3], ids=lambda g: g.name)
def test_stochastic_lands_on_a_neighbour(grid):
    x = torch.from_numpy(_test_points(grid))
    g = torch.from_numpy(_grid_values(grid))
    gen = torch.Generator().manual_seed(0)
    got = round_to_grid(x, grid, stochastic=True, generator=gen)
    mag = x.abs().clamp(max=grid.max_val)
    idx = torch.searchsorted(g, mag, right=True).clamp(max=len(g) - 1)
    lower, upper = g[(idx - 1).clamp(min=0)], g[idx]
    ok = (got.abs() == lower) | (got.abs() == upper) | (got.abs() == mag)
    assert ok.all()


@pytest.mark.parametrize("grid", [E2M1, E4M3], ids=lambda g: g.name)
def test_stochastic_is_unbiased(grid):
    x = torch.linspace(-grid.max_val, grid.max_val, 97)
    gen = torch.Generator().manual_seed(1)
    n = 20_000
    mean = round_to_grid(x.expand(n, -1), grid, stochastic=True, generator=gen).mean(0)
    # Std of the mean is at most ulp / (2 sqrt(n)); allow 5 sigma.
    ulp = _ulp(x.abs(), grid)
    assert ((mean - x).abs() <= 5 * ulp / (2 * n**0.5) + 1e-6).all()


@pytest.mark.parametrize("grid", [E2M1, E4M3], ids=lambda g: g.name)
def test_grid_values_are_fixed_points(grid):
    g = torch.from_numpy(_grid_values(grid))
    g = torch.cat([g, -g])
    gen = torch.Generator().manual_seed(2)
    assert torch.equal(round_to_grid(g, grid), g)
    assert torch.equal(round_to_grid(g, grid, stochastic=True, generator=gen), g)
