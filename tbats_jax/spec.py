"""Model structure specification. Static: fixed before JIT trace."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class TBATSSpec:
    """Fixed TBATS model structure.

    seasonal: tuple of (period, k_harmonics). Empty tuple means no seasonality.
    use_trend: include local-linear trend component
    use_damping: include damping parameter phi (only meaningful with trend)
    use_box_cox: include Box-Cox lambda parameter (requires positive y)
    p: AR order for innovations (0 = no AR)
    q: MA order for innovations (0 = no MA)
    """

    seasonal: Tuple[Tuple[float, int], ...] = field(default_factory=tuple)
    use_trend: bool = False
    use_damping: bool = False
    use_box_cox: bool = False
    p: int = 0
    q: int = 0

    def __post_init__(self):
        if self.use_damping and not self.use_trend:
            raise ValueError("use_damping requires use_trend=True")
        for m, k in self.seasonal:
            if k < 1 or 2 * k > m:
                raise ValueError(f"invalid (m={m}, k={k}): need 1 <= k <= m/2")
        if self.p < 0 or self.q < 0:
            raise ValueError("p and q must be non-negative")

    @property
    def n_gamma(self) -> int:
        return sum(k for _, k in self.seasonal)

    @property
    def state_dim(self) -> int:
        """State: level + [slope] + 2*n_gamma seasonal + p AR lags + q MA lags."""
        return 1 + int(self.use_trend) + 2 * self.n_gamma + self.p + self.q

    @property
    def n_smooth(self) -> int:
        """Count of smoothing scalars: [lambda], alpha, [phi], [beta], gamma1, gamma2, [ar], [ma]."""
        return (int(self.use_box_cox) + 1 + int(self.use_damping)
                + int(self.use_trend) + 2 * self.n_gamma
                + self.p + self.q)

    @property
    def n_params(self) -> int:
        """Total optimizer dim: smoothing params + seed state x0."""
        return self.n_smooth + self.state_dim
