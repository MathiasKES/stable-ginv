"""Early-stop policy for the reconstruction optimization loop."""
import dataclasses


@dataclasses.dataclass(frozen=True)
class EarlyStopPolicy:
    """Stop a restart early once the gradient-matching loss converges.

    Defaults reproduce the original reconstruction-loop behavior exactly: the loop
    breaks with reason 'converged' the first iteration the loss drops below
    1e-6, and otherwise reports 'fixed_iterations' after running to completion.
    """

    convergence_loss: float = 1e-6
    converged_reason: str = "converged"
    default_reason: str = "fixed_iterations"

    def converged(self, loss: float) -> bool:
        """True when `loss` has dropped below the convergence threshold."""
        return loss < self.convergence_loss
