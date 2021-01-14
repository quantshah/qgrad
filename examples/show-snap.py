import numpy as np  # this is ordinary numpy that we will

from qutip.visualization import plot_wigner, hinton
from qutip import Qobj, basis, ket2dm, displace, fidelity
from qutip.states import fock_dm

# Visualization
import matplotlib.pyplot as plt


def snap(hilbert_size, thetas):
    """
    Constructs the matrix for a SNAP gate operation
    that can be applied to a state.
    
    Args:
        hilbert_size (int): Hilbert space cuttoff
        thetas (:obj:`jnp.ndarray`): A vector of theta values to 
                apply SNAP operation
    
    Returns:
        :Qobj:`qutip.Qobj`: Qobj matrix representing the SNAP gate
    """
    op = 0 * np.eye(hilbert_size)
    for i, theta in enumerate(thetas):
        op += np.exp(1j * theta) * fock_dm(hilbert_size, i)
    return Qobj(op, type="oper")


def apply_blocks(alphas, thetas, initial_state):
    """Applies blocks of displace-snap-displace 
       operators to the initial state.
    
    Args:
        alphas (list): list of alpha paramters
                for the Displace operation
        thetas (list): vector of thetas for the
                SNAP operation
        initial_state(:obj:`jnp.array`): initial
                state to apply the blocks on
                
    Returns:
        :obj:`np.array`: evolved state after 
                applying T blocks to the initial
                state
    
    """

    if len(alphas) != len(thetas):
        raise ValueError("The number of alphas and theta vectors should be same")

    N = initial_state.shape[0]
    x = initial_state

    for t in range(len(alphas)):
        x = displace(N, alphas[t]) * x
        x = snap(N, thetas[t]) * x
        x = displace(N, -alphas[t]) * x

    return x


def show_state(state):
    """Shows the Hinton plot and Wigner function for the state"""
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    if state.shape[1] == 1:  # State is a ket
        dm = ket2dm(state)
        hinton(dm, ax=ax[0])
        plot_wigner(dm, ax=ax[1])


N = 6  # Hilbert space cutoff

initial_state = basis(N, 0)  # initial vacuum state
target_state = basis(N, 1)  # target state

alphas = [1.2614179, 0.79295063]  # Displace parameters
thetas = [
    [1.1619031, 0.94322765, -1.6765271, -0.24449646, -0.68798745, -0.60226285],
    [1.4677384, -0.08058921, 0.73822117, -1.1987586, 1.387459, -1.3684849],
]


show_state(initial_state)
plt.suptitle("Initial state")
plt.show()

x = apply_blocks(alphas, thetas, initial_state)
show_state(x)
plt.suptitle("Final state (fidelity {})".format(fidelity(target_state, x)))
plt.show()


# ## References
# [1] Fösel, Thomas, et al. "Efficient cavity control with SNAP gates." arXiv preprint arXiv:2004.14256 (2020).
