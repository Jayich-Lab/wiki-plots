import numpy as _n
from qutip import *
from random import random


class MSGateTimeEvolution():
    """Simulation of Molmer-Sorensen gate time evolution.

    The Hamiltonian equation refrence to Eq.4.24 to Eq.4.27 Martin A. Sepiol thesis.

    Expamples:
        ms = MSGateTimeEvolution()
        result =ms.gate_simulation()
        plt.plot(result[0], result[1], label = "|00>")
        plt.plot(result[0], result[2], label = "|01>")
        plt.plot(result[0], reslut[3], label = "|10>")
        plt.plot(result[0], reslut[4], label = "|11>")

    Args:
        Omega_0: float, angular Rabi frequency of the carrier transition.
        num_ions: int, number of ions.
        n: int, number of motional states included in simulation
        nbar_mode: mean thermal population of motional mode.
        eta: float, Lamb-Dicke parameter.
        phi_diff: the difference phase of the bichromatic field.
        phi_mean: the mean phase of the bichromatic field.
    """

    def __init__(self, Omega_0=2*_n.pi*100e3, num_ions=2, nbar_mode=0, eta=0.12,
                 phi_diff=0):
        self._Omega_0 = Omega_0
        self._num_ions = num_ions
        self._n = 20
        self._nbar_mode = nbar_mode
        self._eta = eta
        self._phi_diff = phi_diff
        self._phi_mean = random()*2*_n.pi

        self.e = Qobj([[0, 0], [0, 1]])
        self.g = Qobj([[1, 0], [0, 0]])

        self.ee = tensor(self.e, self.e, qeye(self._n))
        self.eg = tensor(self.e, self.g, qeye(self._n))
        self.ge = tensor(self.g, self.e, qeye(self._n))
        self.gg = tensor(self.g, self.g, qeye(self._n))

        self.sm = tensor(sigmam(), qeye(self._num_ions), qeye(self._n))+tensor(qeye(self._num_ions),
                                                                      sigmam(), qeye(self._n))
        self.sp = tensor(sigmap(), qeye(self._num_ions), qeye(self._n))+tensor(qeye(self._num_ions),
                                                                      sigmap(), qeye(self._n))

        self.a = tensor(qeye(self._num_ions), qeye(self._num_ions), destroy(self._n))
        self.ad = self.a.dag()

    @property
    def delta_g(self):
        """Gate detuning from the motional sidebands

        Retruns:
            float, gate detuning
        """
        return 2*self._eta*self._Omega_0

    @property
    def gate_time(self):
        """Gate time to prepare a Bell state.

        Returns:
            float, gate time
        """
        return 4*_n.pi/self.delta_g

    def _gate_H(self,rho_in, times, phi_offset=0, delta_LS = 0):
        """ Solving master equation evolution of a density matrix for the ms gate Hamiltonian.

        Args:
            rho_in: initial state density matrix before the gate operation.
            times: list of floats, gate times in seconds

        Returns:
            class, qutip.Result, which contains an array of "Result.expect" of expectation values for the times,
            or an array of "Result.states" of state vectors or density matrices corresponding to the times.
        """
        cons = self._eta*self._Omega_0/2

        H1  = -1j*cons*self.sp*self.ad*_n.exp(1j*(self._phi_diff + self._phi_mean))
        H1b = 1j*cons*self.sm*self.ad*_n.exp(1j*(self._phi_diff + self._phi_mean))
        H2  = 1j*cons*self.sm*self.a*_n.exp(-1j*(self._phi_diff + self._phi_mean))
        H2b = -1j*cons*self.sp*self.a*_n.exp(-1j*(self._phi_diff + self._phi_mean))
        def H1_coeff(t,args):
            return _n.exp(-1j*((self.delta_g+delta_LS)*t - phi_offset))
        def H2_coeff(t,args):
            return _n.exp(1j*((self.delta_g+delta_LS)*t - phi_offset))
        def H1b_coeff(t,args):
            return _n.exp(-1j*((self.delta_g-delta_LS)*t - phi_offset))
        def H2b_coeff(t,args):
            return _n.exp(1j*((self.delta_g-delta_LS)*t - phi_offset))

        options = Options(nsteps=1000000)
        H_ms = [[H1, H1_coeff], [H2, H2_coeff], [H1b, H1b_coeff], [H2b, H2b_coeff]]
        after_ms = mesolve(H_ms, rho_in, times, options = options)
        return after_ms

    def gate_simulation(self, nT=100, gates=1):
        """Gate time evolution simulation.

        Args:
            nT: number of times to generate.

        Returns:
            times: list of every interval time.
            population_ee: list, the population ions in |ee> state for each times during gate operation.
            population_eg: list, the population ions in |eg> state for each times during gate operation.
            population_ge: list, the population ions in |ge> state for each times during gate operation.
            population_gg: list, the population ions in |gg> state for each times during gate operation.
        """
        times = _n.linspace(0, gates*self.gate_time, nT)

        rho_in = tensor(fock_dm(self._num_ions, 1), fock_dm(self._num_ions, 1), thermal_dm(self._n, self._nbar_mode))
        output = self._gate_H(rho_in, times)
        population_ee = expect(self.ee*self.ee.dag(), output.states)
        population_eg = expect(self.eg*self.eg.dag(), output.states)
        population_ge = expect(self.ge*self.ge.dag(), output.states)
        population_gg = expect(self.gg*self.gg.dag(), output.states)
        return times, population_ee, population_eg, population_ge, population_gg
