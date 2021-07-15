import numpy as _np
import qutip as _qt


class MSGateSimulation():
    """Simulation of Molmer-Sorensen gate.

    The Hamiltonian equation refers to Eq. 4.24-Eq.4.27 in Sepiol2016.
    The num of ions is fixed to 2 in our system and the number of motional
    states (n) is fixed to 20, so this class only simulates up to 19 motional quantas.

    Example:
        ms = MSGateSimulation()
        gate_time = ms.gate_time
        result1 = ms.time_scan(gate_time)
        plt.plot(result1[0], result1[1], label="|11>")
        plt.plot(result1[0], result1[2], label="|10>")
        plt.plot(result1[0], reslut1[3], label="|01>")
        plt.plot(result1[0], reslut1[4], label="|00>")
        plt.plot(result1[0], reslut1[5], label="fidelity")

        result2 = ms.detuning_scan()
        plt.plot(result2[0], result2[1], label="P11")
        plt.plot(result2[0], np.array(result2[2]) + np.array(result2[3]), label="P10+P01")
        plt.plot(result2[0], result2[4], label="P00")
        plt.plot(result2[0], result2[5], label="fidelity")

        result3 = ms.parity_scan()
        plt.plot(result3[0], result3[1])

    Args:
        Omega_0: float, angular Rabi frequency of the carrier transition in radians.
        delta_g: float, bichromatic light angular frequency detuning from the motional
            sidebands in radians/s.
        nbar: float, mean phonon occupancy.
        eta: float, Lamb-Dicke parameter.
        phi_diff: float, the difference of the blue sideband laser phase and red side band
            laser phase in radians.
        phi_mean: float, the mean phase of the bichromatic field in radians, relative to
            the gate starts at a maximum of the intensity of amplitude-modulated
            beam at phi_mean=0, starts at minimum of the intensity of the beam
            at phi_mean=pi/2.
        delta_LS: float, bichromatic field asymmetric angular frequency detuning from the
            carrier transition in radians, defined by ((omega_b+omega_r)/2-omega_0)，usually
            set at 0.
        n_dot: float, unit in quanta/s, motional heating rate.
        tau_mot: float, unit in s, motional coherence time for a motional superposition
            state (|n> + |n'>).
    """

    def __init__(self, omega_0=2*_np.pi*100e3, delta_g=2*_np.pi*24e3, nbar=0, eta=0.12,
                 phi_diff=0, phi_mean=0, delta_LS=0, n_dot=0, tau_mot=0):
        self._omega_0 = omega_0
        self._nbar = nbar
        self._eta = eta
        self._phi_diff = phi_diff
        self._phi_mean = phi_mean
        self._delta_LS = delta_LS
        self._num_ions = 2
        self._phonons = 20
        self._n_dot = n_dot
        self._tau_mot = tau_mot
        self._delta_g = delta_g

        self._e = _qt.Qobj([[0, 0], [0, 1]])
        self._g = _qt.Qobj([[1, 0], [0, 0]])

        self._ee = _qt.tensor(self._e, self._e, _qt.qeye(self._phonons))
        self._eg = _qt.tensor(self._e, self._g, _qt.qeye(self._phonons))
        self._ge = _qt.tensor(self._g, self._e, _qt.qeye(self._phonons))
        self._gg = _qt.tensor(self._g, self._g, _qt.qeye(self._phonons))

        self._sm = (_qt.tensor(_qt.sigmam(), _qt.qeye(self._num_ions), _qt.qeye(self._phonons))
                    + _qt.tensor(_qt.qeye(self._num_ions), _qt.sigmam(), _qt.qeye(self._phonons)))
        self._sp = (_qt.tensor(_qt.sigmap(), _qt.qeye(self._num_ions), _qt.qeye(self._phonons))
                    + _qt.tensor(_qt.qeye(self._num_ions), _qt.sigmap(), _qt.qeye(self._phonons)))

        self._a = _qt.tensor(_qt.qeye(self._num_ions), _qt.qeye(self._num_ions),
                             _qt.destroy(self._phonons))
        self._ad = self._a.dag()

    @property
    def gate_time(self):
        """Gate time to prepare a Bell state in s."""
        return 2*_np.pi/self._delta_g

    def _bell_state(self, initial_state="00"):
        """The ideal Bell state after taking in a quantum state and actually acting
        on it with the gate operation.

        Refers to the eq.(4.42) - eq.(4.45) in Sepiol2016.

        Args:
            initial_state: string, initial state at the begining of gate operation,
                can be either of "00", "11", "01", "10".

        Returns:
            qobj, Bell state.
        """
        if initial_state == "00":
            bell_state = (_qt.tensor(_qt.basis(2, 0), _qt.basis(2, 0)) + _np.exp(-2*1j
                          * self._phi_mean) * 1j * _qt.tensor(_qt.basis(2, 1), _qt.basis(2, 1)))
        elif initial_state == "11":
            bell_state = (_np.exp(2*1j*self._phi_mean)*_qt.tensor(_qt.basis(2, 0), _qt.basis(2, 0))
                          - 1j*_qt.tensor(_qt.basis(2, 1), _qt.basis(2, 1)))
        elif initial_state == "01":
            bell_state = (_qt.tensor(_qt.basis(2, 0), _qt.basis(2, 1))
                          + 1j*_qt.tensor(_qt.basis(2, 1), _qt.basis(2, 0)))
        elif initial_state == "10":
            bell_state = (_qt.tensor(_qt.basis(2, 0), _qt.basis(2, 1))
                          - 1j*_qt.tensor(_qt.basis(2, 1), _qt.basis(2, 0)))

        return bell_state.unit()

    def _gate_H(self, delta_g, rho_in, times):
        """Solving master equation evolution of a density matrix for the ms gate Hamiltonian.

        Args:
            delta_g: float, bichromatic light detuning from the motional sidebands in radians/s.
            rho_in: initial state density matrix before the gate operation.
            times: list of floats, gate times in seconds

        Returns:
            qutip.Result object, returned value from qutip.mesolve.
        """
        cons = self._eta*self._omega_0/2

        H1 = -1j*cons*self._sp*self._ad*_np.exp(1j*(self._phi_diff + self._phi_mean))
        H1b = 1j*cons*self._sm*self._ad*_np.exp(1j*(self._phi_diff + self._phi_mean))
        H2 = 1j*cons*self._sm*self._a*_np.exp(-1j*(self._phi_diff + self._phi_mean))
        H2b = -1j*cons*self._sp*self._a*_np.exp(-1j*(self._phi_diff + self._phi_mean))

        def H1_coeff(t, args):
            return _np.exp(-1j*((delta_g + self._delta_LS)*t))

        def H2_coeff(t, args):
            return _np.exp(1j*((delta_g + self._delta_LS)*t))

        def H1b_coeff(t, args):
            return _np.exp(-1j*((delta_g - self._delta_LS)*t))

        def H2b_coeff(t, args):
            return _np.exp(1j*((delta_g - self._delta_LS)*t))

        c_ops = []
        if self._n_dot != 0:
            c_ops.append(_np.sqrt(self._n_dot)*self._a)
            c_ops.append(_np.sqrt(self._n_dot)*self._ad)
        if self._tau_mot != 0:
            c_ops.append(_np.sqrt(2/self._tau_mot)*self._ad*self._a)

        options = _qt.Options(nsteps=1000000)
        H_ms = [[H1, H1_coeff], [H2, H2_coeff], [H1b, H1b_coeff], [H2b, H2b_coeff]]
        after_ms = _qt.mesolve(H_ms, rho_in, times, c_ops, options=options)
        return after_ms

    def time_scan(self, end_time, nT=100):
        """Gate time evolution simulation.

        Args:
            end_time: float, the simulation end time in s.
            nT: int, number of times to calculate the populations for.

        Returns:
            (times, population_ee, population_eg, population_ge, population_gg, fidelity)
            times: list of float, times that the populations are calculated at.
            population_ee: list of floats, the population in |ee> state.
            population_eg: list of floats, the population in |eg> state.
            population_ge: list of floats, the population in |ge> state.
            population_gg: list of floats, the population in |gg> state.
            fidelity: list of floats, the similarity between the target and the actual output state
                of the gate
        """
        times = _np.linspace(0, end_time, nT)
        rho_in = _qt.tensor(_qt.fock_dm(self._num_ions, 0), _qt.fock_dm(self._num_ions, 0),
                            _qt.thermal_dm(self._phonons, self._nbar))
        rho_target = self._bell_state()*self._bell_state().dag()

        output = self._gate_H(self._delta_g, rho_in, times)
        rhos_after_gate = output.states

        population_ee = _qt.expect(self._ee*self._ee.dag(), rhos_after_gate)
        population_eg = _qt.expect(self._eg*self._eg.dag(), rhos_after_gate)
        population_ge = _qt.expect(self._ge*self._ge.dag(), rhos_after_gate)
        population_gg = _qt.expect(self._gg*self._gg.dag(), rhos_after_gate)

        fidelity = []
        for kk in range(len(times)):
            fidelity.append(_qt.fidelity(rho_target, _qt.ptrace(rhos_after_gate[kk], [0, 1])))

        return times, population_ee, population_eg, population_ge, population_gg, fidelity

    def detuning_scan(self, scan_range_factor=2, nD=100):
        """Gate state populations as a function of gate detuning.

        Refers to Figure 7.11 in Sepiol2016.

        Args:
            scan_range_factor: float, determines the range of the maximum gate detuning
                for simulation.
            nD: int, number of gate detunings to generate for simulation.

        Returns:
            (gate_detunings, population_ee, population_eg, population_ge, population_gg, fidelity)
            gate_detunings: list of floats, detuning of bichromatic light from the motional
                sidebands in radians/s
            population_ee: list of floats, the population in |ee> state.
            population_eg: list of floats, the population in |eg> state.
            population_ge: list of floats, the population in |ge> state.
            population_gg: list of floats, the population in |gg> state.
            fidelity: list of floats, the similarity between the target and the actual output state
                of the gate.
        """
        rho_in = _qt.tensor(_qt.fock_dm(self._num_ions, 0), _qt.fock_dm(self._num_ions, 0),
                            _qt.thermal_dm(self._phonons, self._nbar))
        rho_target = self._bell_state()*self._bell_state().dag()
        times = _np.linspace(0, self.gate_time, 2)

        maximum_detuning = scan_range_factor*self._delta_g
        gate_detunings = _np.linspace(-maximum_detuning, maximum_detuning, nD)

        population_ee = []
        population_eg = []
        population_ge = []
        population_gg = []
        fidelity = []

        for kk in gate_detunings:
            output = self._gate_H(kk, rho_in, times)
            rhos_after_gate = output.states
            population_ee.append(_qt.expect(self._ee*self._ee.dag(), rhos_after_gate[-1]))
            population_eg.append(_qt.expect(self._eg*self._eg.dag(), rhos_after_gate[-1]))
            population_ge.append(_qt.expect(self._ge*self._ge.dag(), rhos_after_gate[-1]))
            population_gg.append(_qt.expect(self._gg*self._gg.dag(), rhos_after_gate[-1]))
            fidelity.append(_qt.fidelity(rho_target, _qt.ptrace(rhos_after_gate[-1], [0, 1])))

        return gate_detunings, population_ee, population_eg, population_ge, population_gg, fidelity

    def _u_rot(self, theta, phi=0):
        """The unitary rotation operator.

        Please refer to the page 10 in website
        http://www.vcpc.univie.ac.at/~ian/hotlist/qc/talks/bloch-sphere-rotations.pdf

        Args:
            theta: float, the theta of spherical coordinates in Bloch sphere in radians.
            phi: float, the phi of spherical coordinates in Bloch sphere in radians.

        Returns:
            rot: qt.qobj, rotation operator of two ions.
        """
        rot_ions1 = ((_np.cos(theta/2) * _qt.qeye(self._num_ions)) - (1j * _np.sin(theta/2)
                     * (_np.cos(phi) * _qt.sigmax() + _np.sin(phi) * _qt.sigmay())))
        rot_ions2 = rot_ions1
        rot = _qt.tensor(rot_ions1, rot_ions2, _qt.qeye(self._phonons))
        return rot

    def parity_scan(self, nP=100):
        """Parity oscillation as function of phi of analysis pulse.

        Please refer to Eq. 4.51 in Sepiol2016

        Returns:
            (analysis_phases, parity)
            analysis_phases: list of floats, phases of analysis pulses in radians.
            parity: list of floats, population difference between (|ee> + |gg>) and
                (|eg> + |ge>) states.
        """

        rho_in = _qt.tensor(_qt.fock_dm(self._num_ions, 1), _qt.fock_dm(self._num_ions, 1),
                            _qt.thermal_dm(self._phonons, self._nbar))
        times = _np.linspace(0, self.gate_time, 2)
        output = self._gate_H(self._delta_g, rho_in, times)
        rho_after_gate = output.states[-1]

        analysis_phases = _np.linspace(0, 3*_np.pi, nP)

        dm = (self._ee*self._ee.dag() + self._gg*self._gg.dag() - self._eg*self._eg.dag()
              - self._ge*self._ge.dag())

        parity = []
        for phi in analysis_phases:
            phi_analysis = phi + self._phi_mean
            rho_after_analysis = (self._u_rot(theta=_np.pi/2, phi=phi_analysis) * rho_after_gate
                                  * self._u_rot(theta=_np.pi/2, phi=phi_analysis).dag())

            parity.append((_qt.expect(dm, rho_after_analysis)).real)

        return analysis_phases, parity

    def set_heating_rate(self, ndot):
        """Set motional heating rate."""
        self._n_dot = ndot

    def set_mot_coherence_time(self, tau):
        """Set motional coherence time."""
        self._tau_mot = tau

    def gate_detuning_for_loops(self, K):
        """Returns to gate detunings for different gate loops.

        Please refer to Eq.(4.39) in Sepiol 2016.

        Args:
            K: int, number of gate loops.
        """
        delta_g = 2 * self._eta * self._omega_0 * _np.sqrt(K)
        return delta_g

    def gate_time_for_loops(self, K):
        """Returns to gate time for different gate loops.

        Please refer to Eq.(4.38) in Sepiol 2016.

        Args:
            K: int, number of gate loops.
        """
        delta_g = self.gate_detuning_for_loops(K)
        gate_time = 2 * _np.pi * K/delta_g
        return gate_time

    def gate_error_by_heating_rate(self, max_n_dot=100, num_points=10, K=1):
        """Gate errors caused by motional heating rate.

        Args:
            max_n_dot: float, unit in quanta/s, the maximum value of motional heating rate.
            num_points: int, number of heating rate to generate for simulation
            K: int, number of gate loops

        Returns:
            (n_dot, errors)
            n_dot: list of floats, motional heating rate for simulation.
            errors: list of floats, gate errors caused by motional heating rate.
        """

        rho_in = _qt.tensor(_qt.fock_dm(self._num_ions, 0), _qt.fock_dm(self._num_ions, 0),
                            _qt.thermal_dm(self._phonons, self._nbar))
        rho_target = self._bell_state()*self._bell_state().dag()
        self._delta_g = self.gate_detuning_for_loops(K)
        gate_time = self.gate_time_for_loops(K)

        times = _np.linspace(0, gate_time, 2)
        n_dot = _np.linspace(0, max_n_dot, num_points)

        fidelity = []

        for kk in n_dot:
            self.set_heating_rate(kk)
            output = self._gate_H(self._delta_g, rho_in, times)
            rhos_after_gate = output.states
            fidelity.append(_qt.fidelity(rho_target, _qt.ptrace(rhos_after_gate[-1], [0, 1])))

        errors = [1 - kk for kk in fidelity]
        return n_dot, errors

    def gate_error_by_motional_decoherence(self, max_tau=400e-3, num_points=50, K=1):
        """Gate errors caused by motional decoherence.

        Args:
            max_tau: float, the maximum value of motional coherence time in seconds,
                max_tau <= 2/n_dot.
            num_points: int, number of heating rate to generate for simulation.
            K: int, number of gate loops.

        Returns:
            (taus, errors)
            taus: list of floats, motional decoherence time for simulation.
            errors: list of floats, gate errors caused by motional decoherence.
        """
        rho_in = _qt.tensor(_qt.fock_dm(self._num_ions, 0), _qt.fock_dm(self._num_ions, 0),
                            _qt.thermal_dm(self._phonons, self._nbar))
        rho_target = self._bell_state()*self._bell_state().dag()
        self._delta_g = self.gate_detuning_for_loops(K)
        gate_time = self.gate_time_for_loops(K)

        times = _np.linspace(0, gate_time, 2)
        taus = _np.linspace(0, max_tau, num_points)

        fidelity = []
        for tau in taus:
            self.set_mot_coherence_time(tau)
            output = self._gate_H(self._delta_g, rho_in, times)
            rhos_after_gate = output.states
            fidelity.append(_qt.fidelity(rho_target, _qt.ptrace(rhos_after_gate[-1], [0, 1])))

        errors = [1 - kk for kk in fidelity]
        return taus, errors
