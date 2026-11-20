"""A decision model of your own: full ECCD on the largest island's surface while it is wider than on_cm.
The harness calls make(P_max=..., physics=..., **args) once, then the function it returns after every TORAX step with the
observation (t, w: island widths in m, largest first, rho: their surfaces in rho_norm, dwdt, P, aim, j_bs). It returns the
gyrotron power in MW and where to aim it, in rho_norm. Used by examples/my_decision_session.json."""


def make(P_max, physics, on_cm=2.0):
    def decide(obs):
        w, rho = obs["w"][0], obs["rho"][0]
        aim = rho if rho == rho else 0.5             # no island found: park the mirror mid-radius
        return (P_max if w * 100 > on_cm else 0.0), aim
    return decide
