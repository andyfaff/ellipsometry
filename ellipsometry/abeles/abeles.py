"""
Parts modified from the tmm project:

Copyright (C) 2012-2017 Steven Byrnes

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import numpy as np

EPSILON = np.finfo(float).eps


def is_forward_angle(n, theta):
    """
    if a wave is traveling at angle theta from normal in a medium with index n,
    calculate whether or not this is the forward-traveling wave (i.e., the one
    going from front to back of the stack, like the incoming or outgoing waves,
    but unlike the reflected wave). For real n & theta, the criterion is simply
    -pi/2 < theta < pi/2, but for complex n & theta, it's more complicated.
    See https://arxiv.org/abs/1603.02720 appendix D. If theta is the forward
    angle, then (pi-theta) is the backward angle and vice-versa.
    """
    re_im = np.real(n) * np.imag(n)
    if np.any(re_im < 0):
        raise ValueError("For materials with gain, it's ambiguous which "
                         "beam is incoming vs outgoing. See "
                         "https://arxiv.org/abs/1603.02720 Appendix C.")

    ncostheta = np.asarray(n * np.cos(theta))

    is_forward = np.ones_like(ncostheta, dtype=bool)

    imag_gt0 = np.abs(ncostheta.imag) > 100 * EPSILON

    # Either evanescent decay or lossy medium. Either way, the one that
    # decays is the forward-moving wave
    is_forward[imag_gt0] = np.imag(ncostheta) > 0

    # Forward is the one with positive Poynting vector
    # Poynting vector is Re[n cos(theta)] for s-polarization or
    # Re[n cos(theta*)] for p-polarization, but it turns out they're consistent
    # so I'll just assume s then check both below
    is_forward[~imag_gt0] = np.real(ncostheta) > 0

    # double-check the answer ... can't be too careful!
    error_string = ("It's not clear which beam is incoming vs outgoing. Weird"
                    " index maybe?\n"
                    "n: " + str(n) + "   angle: " + str(theta))

    # if answer is True:
    #     assert ncostheta.imag > -100 * EPSILON, error_string
    #     assert ncostheta.real > -100 * EPSILON, error_string
    #     assert (n * np.cos(theta.conjugate())).real > -100 * EPSILON, error_string
    # else:
    #     assert ncostheta.imag < 100 * EPSILON, error_string
    #     assert ncostheta.real < 100 * EPSILON, error_string
    #     assert (n * np.cos(theta.conjugate())).real < 100 * EPSILON, error_string

    return is_forward


def snell(n0, n1, theta):
    """
    return list of angle theta in each layer based on angle theta in layer 0,
    using Snell's law. n_list is index of refraction of each layer. Note that
    "angles" may be complex!!
    """
    angles = np.arcsin(n0 * np.sin(theta) / n1)

    # Correct the forward angle
    forward = is_forward_angle(n0, angles)
    angles[~forward] = np.pi - angles

    forward = is_forward_angle(n1, angles)
    angles[~forward] = np.pi - angles

    return angles


def interface_r(polarisation, n_i, n_f, th_i, th_f):
    """
    reflection amplitude (from Fresnel equations)
    polarization is either "s" or "p" for polarization
    n_i, n_f are (complex) refractive index for incident and final
    th_i, th_f are (complex) propegation angle for incident and final
    (in radians, where 0=normal). "th" stands for "theta".
    """
    if polarisation == 's':
        return ((n_i * np.cos(th_i) - n_f * np.cos(th_f)) /
                (n_i * np.cos(th_i) + n_f * np.cos(th_f)))
    elif polarisation == 'p':
        return ((n_f * np.cos(th_i) - n_i * np.cos(th_f)) /
                (n_f * np.cos(th_i) + n_i * np.cos(th_f)))
    else:
        raise ValueError("Polarisation must be 's' or 'p'")


def interface_t(polarisation, n_i, n_f, th_i, th_f):
    """
    transmission amplitude (frem Fresnel equations)
    polarization is either "s" or "p" for polarization
    n_i, n_f are (complex) refractive index for incident and final
    th_i, th_f are (complex) propegation angle for incident and final
    (in radians, where 0=normal). "th" stands for "theta".
    """
    if polarisation == 's':
        return (2 * n_i * np.cos(th_i) /
                (n_i * np.cos(th_i) + n_f * np.cos(th_f)))
    elif polarisation == 'p':
        return (2 * n_i * np.cos(th_i) /
                (n_f * np.cos(th_i) + n_i * np.cos(th_f)))
    else:
        raise ValueError("Polarisation must be 's' or 'p'")


def tmm(pol, n, d, theta, lam_vac):
    """
    Main "coherent transfer matrix method" calc. Given parameters of a stack,
    calculates everything you could ever want to know about how light
    propagates in it. (If performance is an issue, you can delete some of the
    calculations without affecting the rest.)

    Parameters
    ----------
    pol: {'s', 'p'}
        light polarization, "s" or "p".
    n: array-like
        refractive indices, in the order that the light would pass through
        The first and last elements are the semi-infinite fronting and backing
        media.
    d: array-like
        layer thicknesses (front to back). Should correspond one-to-one with
         elements of n_list. First and last elements are ignored.
    theta: float
        The angle of incidence with respect to the normal. For a dissipative
        incoming medium (n[0] is not real), `theta` should be complex so that
        ``n[0] * sin(theta)`` is real (intensity is constant as a function of
        lateral position).
    lam_vac: float
        vacuum wavelength of the light.

    Outputs the following as a dictionary (see manual for details)
    * r--reflection amplitude
    * t--transmission amplitude
    * R--reflected wave power (as fraction of incident)
    * T--transmitted wave power (as fraction of incident)
    * power_entering--Power entering the first layer, usually (but not always)
      equal to 1-R (see manual).
    * vw_list-- n'th element is [v_n,w_n], the forward- and backward-traveling
      amplitudes, respectively, in the n'th medium just after interface with
      (n-1)st medium.
    * kz_list--normal component of complex angular wavenumber for
      forward-traveling wave in each layer.
    * th_list--(complex) propagation angle (in radians) in each layer
    * pol, n_list, d_list, th_0, lam_vac--same as input
    """

    # Convert lists to numpy arrays if they're not already.
    n = np.asfarray(n)
    d = np.asfarray(d, dtype=float)

    # Input tests
    if abs((n[0] * np.sin(theta)).imag) < 100 * EPSILON:
        raise ValueError('Error in n[0] or theta!')
    if not is_forward_angle(n[0], theta).all():
        raise ValueError('Error in n[0] or theta!')

    nlayers = n.size

    # th_list is a list with, for each layer, the angle that the light travels
    # through the layer. Computed with Snell's law. Note that the "angles" may be
    # complex!
    thetas = snell(n[0], n, theta)

    # kz is the z-component of (complex) angular wavevector for forward-moving
    # wave. Positive imaginary part means decaying.
    kz = 2 * np.pi * n * np.cos(thetas) / lam_vac

    # delta is the total phase accrued by traveling through a given layer.
    delta = kz * d

    # For a very opaque layer, reset delta to avoid divide-by-0 and similar
    # errors. The criterion imag(delta) > 35 corresponds to single-pass
    # transmission < 1e-30 --- small enough that the exact value doesn't
    # matter.
    delta[np.imag(delta) > 35.] = np.real(delta) + 35j

    # t_list[i,j] and r_list[i,j] are transmission and reflection amplitudes,
    # respectively, coming from i, going to j. Only need to calculate this when
    # j = i + 1. (2D array is overkill but helps avoid confusion.)
    t_list = np.zeros((nlayers, nlayers), dtype=complex)
    r_list = np.zeros((nlayers, nlayers), dtype=complex)

    for i in range(nlayers-1):
        t_list[i, i+1] = interface_t(pol, n[i], n[i + 1],
                                    thetas[i], thetas[i+1])
        r_list[i, i+1] = interface_r(pol, n[i], n[i + 1],
                                    thetas[i], thetas[i+1])

    # At the interface between the (n-1)st and nth material, let v_n be the
    # amplitude of the wave on the nth side heading forwards (away from the
    # boundary), and let w_n be the amplitude on the nth side heading backwards
    # (towards the boundary). Then (v_n,w_n) = M_n (v_{n+1},w_{n+1}). M_n is
    # M_list[n]. M_0 and M_{num_layers-1} are not defined.
    # My M is a bit different than Sernelius's, but Mtilde is the same.
    M_list = np.zeros((nlayers, 2, 2), dtype=complex)

    for i in range(1, nlayers-1):
        M_list[i] = (1 / t_list[i, i+1]) * np.dot(
            make_2x2_array(np.exp(-1j*delta[i]), 0, 0, np.exp(1j*delta[i]),
                           dtype=complex),
            make_2x2_array(1, r_list[i,i+1], r_list[i,i+1], 1, dtype=complex))

    Mtilde = np.eye(2, dtype=np.complex128)

    for i in range(1, nlayers-1):
        Mtilde = np.dot(Mtilde, M_list[i])

    Mtilde = np.dot(make_2x2_array(1, r_list[0,1], r_list[0,1], 1,
                                   dtype=complex)/t_list[0,1], Mtilde)

    # Net complex transmission and reflection amplitudes
    r = Mtilde[1, 0] / Mtilde[0, 0]
    t = 1 / Mtilde[0, 0]

    # Net transmitted and reflected power, as a proportion of the incoming light
    # power.
    R = R_from_r(r)
    T = T_from_t(pol, t, n[0], n[-1], theta, thetas[-1])
    power_entering = power_entering_from_r(pol, r, n[0], theta)

    return {'r': r, 't': t, 'R': R, 'T': T, 'power_entering': power_entering,
            'kz': kz, 'thetas': thetas,
            'pol': pol, 'n': n, 'd': d, 'theta': theta,
            'lam_vac':lam_vac}
