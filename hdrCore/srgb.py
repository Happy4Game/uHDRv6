# -*- coding: utf-8 -*-
"""
HDR Core sRGB Transfer Functions Module

This module provides sRGB electro-optical transfer function (EOTF/EOCF) implementations
for HDR image processing workflows. It includes both forward and inverse sRGB transfer
functions that are essential for proper color space conversions in HDR imaging.

These functions are based on the IEC 61966-2-1:1999 standard and ITU-R BT.709-6
recommendations, ensuring accurate color reproduction and display compatibility.

Functions:
    - eotf_inverse_sRGB: Convert linear luminance to gamma-corrected sRGB values
    - eotf_sRGB: Convert gamma-corrected sRGB values to linear luminance

Note:
    This module is derived from the Colour Science library and adapted for use
    within the uHDR processing pipeline. It maintains compatibility with the
    original Colour library interface while providing optimized implementations
    for HDR imaging workflows.

References:
    - IEC 61966-2-1:1999 - sRGB colour space standard
    - ITU-R BT.709-6 - HDTV parameter values
    - Colour Science library (https://colour-science.org)
"""

import numpy as np

from colour.algebra import spow
from colour.utilities import (as_float, domain_range_scale, from_range_1,
                              to_domain_1)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['eotf_inverse_sRGB', 'eotf_sRGB']


def eotf_inverse_sRGB(L):
    """
    Defines the *IEC 61966-2-1:1999* *sRGB* inverse electro-optical transfer
    function (EOTF / EOCF).

    Parameters
    ----------
    L : numeric or array_like
        *Luminance* :math:`L` of the image.

    Returns
    -------
    numeric or ndarray
        Corresponding electrical signal :math:`V`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`InternationalElectrotechnicalCommission1999a`,
    :cite:`InternationalTelecommunicationUnion2015i`

    Examples
    --------
    >>> eotf_inverse_sRGB(0.18)  # doctest: +ELLIPSIS
    0.4613561...
    """

    L = to_domain_1(L)

    V = np.where(L <= 0.0031308, L * 12.92, 1.055 * spow(L, 1 / 2.4) - 0.055)

    return as_float(from_range_1(V))


def eotf_sRGB(V):
    """
    Defines the *IEC 61966-2-1:1999* *sRGB* electro-optical transfer function
    (EOTF / EOCF).

    Parameters
    ----------
    V : numeric or array_like
        Electrical signal :math:`V`.

    Returns
    -------
    numeric or ndarray
        Corresponding *luminance* :math:`L` of the image.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`InternationalElectrotechnicalCommission1999a`,
    :cite:`InternationalTelecommunicationUnion2015i`

    Examples
    --------
    >>> eotf_sRGB(0.461356129500442)  # doctest: +ELLIPSIS
    0.1...
    """

    V = to_domain_1(V)

    with domain_range_scale('ignore'):
        L = np.where(
            V <= eotf_inverse_sRGB(0.0031308),
            V / 12.92,
            spow((V + 0.055) / 1.055, 2.4),
        )

    return as_float(from_range_1(L))
