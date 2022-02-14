"""test for loading atomic data."""

import pytest
import numpy as np
from exojax.chem.mass_action import logK_FC


def test_FastChem():
    T=1500.0
    #H2 Hydrogen : H 2 # Chase, M. et al., JANAF thermochemical tables, 1998.
    kpH2=[5.1909637142380554e+04,-1.8011701211306956e+00,8.7224583233705744e-02,2.5613890164973008e-04,-5.3540255367406060e-09]
    #C1O1 Carbon_Monoxide : C 1 O 1 # Chase, M. et al., JANAF thermochemical tables, 1998.
    kpCO=[1.2899777785630804e+05,-1.7549835812545211e+00,-3.1625806804795502e+00,4.1336204683783961e-04 ,-2.3579962985989574e-08] 
    #C1H4 Methane : C 1 H 4 # Chase, M. et al., JANAF thermochemical tables, 1998.
    kpCH4=[1.9784584536781305e+05,-8.8316803072239054e+00,5.2793066855988400e+00,2.7567674752936866e-03,-1.3966691995535711e-07]
    #H2O1 Water : H 2 O 1 # Chase, M. et al., JANAF thermochemical tables, 1998.
    kpH2O=[1.1033645388793820e+05,-4.1783597409582285e+00,3.1744691010633233e+00,9.4064684023068001e-04,-4.0482461482866891e-08]
    ma_coeff=np.array([kpH2,kpCO,kpCH4,kpH2O]).T
    nu=np.array([[2,0,0],[0,1,1],[4,1,0],[2,0,1]]) #(2.1) formula matrix
    logK=logK_FC(T,nu,ma_coeff)
    refs=np.array([-21.12764,27.547245,-95.67498,-38.54748 ])
    residuals=(np.sum(logK-refs)**2)
    assert residuals < 1.e-16



if __name__ == '__main__':
    test_FastChem()
