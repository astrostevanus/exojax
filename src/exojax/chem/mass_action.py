from exojax.utils.constants import kB

def logK_FC(T,nu,ma_coeff):
    """mass action constant of FastChem form

    Args:
       T: temperature
       nu: form matrix
       ma_coeff: mass action coefficient of FastChem form

    Returns:
       mass action
    
    """
    sigma = 1 - np.sum(nu_,axis=1)
    A,B,C,D,E=ma_coeff
    logK0=(A/T+B*np.log(T) + C + D*T + E*T**2)
    return logK0 - sigma*np.log(1.e-6*kB*T)
