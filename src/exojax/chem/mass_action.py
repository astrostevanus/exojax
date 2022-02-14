from exojax.utils.constants import kB
import jax.numpy as jnp

def logK_FC(T,nuf,ma_coeff):
    """mass action constant of FastChem form

    Args:
       T: temperature
       nuf: formula matrix
       ma_coeff: mass action coefficient of FastChem form

    Returns:
       mass action
    
    """
    sigma = 1.0 - jnp.sum(nuf,axis=1)
    A,B,C,D,E=ma_coeff
    logK0=(A/T+B*jnp.log(T) + C + D*T + E*T**2)
    return logK0 - sigma*jnp.log(1.e-6*kB*T)
