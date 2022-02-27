import numpy as np
import jax.numpy as jnp
from jax.lax import scan

def calc_nufmask(nuf):
    """calc zero-replaced to nan formula matrix mask
    Args:
        nuf: formula matrix
        
    Returns:
        nufmask (float32) 
    """
    nufmask=np.copy(nuf)
    msk=nufmask==0
    nufmask[~msk]=1.0
    nufmask[msk]=np.nan
    return nufmask


def calc_epsiloni(nufmask,epsilonj):
    """calc species abundaunce=epsilon_i (2.24) in Stock et al.(2018)
    
    Args:
        nufmask: formula matrix mask
        epsilonj: element abundance (epsilon_j)
        
    Returns:
        species abundaunce= epsilon_i
    
    """
    emat=(np.full_like(nufmask,1)*epsilonj)
    return np.nanmin(emat*nufmask,axis=1)

def calc_Nj(nuf,epsiloni,epsilonj):
    """calc Nj defined by (2.25) in Stock et al. (2018)
    
    Args:
        nuf: formula matrix
        epsiloni: elements abundance
        epsilonj: species abundance
        
    Returns:
        Nj (ndarray)
        Njmax 
    """
    mse=mask_diff_epsilon(epsiloni,epsilonj)
    masked_nuf=np.copy(nuf)
    masked_nuf[mse]=0.0
    Nj=np.array(np.max(masked_nuf,axis=0),dtype=int)
    return Nj, np.max(Nj)

def mask_diff_epsilon(epsiloni,epsilonj):
    """epsilon_i = epsilon_j
    
    Args:
        epsiloni: elements abundance
        epsilonj: species abundance
        
    Returns:
        mask for epsilon_i > epsilon_j
    """
    de=np.abs(np.array(epsiloni[:,np.newaxis]-epsilonj[np.newaxis,:]))
    mse=de>1.e-18 #should be refactored
    return np.array(mse)


def species_index_same_epsilonj(epsiloni,epsilonj,nuf):
    """species index of i for epsilon_i = epsilon_j for given element index j
    
    Note:
        isamej is the species index i(j) for epsilon_i = epsilon_j. nufsamej is the formula matrix component nuf(j) for epsilon_i = epsilon_j.
        
    Args:
        epsiloni: elements abundance
        epsilonj: species abundance
        
    Returns:
        isamej (the species index i(j) for epsilon_i = epsilon_j), nufsamej (the formula matrix component nuf(j) for epsilon_i = epsilon_j) 
    """
    mm=mask_diff_epsilon(epsiloni,epsilonj)
    si=np.arange(0,len(epsiloni))
    isamej=[]
    nufsamej=[]
    for j in range(0,len(epsilonj)):
        isamej.append(si[~mm[:,j]])
        nufsamej.append(np.array(nuf[:,j][~mm[:,j]],dtype=int))
    return isamej, nufsamej

def calc_Amatrix_np(nuf,xj,Keq,Aj0,isamej,nufsamej,Njmax):
    """calc A matrix in Stock et al. (2018) (2.28, 2.29) numpy version
    
    Args:
        nuf: formula matrix
        xj: elements activity
        Keq: equilibrium constant 
        Aj0: Aj0 component defined by (2.27)
        isamej: isamej (the species index i(j) for epsilon_i = epsilon_j)
        nufsamej: nufsamej (the formula matrix component nuf(j) for epsilon_i = epsilon_j) 
        Njmax: Njmax
        
    Returns:
        A matrix
    """
    numi,numj=np.shape(nuf)
    Ap=np.zeros((numj,Njmax+1))
    Ap[:,0]=Aj0
    Ap[:,1]=1.0
    xnuf=xj**nuf # 
    for j in range(0,numj):
        i=isamej[j]
        klist=nufsamej[j]
        Ki=Keq[isamej[j]]
        #print("i_same_j",i,"K_i",Ki,"k=nu_ij",klist)
        lprod_i=np.prod(np.delete(xnuf,j,axis=1),axis=1) # Prod n_l^nu_{ij}(2.29) for all i
        kprodi=Ki*lprod_i[i]
        for ik,k in enumerate(klist):
            Ap[j,k]=Ap[j,k]+k*kprodi[ik]
    return Ap


def set_samej_formatted(isamej,nufsamej,Nj,numi):
    """make formatted isamej and nufsamej
    
    Note:
       species_index_same_epsilonj generates the inputs of this function.
    
    Args:
       isamej: isamej (the species index i(j) for epsilon_i = epsilon_j)
       nufsamej: nufsamej (the formula matrix component nuf(j) for epsilon_i = epsilon_j) 
       Nj: Nj computed by calc_Nj
       numi: number of the species
        
    Returns:
       isamej_formatted, nufsamej_formatted
       
    
    """
    isamej_formatted=np.zeros((len(isamej),numi))
    nufsamej_formatted=np.zeros((len(isamej),numi))
    for j in range(0,len(isamej)):
        isamej_formatted[j,0:Nj[j]]=isamej[j]
        nufsamej_formatted[j,0:Nj[j]]=nufsamej[j]
    return isamej_formatted, nufsamej_formatted

def calc_Amatrix(nuf,xj,Keq,Aj0,isamej_formatted, nufsamej_formatted,Njmax):
    """calc A matrix in Stock et al. (2018) (2.28, 2.29) jax version
    
    Note:
        isamej_formatted and nufsamej_formatted can be computed using set_samej_formatted
    
    Args:
        nuf: formula matrix
        xj: elements activity
        Keq: equilibrium constant 
        Aj0: Aj0 component defined by (2.27)
        isamej_formatted: formatted isamej (the species index i(j) for epsilon_i = epsilon_j)
        nufsamej_formatted: formatted nufsamej (the formula matrix component nuf(j) for epsilon_i = epsilon_j) 
        Njmax: Njmax
        
    Returns:
        A matrix
    """
    xnuf=xj**nuf
    lprod_ij=jnp.prod(xnuf,axis=1)
    lprod_ij=lprod_ij[:,jnp.newaxis]/xnuf
    Klprod_ij=Keq[:,jnp.newaxis]*lprod_ij
    
    numi,numj=jnp.shape(nuf)
    xs=jnp.hstack([nufsamej_formatted,Klprod_ij.T,isamej_formatted])
    def f(Apj,x):
        j=Apj[0]
        #j=j+1
        Ap=Apj[1]
        klist=x[:numi]
        Klprod_each=x[numi:2*numi]
        isamej_each=x[2*numi:]
    
        def g(Ap,x):
            k=x[0]#[:numi]
            isamej=x[1]#[numi:]
            Ap=Ap.at[j.astype(int),k.astype(int)].add(k*Klprod_each[isamej.astype(int)])
            return Ap, 0
    
        xt=jnp.vstack([klist,isamej_each]).T
        Ap,_=scan(g,Ap,xt)
        Apj=[j+1,Ap]
        return Apj, 0 

    #Apj initialization
    Ap=jnp.zeros((numj,Njmax+1))
    Ap=Ap.at[:,1].set(1.0)
    Ap=Ap.at[:,0].set(Aj0)
    Apj=[0,Ap]

    Apj,_=scan(f,Apj,xs)
    j,Ap=Apj
    return Ap

