def zero_replaced_nuf(nuf):
    """calc zero-replaced formula matrix
    Args:
        nuf: formula matrix
        
    Returns:
        zero-replaced formula matrix (float32) 
    """
    znuf=np.copy(nuf)
    znuf[znuf==0]=np.nan
    return znuf

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
    de=np.abs(np.array(epsiloni_[:,np.newaxis]-epsilonj_[np.newaxis,:]))
    mse=de>1.e-18 #should be refactored
    return np.array(mse)


def species_index_same_epsilonj(epsiloni,epsilonj,nuf):
    """species index of i for epsilon_i = epsilon_j for given element index j
    
    Args:
        epsiloni: elements abundance
        epsilonj: species abundance
        
    Returns:
        i(j) for epsilon_i = epsilon_j
    """
    mm=mask_diff_epsilon(epsiloni,epsilonj)
    si=np.arange(0,len(epsiloni))
    isamej=[]
    nufsamej=[]
    for j in range(0,len(epsilonj)):
        isamej.append(si[~mm[:,j]])
        nufsamej.append(np.array(nuf[:,j][~mm[:,j]],dtype=int))
    return isamej, nufsamej

def calc_Amatrix_np(nuf,xj,Aj0):
    """calc A matrix in Stock et al. (2018) (2.28, 2.29) numpy version
    
    Args:
        nuf: formula matrix
        xj: elements activity
        Aj0: Aj0 component defined by (2.27)
        
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
        Ki=K_[isamej[j]]
        #print("i_same_j",i,"K_i",Ki,"k=nu_ij",klist)
        lprod_i=np.prod(np.delete(xnuf,j,axis=1),axis=1) # Prod n_l^nu_{ij}(2.29) for all i
        kprodi=Ki*lprod_i[i]
        for ik,k in enumerate(klist):
            Ap[j,k]=Ap[j,k]+k*kprodi[ik]
    return Ap
