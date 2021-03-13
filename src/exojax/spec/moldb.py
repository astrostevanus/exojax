"""Molecular database (MDB) class

   * MdbHit is the MDB for HITRAN or HITEMP  
   
"""
import numpy as np
import jax.numpy as jnp
import pathlib
from exojax.spec import hapi, exomolapi, exomol
from exojax.spec.hitran import gamma_natural as gn
import pandas as pd

__all__ = ['MdbExomol','MdbHit']

class MdbExomol(object):
    def __init__(self,path,nurange=[-np.inf,np.inf],margin=250.0,crit=-np.inf):
        """Molecular database for Exomol form

        Args: 
           path: path for Exomol data directory/tag. For instance, "/home/CO/12C-16O/Li2015/12C-16O__Li2015"
           nurange: wavenumber range list (cm-1)
           margin: margin for nurange (cm-1)
           crit: line strength lower limit for extraction

        """
        self.path = pathlib.Path(path)
        molec=str(self.path.stem)
        self.trans_file = self.path.parent/pathlib.Path(molec+".trans.bz2")
        self.states_file = self.path.parent/pathlib.Path(molec+".states.bz2")
        self.pf_file = self.path.parent/pathlib.Path(molec+".pf")
        self.def_file = self.path.parent/pathlib.Path(molec+".def")
        self.crit = crit
        self.margin = margin
        self.nurange=[np.min(nurange),np.max(nurange)]

        #downloading
        if not self.trans_file.exists():
            self.download()

        #loading exomol files
        trans=exomolapi.read_trans(self.trans_file)
        states=exomolapi.read_states(self.states_file)
        pf=exomolapi.read_pf(self.pf_file)
        self.gQT=jnp.array(pf["QT"].to_numpy()) #grid QT
        self.T_gQT=jnp.array(pf["T"].to_numpy()) #T forgrid QT
        self.n_Texp, self.alpha_ref, self.molmass=exomolapi.read_def(self.def_file)

        #compute gup and elower
        A, self.nu_lines, elower, gpp=exomolapi.pickup_gE(states,trans)        
        self.Tref=296.0        
        self.QTref=np.array(self.QT_interp(self.Tref))
        ##input should be ndarray not jnp array
        self.Sij0=exomol.Sij0(A,gpp,self.nu_lines,elower,self.QTref)
        
        ### MASKING ###
        mask=(self.nu_lines>self.nurange[0]-self.margin)\
        *(self.nu_lines<self.nurange[1]+self.margin)\
        *(self.Sij0>self.crit)

        #numpy float 64 Do not convert them jnp array
        self.nu_lines = self.nu_lines[mask]
        self.Sij0 = self.Sij0[mask]        


        #jnp arrays
        self.A=jnp.array(A[mask])
        self.gamma_natural=gn(self.A)
        self.elower=jnp.array(elower[mask])
        self.gpp=jnp.array(gpp[mask])
        self.logsij0=jnp.array(np.log(self.Sij0))
        self.dev_nu_lines=jnp.array(self.nu_lines)
        
    def QT_interp(self,T):
        """interpolated partition function
        Args:
           T: temperature
        Returns:
           Q(T) interpolated in jnp.array
        """
        return jnp.interp(T,self.T_gQT,self.gQT)
    
    def qr_interp(self,T):
        """interpolated partition function ratio
        Args:
           T: temperature
        Returns:
           qr(T)=Q(T)/Q(Tref) interpolated in jnp.array
        """
        return self.QT_interp(T)/self.QT_interp(self.Tref)
    
        
    def download(self):
        """Downloading Exomol files

        Note:
           The download URL is written in exojax.utils.url.

        """
        import urllib.request
        from exojax.utils.url import url_ExoMol
        print("Downloading parfile from "+url)
        url = url_ExoMol()+self.path.name
#        try:
#            urllib.request.urlretrieve(url,str(self.path))
#        except:
#            print("Couldn't download and save.")




            

class MdbHit(object):
    def __init__(self,path,nurange=[-np.inf,np.inf],margin=250.0,crit=-np.inf):
        """Molecular database for HITRAN/HITEMP form

        Args: 
           path: path for HITRAN/HITEMP par file
           nurange: wavenumber range list (cm-1)
           margin: margin for nurange (cm-1)
           crit: line strength lower limit for extraction

        """        
        self.path = pathlib.Path(path)
        molec=str(self.path.stem)
        #downloading
        if not self.path.exists():
            self.download()
        hapi.db_begin(str(self.path.parent))            
        self.Tref=296.0        
        self.molecid = search_molecid(molec)
        self.crit = crit
        self.margin = margin
        self.nurange=[np.min(nurange),np.max(nurange)]
        self.nu_lines = hapi.getColumn(molec, 'nu')
        self.Sij0 = hapi.getColumn(molec, 'sw')

        ### MASKING ###
        mask=(self.nu_lines>self.nurange[0]-self.margin)\
        *(self.nu_lines<self.nurange[1]+self.margin)\
        *(self.Sij0>self.crit)

        #numpy float 64 Do not convert them jnp array
        self.nu_lines = self.nu_lines[mask]
        self.Sij0 = self.Sij0[mask]        
        self.delta_air = hapi.getColumn(molec, 'delta_air')[mask]

        #jnp array
        A=hapi.getColumn(molec, 'a')[mask]
        self.A = jnp.array(A)
        self.gamma_natural=gn(A)

        self.n_air = jnp.array(hapi.getColumn(molec, 'n_air')[mask])
        self.gamma_air = jnp.array(hapi.getColumn(molec, 'gamma_air')[mask])
        self.gamma_self = jnp.array(hapi.getColumn(molec, 'gamma_self')[mask])        
        self.elower = jnp.array(hapi.getColumn(molec, 'elower')[mask])
        self.gpp = jnp.array(hapi.getColumn(molec, 'gpp')[mask]) 
        self.logsij0=jnp.array(np.log(self.Sij0)) 
        self.dev_nu_lines=jnp.array(self.nu_lines)

        #int
        self.isoid = hapi.getColumn(molec,'local_iso_id')[mask]
        self.uniqiso=np.unique(self.isoid)

        

        
    def download(self):
        """Downloading HITRAN par file

        Note:
           The download URL is written in exojax.utils.url.

        """
        import urllib.request
        from exojax.utils.url import url_HITRAN12
        print("Downloading parfile from "+url)
        url = url_HITRAN12()+self.path.name
        try:
            urllib.request.urlretrieve(url,str(self.path))
        except:
            print("Couldn't download and save.")
            
    def Qr(self,Tarr):
        """Partition Function ratio using HAPI partition sum

        Args:
           Tarr: temperature array (K)
        
        Returns:
           Qr = partition function ratio array [N_Tarr x N_iso]

        Note: N_Tarr = len(Tarr), N_iso = len(self.uniqiso)

        """
        allT=list(np.concatenate([[self.Tref],Tarr]))
        Qrx=[]
        for iso in self.uniqiso:
            Qrx.append(hapi.partitionSum(self.molecid,iso, allT))
        Qrx=np.array(Qrx)
        qr=Qrx[:,1:].T/Qrx[:,0] #Q(T)/Q(Tref)
        return qr

    def Qr_line(self,T):
        """Partition Function ratio using HAPI partition sum

        Args:
           T: temperature (K)

        Returns:
           Qr_line, partition function ratio array for lines [Nlines]

        Note: Nlines=len(self.nu_lines)

        """
        qr_line=np.ones_like(self.isoid,dtype=np.float64)
        qrx=self.Qr([T])
        for idx,iso in enumerate(self.uniqiso):
            mask=self.isoid==iso
            qr_line[mask]=qrx[0,idx]
        return qr_line

    def Qr_layer(self,Tarr):
        """Partition Function ratio using HAPI partition sum

        Args:
           Tarr: temperature array (K)

        Returns:
           Qr_layer, partition function ratio array for lines [N_Tarr x Nlines]

        Note: 
           Nlines=len(self.nu_lines)
           N_Tarr=len(Tarr)

        """
        NP=len(Tarr)
        qt=np.zeros((NP,len(self.isoid)))
        qr=self.Qr(Tarr)
        for idx,iso in enumerate(self.uniqiso):
            mask=self.isoid==iso
            for ilayer in range(NP):
                qt[ilayer,mask]=qr[ilayer,idx]
        return qt
    
def search_molecid(molec):
    """molec id from molec (source table name) of HITRAN/HITEMP

    Args:
       molec: source table name

    Return:
       int: molecid (HITRAN molecular id)

    """
    try:
        hitf=molec.split("_")
        molecid=int(hitf[0])
        return molecid

    except:
        print("Warning: Define molecid by yourself.")
        return None

if __name__ == "__main__":
    mdbCO=MdbExomol("/home/kawahara/exojax/data/exomol/CO/12C-16O/Li2015/12C-16O__Li2015")
    print(mdbCO.Sij0)

    print(mdbCO.logsij0)
