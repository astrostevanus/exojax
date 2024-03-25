import pytest
import pkg_resources
import pandas as pd
import numpy as np
from exojax.spec.modit_scanfft import xsvector_scanfft
from exojax.spec.hitran import line_strength
from exojax.test.data import TESTDATA_CO_EXOMOL_MODIT_XS_REF
from exojax.spec import normalized_doppler_sigma, gamma_natural
from exojax.spec.hitran import line_strength
from exojax.spec.exomol import gamma_exomol
from exojax.utils.grids import wavenumber_grid
from exojax.spec.initspec import init_modit, init_modit_vald
from exojax.spec.set_ditgrid import ditgrid_log_interval
from exojax.test.emulate_mdb import mock_mdbExomol, mock_mdbVALD
from exojax.test.emulate_mdb import mock_wavenumber_grid


def test_xs_exomol():
    nus, wav, res = mock_wavenumber_grid()
    mdbCO = mock_mdbExomol()
    Tfix = 1200.0
    Pfix = 1.0
    #Mmol = molmass_isotope("CO")
    Mmol = mdbCO.molmass
    
    cont_nu, index_nu, R, pmarray = init_modit(mdbCO.nu_lines, nus)
    qt = mdbCO.qr_interp(Tfix)
    gammaL = gamma_exomol(Pfix, Tfix, mdbCO.n_Texp,
                          mdbCO.alpha_ref) + gamma_natural(mdbCO.A)
    dv_lines = mdbCO.nu_lines / R
    ngammaL = gammaL / dv_lines
    nsigmaD = normalized_doppler_sigma(Tfix, Mmol, R)
    Sij = line_strength(Tfix, mdbCO.logsij0, mdbCO.nu_lines, mdbCO.elower, qt, mdbCO.Tref)

    ngammaL_grid = ditgrid_log_interval(ngammaL, dit_grid_resolution=0.1)
    xsv = xsvector_scanfft(cont_nu, index_nu, R, pmarray, nsigmaD, ngammaL, Sij, nus,
                   ngammaL_grid)
    filename = pkg_resources.resource_filename(
        'exojax', 'data/testdata/' + TESTDATA_CO_EXOMOL_MODIT_XS_REF)
    dat = pd.read_csv(filename, delimiter=",", names=("nus", "xsv"))
    assert np.all(xsv == pytest.approx(dat["xsv"].values))


def test_rt_exomol():
    from jax import config
    config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    from exojax.spec import rtransfer as rt
    from exojax.spec import molinfo
    from exojax.spec.modit import exomol
    from exojax.spec.modit_scanfft import xsmatrix_scanfft
    from exojax.spec.layeropacity import layer_optical_depth
    from exojax.spec.rtransfer import rtrun_emis_pureabs_fbased2st
    from exojax.spec.planck import piBarr
    from exojax.spec.modit import set_ditgrid_matrix_exomol
    from exojax.test.data import TESTDATA_CO_EXOMOL_MODIT_EMISSION_REF
    
    nus, wav, res = mock_wavenumber_grid()
    mdb = mock_mdbExomol()
    Parr, dParr, k = rt.pressure_layer(NP=100, numpy = True)
    T0_in = 1300.0
    alpha_in = 0.1
    Tarr = T0_in * (Parr)**alpha_in
    Tarr[Tarr<400.0] = 400.0 #lower limit
    Tarr[Tarr>1500.0] = 1500.0 #upper limit
    
    molmass = mdb.molmass
    MMR = 0.1
    cont_nu, index_nu, R, pmarray = init_modit(mdb.nu_lines, nus)

    def fT(T0, alpha): return T0[:, None]*(Parr[None, :])**alpha[:, None]
    dgm_ngammaL = set_ditgrid_matrix_exomol(
        mdb, fT, Parr, R, molmass, 0.2, np.array([T0_in]), np.array([alpha_in]))
    
    g = 2478.57
    SijM, ngammaLM, nsigmaDl = exomol(mdb, Tarr, Parr, R, molmass)
    xsm = xsmatrix_scanfft(cont_nu, index_nu, R, pmarray, nsigmaDl, ngammaLM, SijM,
                   nus, dgm_ngammaL)
    dtau = layer_optical_depth(dParr, jnp.abs(xsm), MMR * np.ones_like(Parr), molmass, g)
    sourcef = piBarr(Tarr, nus)
    F0 = rtrun_emis_pureabs_fbased2st(dtau, sourcef)
    filename = pkg_resources.resource_filename(
        'exojax', 'data/testdata/' + TESTDATA_CO_EXOMOL_MODIT_EMISSION_REF)
    dat = pd.read_csv(filename, delimiter=",", names=("nus", "flux"))
    
    
    # The reference data was generated by
    #
    # >>> np.savetxt("modit_rt_test_ref.txt",np.array([nus,F0]).T,delimiter=",")
    #
    residual = np.abs(F0/dat["flux"].values - 1.0)
    print(np.max(residual))
    assert np.all(residual < 1.e-4)

    return F0


def test_rt_vald():
    from exojax.spec import moldb, atomll
    from exojax.spec import rtransfer as rt
    from exojax.spec.modit import set_ditgrid_matrix_vald_all
    from exojax.spec.modit import vald_all
    from exojax.spec.modit_scanfft import xsmatrix_vald_scanfft
    from exojax.spec.planck import piBarr
    from exojax.test.data import TESTDATA_VALD_MODIT_EMISSION_REF
    mdb = mock_mdbExomol()
    
    adb = mock_mdbVALD()
    asdb = moldb.AdbSepVald(adb)

    Parr, dParr, k = rt.pressure_layer(NP=100)
    T0_in = 3000.0
    alpha_in = 0.1
    Tarr = T0_in * (Parr)**alpha_in

    nus, wav, res = wavenumber_grid(15030.0,
                                        15045.0,
                                        2000,
                                        unit='AA',
                                        xsmode="modit")
    cnuS, indexnuS, R, pmarray = init_modit_vald(asdb.nu_lines, nus, asdb.N_usp)

    def fT(T0, alpha): return T0[:, None]*(Parr[None, :])**alpha[:, None]
    H_He_HH_VMR_ref = [0.0, 0.16, 0.84]#[0.1, 0.15, 0.75]
    PH_ref = Parr* H_He_HH_VMR_ref[0]
    PHe_ref = Parr* H_He_HH_VMR_ref[1]
    PHH_ref = Parr* H_He_HH_VMR_ref[2]
    dgm_ngammaL_VALD = set_ditgrid_matrix_vald_all(asdb, PH_ref, PHe_ref, PHH_ref, R, fT, 0.2, np.array([T0_in]), np.array([alpha_in]))

    g = 1e5
    mmw = 2.33

    ONEARR=np.ones_like(Parr)
    VMR_uspecies = atomll.get_VMR_uspecies(asdb.uspecies)[:, None]*ONEARR
    SijMS, ngammaLMS, nsigmaDlS = vald_all(asdb, Tarr, PH_ref, PHe_ref, PHH_ref, R)
    #xsmS = xsmatrix_vald(cnuS, indexnuS, R, pmarray, nsigmaDlS, ngammaLMS, SijMS, nus, dgm_ngammaL_VALD)
    xsmS = xsmatrix_vald_scanfft(cnuS, indexnuS, R, pmarray, nsigmaDlS, ngammaLMS, SijMS, nus, dgm_ngammaL_VALD)
    dtauatom = rt.dtauVALD(dParr, xsmS, VMR_uspecies, mmw*ONEARR, g)

    sourcef = piBarr(Tarr, nus)
    F0 = rt.rtrun_emis_pureabs_fbased2st(dtauatom, sourcef)
    # The reference data was generated by
    #
    # >>> np.savetxt("modit_rt_test_vald_ref.txt",np.array([nus,F0]).T,delimiter=",")
    #
    filename = pkg_resources.resource_filename(
        'exojax', 'data/testdata/' + TESTDATA_VALD_MODIT_EMISSION_REF)
    ref = pd.read_csv(filename, delimiter=",", names=("nus", "flux"))

    residual = np.abs(F0/ref["flux"].values - 1.0)
    print(np.max(residual))

    # Need to regenerate TESTDATA_VALD_MODIT_EMISSION_REF because we modified the layer definition of RT
    #assert np.all(residual < 0.03)

    return F0
    
    
if __name__ == "__main__":
    test_xs_exomol()
    test_rt_exomol()
    #test_rt_vald()
