def test_imports():
    """Test we can import the necessary packages"""
    import os
    import pkg_resources
    import jwst_pancake as pancake
    print("PanCAKE Has been imported successfully!")
    print("You have PanCAKE version {} isntalled".format(pancake.__version__))
    import pandeia.engine
    print("Pandeia is version {}".format(pkg_resources.get_distribution("pandeia.engine").version))
    pandeia_version_file = os.path.join(os.environ["pandeia_refdata"], "VERSION_PSF")
    with open(pandeia_version_file, 'r') as inf:
        pandeia_data_version = inf.readline().strip()
    print("You have version {} of the pandeia data".format(pandeia_data_version))
    import webbpsf
    print("Webbpsf is version {}".format(webbpsf.__version__))
    webbpsf_version_file = os.path.join(os.environ["WEBBPSF_PATH"], "version.txt")
    with open(webbpsf_version_file, 'r') as inf:
        webbpsf_data_version = inf.readline().strip()
    print("You have version {} of the webbpsf data".format(webbpsf_data_version))

def test_basic_calc():
    import copy
    import matplotlib.pyplot as plt
    import numpy as np
    import jwst_pancake

    from pandeia.engine.calc_utils import build_default_calc

    config = build_default_calc('jwst', 'nircam', 'coronagraphy')

    # put in something more sensible for the target brightness than the defaults
    # How about a 7th mag star
    targetstar = config['scene'][0]
    targetstar['spectrum']['normalization']['type'] = 'photsys'
    targetstar['spectrum']['normalization']['norm_flux'] = 7
    targetstar['spectrum']['normalization']['norm_fluxunit'] = 'abmag'
    targetstar['spectrum']['normalization']['bandpass'] = 'johnson,v'
    del targetstar['spectrum']['normalization']['norm_wave'] # not needed for bandpass
    del targetstar['spectrum']['normalization']['norm_waveunit'] # not needed for bandpass
    targetstar['spectrum']['sed']['key'] = 'a5v'
    targetstar['id'] = 1 #each source must have a unique ID, starting at 1

    # We adopt a brighter but spectrally-mismatched reference
    config['strategy']['psf_subtraction_source'] = copy.deepcopy(targetstar)
    config['strategy']['psf_subtraction_source']['spectrum']['normalization']['norm_flux'] = 6
    config['strategy']['psf_subtraction_source']['spectrum']['sed']['key'] = 'a3v'
    config['strategy']['psf_subtraction_source']['id'] = 4

    jwst_pancake.engine.options.on_the_fly_PSFs = False
    jwst_pancake.engine.options.wave_sampling = 6
    jwst_pancake.engine.options.verbose = True # Uncomment this line to get logging output during calculation

    result = jwst_pancake.engine.calculate_subtracted(config)

    assert result['target'].shape == (101, 101), "Output target image is not expected shape"
    assert result['subtracted'].shape == (101, 101), "Output subtracted image is not expected shape"

    assert result['target'].sum() > 1e5, "too few counts in target image"
    assert np.abs(result['subtracted']).sum() / result['target'].sum() < 20, "Much more residual flux than expected in subtracted image"

    print("Basic PanCAKE Calculation Succeeded!")

    plt.figure(figsize=(10,4))
    plt.imshow(result['target'])
    plt.title('Target')
    plt.colorbar().set_label('e$^{-}$/s')

    plt.figure(figsize=(10,4))
    plt.imshow(result['subtracted'])
    plt.title('Target - Reference PSF')
    plt.colorbar().set_label('e$^{-}$/s')
