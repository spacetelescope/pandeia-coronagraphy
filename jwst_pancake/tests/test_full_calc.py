import matplotlib.pyplot as plt

from pandeia.engine.calc_utils import build_default_calc
import jwst_pancake as pancake


def test_full_calculation(plot=False, wave_sampling=2):
    """ Perform a full test calculation, as implemented in the
    stsci_pancake_installation.ipynb notebook

    Parameters:
    ------------
    plot : bool
        Display diagnostic plots of the output
    wave_sampling : int
        Number of wavelengths to use in the calculation
    """

    if wave_sampling < 2:
        raise ValueError("Pandeia calc requires at least 2 wavelengths. Adjust wave_sampling.")

    config = build_default_calc('jwst', 'nircam', 'coronagraphy')

    pancake.engine.options.on_the_fly_PSFs = True
    pancake.engine.options.wave_sampling = wave_sampling
    pancake.engine.options.verbose = True # Uncomment this line to get logging output during calculation

    result = pancake.engine.calculate_subtracted(config)

    assert len(result['target'].shape) ==2, "Expected to get a target image array but didn't"
    assert isinstance(result['references'], list), "Expected a list of references but didn't get it"
    assert len(result['references'][0].shape) ==2, "Expected first reference to be an image array but it isn't"
    assert len(result['subtracted'].shape) ==2, "Expected to get a subtracted image array but didn't"

    print("Basic PanCAKE Calculation Succeeded!")

    if plot:
        plt.figure(figsize=(10,4))
        plt.imshow(result['subtracted'])
        plt.title('Target - Reference Counts')
        plt.colorbar().set_label('e$^{-}$/s')
