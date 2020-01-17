'''

script with some of the firefly dependencies 


'''
import numpy as np 
from scipy.stats import chi2 


def hpf(flux, windowsize=0, w_start=0):
    ''' What does this one do ? High pass filtering ?
    '''
    D = np.size(flux)
    w = w_start
    f = flux

    # Rita's typical inputs for SDSS:
    # w = 10 # 10
    # windowsize = 20 # 20

    # My MaNGA inputs:
    # if w == 0 and windowsize == 0:
    #     w = 40
    #     windowsize = 0
    if w == 0 and windowsize == 0:
        w = int(D/100.0)
        windowsize = 0.0

    h           = np.fft.fft(f)
    h_filtered  = np.zeros(D,dtype=complex)
    window      = np.zeros(D)
    unwindow    = np.zeros(D)
    dw          = int(windowsize)
    dw_float    = float(windowsize)
    window[0]   = 1 # keep F0 for normalisation
    unwindow[0] = 1

    if windowsize > 0:
        for i in range(dw):
            window[w+i] = (i+1.0)/dw_float
            window[D-1-(w+dw-i)] = (dw_float-i)/dw_float

        window[w+dw:D-(w+dw)] = 1
    else:
        window[w:D-w] = 1

    unwindow        = 1 - window
    unwindow[0]     = 1

    h_filtered      = h * window
    un_h_filtered   = h*unwindow

    res     = np.real(np.fft.ifft(h_filtered))
    unres   = np.real(np.fft.ifft(un_h_filtered)) 
    res_out = (1.0+(res-np.median(res))/unres) * np.median(res) 

    return res_out 


def curve_smoother(x, y, smoothing_length):
	"""
	Smoothes a curve y = f(x) with a running median over a given smoothing length.

	Returns the smoothed array.
	
	Used internally in function determine_attenuation

	:param x: x
	:param y: y
	:param smoothing_length: smoothing length in the same unit than x.
	"""
	y_out = np.zeros(len(y))
	for w in range(len(x)):
		check_index = (x < x[w]+smoothing_length)&(x>x[w]-smoothing_length)
		y_out[w] = np.median(y[check_index])
	return y_out


def convert_chis_to_probs(chis,dof):
	"""
	Converts chi squares to probabilities.

	:param chis: array containing the chi squares.
	:param dof: array of degrees of freedom.
	"""
	chis = chis / np.min(chis) * dof
	prob =  1.0 - chi2.cdf(chis,dof)
	prob = prob / np.sum(prob)
	return prob


def calculate_averages_pdf(probs, light_weights, mass_weights, unnorm_mass, age, metal, sampling, dist_lum,  flux_units): 
    ''' Calculates light- and mass-averaged age and metallicities.
    Also outputs stellar mass and mass-to-light ratios.
    And errors on all of these properties.

    It works by taking the complete set of probs-properties and
    maximising over the parameter range (such that solutions with
    equivalent values but poorer probabilities are excluded). Then,
    we calculate the median and 1/2 sigma confidence intervals from 
    the derived 'max-pdf'.

    NB: Solutions with identical SSP component contributions 
    are re-scaled such that the sum of probabilities with that
    component = the maximum of the probabilities with that component.
    i.e. prob_age_ssp1 = max(all prob_age_ssp1) / sum(all prob_age_ssp1) 
    This is so multiple similar solutions do not count multiple times.

    Outputs a dictionary of:
    - light_[property], light_[property]_[1/2/3]_sigerror
    - mass_[property], mass_[property]_[1/2/3]_sigerror
    - stellar_mass, stellar_mass_[1/2/3]_sigerror
    - mass_to_light, mass_to_light_[1/2/3]_sigerror
    - maxpdf_[property]
    - maxpdf_stellar_mass
    where [property] = [age] or [metal]

    :param probs: probabilities
    :param light_weights: light (luminosity) weights obtained when model fitting
    :param mass_weights: mass weights obtained when normalizing models to data
    :param unnorm_mass: mass weights obtained from the mass to light ratio
    :param age: age
    :param metal: metallicity
    :param sampling: sampling of the property
    :param dist_lum: luminosity distance in cm
    '''

    # Sampling number of max_pdf (100:recommended) from options
    # Keep the age in linear units of Age(Gyr)
    log_age = age
    
    av = {} # dictionnary where values are stored :
    av['light_age'],av['light_age_1_sig_plus'],av['light_age_1_sig_minus'], av['light_age_2_sig_plus'], av['light_age_2_sig_minus'], av['light_age_3_sig_plus'], av['light_age_3_sig_minus'] = averages_and_errors(probs, np.dot(light_weights, log_age), sampling)
    
    av['light_metal'], av['light_metal_1_sig_plus'], av['light_metal_1_sig_minus'], av['light_metal_2_sig_plus'], av['light_metal_2_sig_minus'], av['light_metal_3_sig_plus'], av['light_metal_3_sig_minus'] = averages_and_errors(probs, np.dot(light_weights, metal), sampling)
    
    av['mass_age'], av['mass_age_1_sig_plus'], av['mass_age_1_sig_minus'], av['mass_age_2_sig_plus'], av['mass_age_2_sig_minus'], av['mass_age_3_sig_plus'], av['mass_age_3_sig_minus'] = averages_and_errors(probs, np.dot(mass_weights, log_age), sampling)
    
    av['mass_metal'], av['mass_metal_1_sig_plus'], av['mass_metal_1_sig_minus'], av['mass_metal_2_sig_plus'], av['mass_metal_2_sig_minus'], av['mass_metal_3_sig_plus'], av['mass_metal_3_sig_minus'] = averages_and_errors(probs, np.dot(mass_weights, metal), sampling)
    
    conversion_factor 	= flux_units * 4 * np.pi * dist_lum**2.0 # unit 1e-17 cm2 

    # Keep the mass in linear units until later M/M_{odot}.
    tot_mass = np.sum(unnorm_mass, 1) * conversion_factor
    av['stellar_mass'], av['stellar_mass_1_sig_plus'], av['stellar_mass_1_sig_minus'], av['stellar_mass_2_sig_plus'], av['stellar_mass_2_sig_minus'], av['stellar_mass_3_sig_plus'], av['stellar_mass_3_sig_minus'] = averages_and_errors(probs, tot_mass, sampling)
    return av


def averages_and_errors(probs, prop, sampling):
    ''' determines the average and error of a property for a given sampling
    
    returns : an array with the best fit value, +/- 1, 2, 3 sigma values.

    :param probs: probabilities
    :param  property: property
    :param  sampling: sampling of the property
    '''
    # This prevents galaxies with 1 unique solution from going any further. This is because the code crashes when constructing the likelihood
    # distributions. HACKY, but we need to think about this...
    if ((len(probs) <= 1) or (len(prop[~np.isnan(prop)]) <= 1)):
        best_fit, upper_onesig,lower_onesig, upper_twosig,lower_twosig, upper_thrsig,lower_thrsig = 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0
    else:
        xdf, y = max_pdf(probs, prop, sampling)
        cdf = np.zeros(np.shape(y))
        cdf_probspace = np.zeros(np.shape(y))

        for m in range(len(y)):
            cdf[m] = np.sum(y[:m])

        cdf = cdf / np.max(cdf)
        area_probspace = y*(xdf[1]-xdf[0])
        area_probspace = area_probspace/np.sum(area_probspace)
        indx_probspace = np.argsort(area_probspace)[::-1]
        desc_probspace = np.sort(area_probspace)[::-1]

        cdf_probspace = np.zeros(np.shape(desc_probspace))
        for m in range(len(desc_probspace)):
            cdf_probspace[m] = np.sum(desc_probspace[:m])

        av_sigs = [0.6827,0.9545,0.9973] # Median, + / - 1 sig, + / - 2 sig, + / - 3 sig

        # Sorts results by likelihood and calculates confidence intervals on sorted space
        index_close = find_closest(cdf_probspace, av_sigs)
        
        best_fit 			= xdf[indx_probspace[0]]
        upper_onesig,lower_onesig 	= np.max(xdf[indx_probspace[:index_close[0]]]),np.min(xdf[indx_probspace[:index_close[0]]])
        upper_twosig,lower_twosig 	= np.max(xdf[indx_probspace[:index_close[1]]]),np.min(xdf[indx_probspace[:index_close[1]]])
        upper_thrsig,lower_thrsig 	= np.max(xdf[indx_probspace[:index_close[2]]]),np.min(xdf[indx_probspace[:index_close[2]]])

        if np.size(xdf) == 0:
                raise Exception('No solutions found??? FIREFLY error (see statistics.py)')

    return [best_fit,upper_onesig,lower_onesig,upper_twosig,lower_twosig,upper_thrsig,lower_thrsig]


def max_pdf(probs, prop, sampling):
    ''' determines the maximum of a pdf of a property for a given sampling

    :param probs: probabilities
    :param property: property
    :param sampling: sampling of the property
    '''
    lower_limit 	= np.min(prop)
    upper_limit 	= np.max(prop)
    error_interval = np.round(upper_limit, 2) - np.round(lower_limit, 2)

    if np.round(upper_limit, 2) == np.round(lower_limit, 2) or error_interval <= abs((upper_limit/100.)*3):
        return np.asarray(prop),np.ones(len(probs))/np.size(probs)

    property_pdf_int= np.arange(lower_limit, upper_limit * 1.001, (upper_limit-lower_limit) /sampling ) + ( upper_limit - lower_limit) * 0.000001	
    prob_pdf 		= np.zeros(len(property_pdf_int))

    for p in range(len(property_pdf_int)-1):
        match_prop = np.where( (prop <= property_pdf_int[p+1]) & (prop > property_pdf_int[p]) )
        if np.size(match_prop) == 0:
            continue
        else:
            prob_pdf[p] = np.max( probs[match_prop] )

    property_pdf = 0.5 * (property_pdf_int[:-1] + property_pdf_int[1:])
    return property_pdf, prob_pdf[:-1]/np.sum(prob_pdf)


# --- dust --- 
def dust_calzetti_py(ebv,lam):
    '''
    '''
    output = []
    for i in lam:
        l = i / 10000.0 #converting from angstrom to micrometers
        if (l >= 0.63 and l<= 2.2):
            k=(2.659*(-1.857+1.040/l)+4.05)
        if (l < 0.63):
            k= (2.659*(-2.156+1.509/l-0.198/l**2+0.011/l**3)+4.05)
        if (l > 2.2):
            k= 0.0

        output.append(10**(-0.4 * ebv * k))
    return output


def dust_allen_py(ebv,lam):
    ''' Calculates the attenuation for the Milky Way (MW) as found in Allen (1976).'''
    from scipy.interpolate import interp1d
    wave = [1000,1110,1250,1430,1670,2000,2220,2500,2860,3330,3650,4000,4400,5000,5530,6700,9000,10000,20000,100000]
    allen_k = [4.20,3.70,3.30,3.00,2.70,2.80,2.90,2.30,1.97,1.69,1.58,1.45,1.32,1.13,1.00,0.74,0.46,0.38,0.11,0.00]
    allen_k = np.array(allen_k)*3.1

    total = interp1d(wave, allen_k, kind='cubic')
    wavelength_vector = np.arange(1000,10000,100)
    fitted_function = total(wavelength_vector)

    output = []
    for l in range(len(lam)):
        k = find_nearest(wavelength_vector,lam[l],fitted_function)
        output.append(10**(-0.4*ebv*k))
    return output


def dust_prevot_py(ebv,lam): 
    ''' Calculates the attenuation for the Small Magellanic Cloud (SMC) as found in Prevot (1984).'''
    from scipy.interpolate import interp1d
    wave = [1275,1330,1385,1435,1490,1545,1595,1647,1700,1755,1810,1860,1910,2000,2115,2220,2335,2445,2550,2665,2778,\
    2890,2995,3105,3704,4255,5291,10000]
    prevot_k = [13.54,12.52,11.51,10.80,9.84,9.28,9.06,8.49,8.01,7.71,7.17,6.90,6.76,6.38,5.85,5.30,4.53,4.24,3.91,3.49,\
    3.15,3.00,2.65,2.29,1.81,1.00,0.74,0.00]
    prevot_k = np.array(prevot_k)*2.72

    total = interp1d(wave, prevot_k, kind='linear')
    wavelength_vector = np.arange(1275,10000,100)
    fitted_function = total(wavelength_vector)

    output = []
    for l in range(len(lam)):
        k = find_nearest(wavelength_vector,lam[l],fitted_function)
        output.append(10**(-0.4*ebv*k))
    return output


def find_closest(A, target):
    ''' returns the id of the target in the array A.
    :param A: Array, must be sorted
    :param target: target value to be located in the array.
    '''
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx
