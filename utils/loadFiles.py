import h5py
import numpy as np

#%% load SAR Image
def load_data_cvdomes(filepath,imageIndex):
    with h5py.File(filepath) as f:
    # read the frequencies used in the image
        data_of_interest_reference = f['freq_save'][imageIndex, :]
        data_of_interest = np.array(f[data_of_interest_reference])
        freqs = data_of_interest
     # read the azimuth angles used in the image
        data_of_interest_reference = f['az_save'][imageIndex, :]
        data_of_interest = np.array(f[data_of_interest_reference])
        azimuthVals = data_of_interest
     # read the elevation angles used in the image
        data_of_interest_reference = f['el_save'][imageIndex, :]
        data_of_interest = np.array(f[data_of_interest_reference])
        elevation = data_of_interest    
     # read the Image 
        data_of_interest_reference = f['im_save'][imageIndex,:,:,:]
        data_of_interest = np.array(f[data_of_interest_reference])
        im_final = np.transpose(data_of_interest['real'] +1j*data_of_interest['imag'])
     # read the phase-history data used in the image
        data_of_interest_reference = f['ph_save'][imageIndex, :]
        data_of_interest = np.array(f[data_of_interest_reference])
        phaseHistory = np.transpose(data_of_interest['real'] +1j*data_of_interest['imag'])
     # Y-Axis coordinates
      #  data_of_interest = np.array(f['yImage'])
      #  yImage = data_of_interest
     # X-Axis coordinates     
      #  data_of_interest = np.array(f['xImage'])
      #  xImage = data_of_interest
    return im_final,azimuthVals,freqs,elevation,phaseHistory