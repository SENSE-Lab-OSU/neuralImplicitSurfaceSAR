#%% import
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
from torchinterp1d import interp1d
import torch
import h5py
#velocity of em-wave
cspeed = sc.c

#%% load SAR Image
def load_data_cvdomes(filepath,imageIndex):
    with h5py.File(filepath) as f:
    # read the frequencies used in the image
        data_of_interest_reference = f['f1'][imageIndex, 0]
        data_of_interest = np.array(f[data_of_interest_reference])
        freqs = data_of_interest
     # read the azimuth angles used in the image
        data_of_interest_reference = f['azimuthVals'][imageIndex, 0]
        data_of_interest = np.array(f[data_of_interest_reference])
        azimuthVals = data_of_interest
     # read the elevation angles used in the image
        data_of_interest_reference = f['elev1'][imageIndex, 0]
        data_of_interest = np.array(f[data_of_interest_reference])
        elevation = data_of_interest    
     # read the Image 
        data_of_interest_reference = f['im_final'][imageIndex, 0]
        data_of_interest = np.array(f[data_of_interest_reference])
        im_final = np.transpose(data_of_interest['real'] +1j*data_of_interest['imag'])
     # read the phase-history data used in the image
    #    data_of_interest_reference = f['phData'][imageIndex, 0]
    #    data_of_interest = np.array(f[data_of_interest_reference])
    #    phaseHistory = np.transpose(data_of_interest['real'] +1j*data_of_interest['imag'])
     # Y-Axis coordinates
        data_of_interest = np.array(f['yImage'])
        yImage = data_of_interest
     # X-Axis coordinates     
        data_of_interest = np.array(f['xImage'])
        xImage = data_of_interest
    return im_final,yImage,xImage,azimuthVals,freqs,elevation

# %%backProjection Tensorflow
def backprojection_farField(Wx,Wy,Nx,Ny,freq,AntAzim,elevationAngle,phdata,cspeed):
    Nfft = 8192  # Number of samples in FFT
    x0 = 0.  # Center of image scene in x direction (m)
    y0 = 0.  # Center of image scene in y direction (m)

    
    AntElev = torch.deg2rad(elevationAngle)
    minF = torch.min(freq)*torch.ones(AntAzim.shape) #Calculate the minimum frequency for each pulse (Hz)
          
    deltaF = torch.diff(freq[0:2,0])
                   
    K,Np = phdata.shape # Determine the number of pulses and the samples per pulse
   
    #Setup imaging grid
    x_vec = torch.linspace(start=x0 - Wx/2, end=x0 + Wx/2, steps=Nx)
    y_vec = torch.linspace(start=y0 - Wy/2, end=y0 + Wy/2, steps=Ny)
    x_mat,y_mat = torch.meshgrid(x_vec,y_vec)

    z_mat = torch.zeros(x_mat.shape)
    
    # imaging
    AntAz = torch.deg2rad(AntAzim) #Determine the azimuth angles of the image pulses (radians)
    #deltaAz = torch.abs(torch.mean(torch.diff(AntAz,axis=0),axis=0)) # Determine the average azimuth angle step size (radians)

    # Determine the total azimuth angle of the aperture (radians)
    #totalAz = torch.max(AntAz) - torch.min(AntAz)
   
    # Determine the maximum wavelength (m)
    #maxLambda = cspeed / (torch.mean(minF,axis=0) + deltaF*K)
   
    # Determine the maximum scene size of the image (m)
    maxWr = cspeed/(2*deltaF)   
    #maxWx = maxLambda/(2*deltaAz)
   
    # Determine the resolution of the image (m)
    #dr = cspeed/(2*deltaF*K)
    #dx = maxLambda/(2*totalAz)

    # Calculate the range to every bin in the range profile (m)
    r_vec = torch.linspace(-Nfft/2,Nfft/2-1,Nfft)*maxWr/Nfft
    max_r_vec=torch.max(r_vec)
    min_r_vec=torch.min(r_vec)
    
    # Initialize the image with all zero values
    im_final = torch.complex(torch.zeros(x_mat.shape),torch.zeros(x_mat.shape))
   
    # Loop through every pulse
    phdataZeropad = torch.concat([phdata,torch.complex(
        torch.zeros([Nfft-K,Np]),torch.zeros([Nfft-K,Np]))],dim=0)
    rc = torch.fft.fftshift(torch.fft.ifft(phdataZeropad),dim=0)
    for ii in np.arange(start=0, stop=Np, step=1):
        
    # Calculate differential range for each pixel in the image (m)
        dR = x_mat*torch.cos(AntElev[ii])*torch.cos(AntAz[ii]) + \
        y_mat*torch.cos(AntElev[ii])*torch.sin(AntAz[ii]) + \
        z_mat*torch.sin(AntElev[ii]) 
        
    # Calculate phase correction for image
        phCorr = torch.complex(torch.cos(4*np.pi*minF[ii]/cspeed*dR),\
        torch.sin(4*np.pi*minF[ii]/cspeed*dR)) 
    # Determine which pixels fall within the range swath
        i1 =torch.logical_and(torch.greater(dR,min_r_vec),torch.less_equal(dR,max_r_vec))        
        
        interpTerm1= interp1d(r_vec.unsqueeze(0),torch.real(rc[:,ii]).unsqueeze(0),dR)
        interpTerm2= interp1d(r_vec.unsqueeze(0),torch.imag(rc[:,ii]).unsqueeze(0),dR)
        
        t1=torch.where(i1,interpTerm1,0.0)
        t2=torch.where(i1,interpTerm2,0.0)
        t3=torch.where(i1,phCorr,torch.complex(torch.zeros(1),torch.zeros(1)))
    # Update the image using linear interpolation
        im_final = im_final + torch.complex(t1,t2)*t3
    return im_final
        
def show_plot(image,title='image'):
    fig = plt.figure()
    valmin = np.max(20*np.log10(np.abs(image.flatten()))) -50
    valmax = np.max(20*np.log10(np.abs(image.flatten())))
    plt.imshow(20*np.log10(np.absolute(image)),cmap="jet",vmin = valmin  ,vmax=valmax);
    plt.colorbar();
    plt.title(title)
    #plt.show()

#%% Extract the CV domes dataset
filepath = './data/Camry_passes_01_HH.mat'
im_index=2100
im_finalTrue,yImage,xImage,AntAzim,freq,elevationAngle = load_data_cvdomes(filepath,im_index)
AntAzim=torch.from_numpy(AntAzim)
freq=torch.from_numpy(freq)
elevationAngle=torch.from_numpy(elevationAngle)

K = freq.shape[1]
P = AntAzim.shape[0]
learning_rate=1e-4
#window1 = tf.complex(tf.linalg.matmul(tf.reshape(tf.signal.hamming_window(K),[K,1]),tf.reshape(tf.signal.hamming_window(P),[1,P])),0.0)
phdataVar=torch.complex(torch.zeros([K,P]),torch.zeros([K,P]))
phdataVar.requires_grad_(True)

im_true=torch.from_numpy(im_finalTrue)
freq=freq.T
Wx=9.0
Wy=9.0
Nx=90
Ny=90
Nfft = 8192  # Number of samples in FFT

#%% Forward operator and loss
def forward():
    im_final = backprojection_farField(Wx=Wx,Wy=Wy,Nx=Nx,Ny=Ny,freq=freq,
        AntAzim=AntAzim,elevationAngle=elevationAngle,phdata=phdataVar,cspeed=cspeed) 
    return im_final

def loss(im1, imTrue):
    return ((im1 - imTrue)**2).mean()

optimizer = torch.optim.SGD([phdataVar], lr=learning_rate)
im1 = None
#%% epochs
for step in np.arange(100):
    im1=forward()
    ret_loss=loss(im1,im_true)
    ret_loss.backward()    
    print('Step_ph={}, Loss={}\n'.format(step,ret_loss))
    optimizer.step()
    optimizer.zero_grad()

    
#%% Check Images

# im_return = backprojection_farField(Wx=9.0,Wy=9.0,Nx=90,Ny=90,freq=freq,
#     AntAzim=AntAzim,elevationAngle=elevationAngle,phdata=phdataVar,cspeed=cspeed)


# show_plot(im_return.numpy(),"Recovered image")    
# show_plot(im_finalTrue,"True image")
# err1 = np.linalg.norm(im_return.numpy()-im_finalTrue,ord='fro')/np.linalg.norm(im_finalTrue,ord='fro')*100
# # %%

# %%
