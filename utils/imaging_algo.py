import torch
import numpy as np
from torchinterp1d import interp1d

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
    deltaAz = torch.abs(torch.mean(torch.diff(AntAz,axis=0),axis=0)) # Determine the average azimuth angle step size (radians)

    # Determine the total azimuth angle of the aperture (radians)
    totalAz = torch.max(AntAz) - torch.min(AntAz)
   
    # Determine the maximum wavelength (m)
    maxLambda = cspeed / (torch.mean(minF,axis=0) + deltaF*K)
   
    # Determine the maximum scene size of the image (m)
    maxWr = cspeed/(2*deltaF)   
    maxWx = maxLambda/(2*deltaAz)
   
    # Determine the resolution of the image (m)
    dr = cspeed/(2*deltaF*K)
    dx = maxLambda/(2*totalAz)

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
        
        interpTerm1= interp1d(x=r_vec.unsqueeze(0),y=torch.real(rc[:,ii]).unsqueeze(0),xnew=dR[i1].unsqueeze(0))
        interpTerm2= interp1d(x=r_vec.unsqueeze(0),y=torch.imag(rc[:,ii]).unsqueeze(0),xnew=dR[i1].unsqueeze(0))
        
        t1=torch.where(i1,interpTerm1,0.0)
        t2=torch.where(i1,interpTerm2,0.0)
        t3=torch.where(i1,phCorr,torch.complex(0.0,0.0))
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