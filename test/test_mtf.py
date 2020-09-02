import sys
import odak
from odak import np

def routine(image,location=[0,100,0,100],px_size=[10,10]):
    roi        = image[location[0]:location[1],location[2]:location[3]]
    mtf,freq,p = odak.measurement.modulation_transfer_function(roi,px_size,fit_degree=[3,3])
    return mtf,freq,p

def test():
    px_size   = 30./151.
    dist      = 580. # 58 cm
    angsize   = np.degrees(np.arctan(px_size/dist))    
    image     = np.random.rand(2000,2000)
    roi       = [340,360,1010,1050]
    label     = 'MTF'
    figure    = odak.visualize.plotshow()
    mtf,freq,p = routine(image,location=roi,px_size=[angsize,angsize])
    figure.add_plot(freq[1],np.abs(mtf[1]),label=label,mode='lines')
    figure.add_plot(freq[1],p[1](freq[1]),label=label,mode='lines')
#    figure.show()
    assert True==True

if __name__ == '__main__':
    sys.exit(test())
