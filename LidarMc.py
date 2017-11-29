###############################
###############################
###############################
##        LidarMC.py         ##
##        written by:        ##
##      Brian Collister      ##
##                           ##
###############################
###############################
###############################



# Import Modules
import math
import numpy as np
import random
from numpy import linalg as LA
from itertools import product
import matplotlib.pyplot as plt
import csv

############################################ Main Function ############################################
def main():

    ######### Set Constants #########

    ## Lidar ##
    detectorRad = 1.5e-1 # detector radius (m)
    detectorDiam = detectorRad * 2 #  detector diameter (m)
    detectorArea = math.pi * detectorRad**2

    FOV = np.deg2rad(20) # half-angle FOV (set in degrees, converts to radians)

    xd , yd, zd = 0.04, 0, 0 # detector position relative to lidar

    ## Medium ##
    a = 0.1 # absorption coefficient (m^-^1)
    b = 0.5 # scattering coefficient (m^-^1)
    c = a + b # beam attenuation coefficient (m^-^1)

    omega = b/c # single scattering albedo

    mueller = np.matrix([[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]])

    # theta is the scattering angle from the porpegation vector of a photon in !!!!!degrees!!!!!
    thetaArray = np.array([0.1,0.12589,0.15849,0.19953,0.25119,0.31623,0.39811,0.50119,0.63096,0.79433,1.0,1.2589,
                1.5849,1.9953,2.5119,3.1623,3.9811,5.0119,6.3096,7.9433,10,15,20,25,30,35,40,45,50,
                55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,
                165,170,175,180])
    # pTheta idescribes the cumulative probability of scattering at angles from 0 - 180 !!!!degrees!!!!

    pTheta = np.array([0.043292475,0.051470904,0.061194794,0.07278701,0.086643078,
        0.103058065,0.122316295,0.144714728,0.170373413,0.199273394,
        0.23138308,0.266595059,0.304726101,0.345302331,0.387470778,
        0.431035572,0.475832438,0.5216971,0.568433689,0.615808429,
        0.663612814,0.749541922,0.806027674,0.847981298,0.877235105,
        0.898464649,0.915776837,0.929298035,0.940418273,0.949074367,
        0.956340431,0.962153282,0.967271119,0.971314842,0.975042649,
        0.978012258,0.9808555,0.983003728,0.985215139,0.986857901,
        0.988690213,0.990017059,0.991533455,0.992607569,0.993934416,
        0.99481898,0.995893094,0.996524926,0.997472673,0.997978139,
        0.998736337,0.999052252,0.999620901,0.999747267,1.0]);

    ## Monte Carlo Parameters

    #nPhotons = 1000 # number of photons to trace # < 1 second
    #nPhotons = 10000 # number of photons to trace # ~ 1.5 seconds
    #Photons = 100000 # number of photons to trace # ~10 seconds
    #nPhotons = 1000000 # number of photons to trace # ~1.5 min
    nPhotons = 1000000 # number of photons to trace # ~10 min

    ## Storage Variables - these variables are appended to every time a photon is counted
    signal = [] # signal list
    distance = [] # distance list
    binEdges = [] # distance bin edges


    ######### Start Photon Tracing #########
    for i in range(nPhotons):

        #initialize new photon
        x1, y1, z1 = 0, 0, 0    # photon position
        x2, y2, z2 = 0, 0, 0    # photon position

        mux1, muy1, muz1 = 0, 0, 1  # direction cosines used to propegate Photons
        mux2, muy2, muz2 = 0, 0, 0  # direction cosines used in rotation math

        # Photon Status Variables
        status = int(1) # 1 = alive, 0 = dead
        rTotal = 0 # total pathlength traveled
        weight = 1 # current weight of photon (omega^nscat)
        nScat = 0
        while (status == 1) and (nScat < 10):
            # Move Photon
            r = -1 * math.log(RNG())/c # generate a random propegation distanec
            x2 = x1 + mux1 * r # update x-position
            y2 = y1 + muy1 * r # update y-position
            z2 = y2 + muz1 * r # update z-position

            # Update Storage variables
            rTotal = rTotal + r

            # Did the photon hit the detector plane?
            if(z2 < zd): # if true: photon passed through the plane of the detector

                fd = (zd - z1)/(z2 - z1) # calculate the multiplicative factor for the distance along the photon trajectory to the detector
                xT = x1 + fd*(x2-x1) # calculate x-location that photon hits plane
                yT = x1 + fd*(y2-y1) # calculate y-location that photon hits plane
                hitRad = math.sqrt((xT-xd)**2 + (yT-yd)**2) # distance from detector center

                # Did the photon hit the detector?
                if(hitRad > detectorRad): # if true, the photon does not pass through the detector
                    status = 0 # if the photon hits outside of the detector, the photon is lost
                # Did the photon hit within the FOV?
                else:

                    anglei = math.pi - intersectionAngle(x1,y1,z1,x2,y2,z2) # calculate the angle betweenthe
                    if(anglei <= FOV):
                        rTotal = rTotal - (r-(fd*r)) # calculate the distance
                        signal.append(omega**nScat) # count photon in the signal
                        distance.append(rTotal) # record the total pathlength traveled by the photon

                        status = 0 # kill the photon and move on to the next one

                    else: status = 0 # kill the photon and move on to the next one

            else:

                # Scatter Photon
                theta = np.deg2rad(np.interp(RNG(),pTheta,thetaArray))
                phi = 2*np.pi*RNG()

                mux2 = updateDirCosX(theta, phi, mux1, muy1, muz1)
                muy2 = updateDirCosY(theta, phi, mux1, muy1, muz1)
                muz2 = updateDirCosZ(theta, phi, mux1, muy1, muz1)

                gammaCalc(muz1, muz2, theta, phi)

                ## Direction Cosines Should Satisfy sqrt(mux^2+muy^2+muz^2) = 1
                #check = math.sqrt(mux1**2 + muy1**2 +muz1**2)
                #print(check)

                # reset position variables
                x1 = x2
                y1 = y2
                z1 = z2

                mux1 = mux2
                muy1 = muy2
                muz1 = muz2


                nScat = nScat+1


### Aggregate Photon Weights and Distances to Create Lidar Signal

    dbin = 0.25 # bin widths
    for i in range(int(np.ceil(max(distance))/dbin)): # create a list of the bin edges
        binEdges.append(i*dbin+dbin)
    distanceBin = np.digitize(distance,binEdges) # index the distance bin that each photon belongs to

## Final Variables
    z = np.asarray(binEdges)/2
    lidarSignal = accum(np.asarray(distanceBin),np.asarray(signal), size = np.size(binEdges)) # Signal
    lidarSignalRC = lidarSignal * z**2 # Range-Corrected Signal

## Write To CSV File

    with open('lidarMCout.csv','w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(['LidarMC.py output file:'])
        writer.writerow(['Detector Parameters:'])
        writer.writerow(['Radius(m) = '+ str(detectorRad)])
        writer.writerow(['FOV(rad) = '+ str(FOV)])
        writer.writerow(['Medium Parameters:'])
        writer.writerow(['a(m^-1) = '+ str(a)])
        writer.writerow(['b(m^-1) = '+ str(b)])
        writer.writerow(['c(m^-1) = '+ str(c)])
        writer.writerow(['Run Parameters:'])
        writer.writerow(['# of photons = ' + str(nPhotons)])
        writer.writerow(['z','signal'])
        for i in range(z.shape[-1]):
            writer.writerow([z[i], lidarSignal[i]])









### Plot Lidar Signal
    plt.figure(1)
    plt.plot(z,lidarSignal,'.')
    plt.xlabel('Distance(m)', Fontsize = 12)
    plt.ylabel('P-Lidar (rel.)', Fontsize = 12)
    plt.show()

    plt.figure(2)
    plt.plot(z,lidarSignalRC,'.')
    plt.xlabel('Distance(m)', Fontsize = 12)
    plt.ylabel('P-Lidar (rel.)', Fontsize = 12)
    plt.show()

    #plt.figure(3)
    #plt.plot(z,lidarSignalRCln,'.')
    #plt.xlabel('Distance(m)', Fontsize = 12)
    #plt.ylabel('P-Lidar (rel.)', Fontsize = 12)
    #plt.show()
############### Subroutines ####################

# Update the direction cosine for X
def updateDirCosX(theta, phi, mux, muy, muz):
    if(math.fabs(muz)>0.99): # if muz is very close to 1...
        muxPrime = math.sin(theta) * math.cos(phi)
    else:
        muxPrime = (1/(math.sqrt(1-muz**2))) * math.sin(theta) * (mux*muz*math.cos(phi)-muy*math.sin(phi)) + mux * math.cos(theta)
    return muxPrime


# Update the dierection cosine for Y
def updateDirCosY(theta, phi, mux, muy, muz):
    if(math.fabs(muz) > 0.99): # if muz is very close to 1...
        muyPrime = math.sin(theta) * math.sin(phi)
    else:
        muyPrime = (1/(math.sqrt(1-muz**2))) * math.sin(theta) * (muy*muz*math.cos(phi)+mux*math.sin(phi)) + muy * math.cos(theta)
    return muyPrime


# Update the dierection cosine for Z
def updateDirCosZ(theta, phi, mux, muy, muz):
    if math.fabs(muz) > 0.99: # if muz is very close to 1...
        muzPrime = math.cos(theta) * muz/math.fabs(muz)
    else:
        muzPrime = (-1*math.sqrt(1-muz**2)) * math.sin(theta) * math.cos(phi) + muz*math.cos(theta)
    return muzPrime

# GammaCalc
# J.W. Houvenier. 1968. Symmetry Relationships for Scattering of Polarized Light in a Slab of Randomly Oriented Particles
def gammaCalc(muz1, muz2, theta, phi):

    if math.pi < phi < 2 * math.pi:
        gammaCos = (muz2*math.cos(theta) - muz1) / math.sqrt((1 - math.cos(theta)**2) * (1 - muz2**2))
        below = math.sqrt((1 - math.cos(theta)**2) * (1 - muz2**2))
    else:
        gammaCos = (muz2*math.cos(theta) - muz1) / (-1 * math.sqrt((1-math.cos(theta)**2) * (1 - muz2**2)))
        below = (-1 * math.sqrt((1-math.cos(theta)**2) * (1 - muz2**2)))
    #print('above' + str((muz2*math.cos(theta) - muz1)))
    #print('below'+ str(below))
    #print(gammaCos)





# Update Stokes Vector
def updateStokes(stokes, mueller, phi, gamma):

    rotationIn = np.matrix( [[1, 0, 0, 0],
                            [0, math.cos(-2*phi), math.sin(-2*phi), 0],
                            [0, -1*math.sin(-2*phi), math.cos(-2*phi), 0], [
                            0, 0, 0, 1]])
    rotationOut = np.matrix( [[1, 0, 0, 0],
                             [0, math.cos(-2*gamma), math.sin(-2*gamma), 0],
                             [0, -1*math.sin(-2*gamma), math.cos(-2*gamma), 0],
                             [0, 0, 0, 1]])

    stokesPrime = rotationOut * mueller * rotationIn *stokes

    return stokesPrime

# Random Number Generator
def RNG():
# RNG() - generates a random number between 0 and 1
# RNG uses a
    n = random.random()
    return n

# Intersection Angle
def intersectionAngle(x1,y1,z1,x2,y2,z2):
# intersectionAngle(c1,c2) - Calculates the intersection angle between a photon trajectory and the plane made by the lidar detector
    # In order to determine if a photon has entered the detector within the FOV of the detector, this function calculates the
    # angle between the unit vector normal to the detector plane and the propegation vector
    c1 = np.array([x1,y1,z1])   # an array containing the first x,y,z points of the propegation vector
    c2 = np.array([x2,y2,z2])   # an array containing the second x,y,z points of the propegation vector
    u = np.array([0,0,1])   # create a unit vector normal to the plane of the detector at the origin
    v = c2-c1   # convert the propegation vector to a unit vector
    angle = math.atan2(LA.norm(np.cross(u,v)),np.dot(u,v))  # calculate the angle between the propegation and the vector normal to the detector plane
    return angle

# accumarray like function
def accum(accmap, a, func=None, size=None, fill_value=0, dtype=None):
    # Check for bad arguments and handle the defaults.
    if accmap.shape[:a.ndim] != a.shape:
        raise ValueError("The initial dimensions of accmap must be the same as a.shape")
    if func is None:
        func = np.sum
    if dtype is None:
        dtype = a.dtype
    if accmap.shape == a.shape:
        accmap = np.expand_dims(accmap, -1)
    adims = tuple(range(a.ndim))
    if size is None:
        size = 1 + np.squeeze(np.apply_over_axes(np.max, accmap, axes=adims))
    size = np.atleast_1d(size)

    # Create an array of python lists of values.
    vals = np.empty(size, dtype='O')
    for s in product(*[range(k) for k in size]):
        vals[s] = []
    for s in product(*[range(k) for k in a.shape]):
        indx = tuple(accmap[s])
        val = a[s]
        vals[indx].append(val)

    # Create the output array.
    out = np.empty(size, dtype=dtype)
    for s in product(*[range(k) for k in size]):
        if vals[s] == []:
            out[s] = fill_value
        else:
            out[s] = func(vals[s])

    return out


if __name__ == "__main__":
    main()
