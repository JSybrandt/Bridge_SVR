# Bridge_SVR

## Overview

The following is a matlab script and various matlab data and figures.
totalData.mat contains simulated data. This represents a period of time in which cars drive over a bridge.

### Data

 - a = acceleration data
 - amax = max acceleration data
 - Td = time of day (counting from midnight - counts up to 48 to represent 2 days)
 - n = number of vehicles on bridge
 - Tact = actual temp at time of measurement
 - rh = relative humidity at time of measurement 
 - m = total mass of bridge
 - k = total stiffness of bridge
 - wn = natural frequency of bridge considering temp, humidity, and car effects
 - e = damping ratio
 - c = damping coefficient
 - t = time of dynamic measurement (time cars are on bridge)
 - u = displacement data
 - v = velocity data
 - umax = max displacements
 - vmax = max velocities
 
 ### Process
 
 We currently use LibSVM to do support vector regression on our data set. We are using a radial basis function kernel and seeing great results.
 
 
