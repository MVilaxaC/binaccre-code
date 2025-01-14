import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from amuse.units import units, constants
from amuse.units.optparse import OptionParser

def omega_dx(x, y, mu):
    ox1 = x
    ox2 = (mu - 1) * (mu + x) / ((mu + x)**2 + y**2)**1.5
    ox3 = mu * (mu + x - 1) / ((mu + x - 1)**2 + y**2)**1.5
    return ox1 + ox2 + ox3

def omega_dy(x, y, mu):
    oy1 = y
    oy2 = y * (mu - 1) / ((mu + x)**2 + y**2)**1.5
    oy3 = mu * y / ((mu + x - 1)**2 + y**2)**1.5
    return oy1 + oy2 + oy3

def L1_eq(acc_mass, don_mass, sma, r, v):
    eq = ( constants.G * ( don_mass / ((sma-r)**2 ))
           - constants.G * ( acc_mass / (r**2) )
           + (v**2/r) )
    return eq

# Computes the distance from the accretor's center to the L1 point of the system    
def distance_to_L1(acc_mass, don_mass, sma, v):
    guess1 = np.linspace(1000, sma.value_in(units.m)-1, 1000) | units.m
    values1 = []
    for i in guess1:
        eq = L1_eq(acc_mass, don_mass, sma, i, v)
        values1.append(np.abs(eq))
    sorted1 = sorted(zip(values1, guess1), key = lambda x: x[0])

    guess2 = np.linspace(sorted1[0][1].value_in(units.m), sorted1[1][1].value_in(units.m), 100) |units.m
    values2 = []
    for j in guess2:
        eq = L1_eq(acc_mass, don_mass, sma, j, v)
        values2.append(eq)
        sorted2 = sorted(zip(values2, guess2), key = lambda x: x[0])
    return sorted2[0][1]

def threebody_fall(macc, mdon, racc, a, vfr):
    # System's units
    mass_unit = macc + mdon
    length_unit = a
    time_unit = (a**3 / (constants.G * (macc + mdon)))**0.5
    G = constants.G * mass_unit * time_unit**2 / length_unit**3
    
    dt = (1 | units.hour) / time_unit
    mu = mdon / mass_unit
    P = 2 * np.pi

    t = 0
    
    # Velocity, positions and acceleration at time 0
    vx = 0
    vy = (1 - vfr) * G**0.5 # Orbital velocity equals G**0.5 in these units
    rL1 = distance_to_L1(macc, mdon, a, (1 - vfr) * (constants.G * mass_unit/length_unit)**0.5) 
    x = (rL1 / length_unit) - mu
    y = 0
    ax = omega_dx(x, y, mu) + 2*vy
    ay = omega_dy(x, y, mu) - 2*vx

    # Distance between parcel and accretor
    r = x + mu

    impact = 0

    while r > racc / length_unit:
        print('Time = {} hours,\tr = {} RSun'.format((t*time_unit).value_in(units.hour), (r*length_unit).value_in(units.RSun)))
        
        t += dt

        # Update positions, velocities and acceleration
        x += vx * dt
        y += vy * dt
        vx += ax * dt
        vy += ay * dt
        ax = omega_dx(x, y, mu) + 2*vy
        ay = omega_dy(x, y, mu) - 2*vx

        r = ((x + mu)**2 + y**2)**0.5

        break1 = (r - rL1 / length_unit > 0.001)
        break2 = (t >= P)
        if break1:
            print('\tr ({}) > rL1 ({})'.format(r, rL1 / length_unit))
            impact = 0
            break
        elif break2:
            print('\tt ({}) >= P ({})'.format(t, P))
            impact = 0
            break
        else:
            impact = 1
    return impact, t, x, y, vx, vy, ax, ay

def impact_data_v(macc, mdon, racc, a, vmin, vmax, vn):
    v_range = np.linspace(vmin, vmax, vn)

    data = {'vfr' : np.zeros(vn),
            't [hour]' : np.zeros(vn),
            'vx [km s-1]' : np.zeros(vn),
            'vy [km s-1]' : np.zeros(vn)}
    df = pd.DataFrame(data=data)
    
    for i in range(vn):
        impact, t, x, y, vx, vy, ax, ay = threebody_fall(macc, mdon, racc, a, v_range[i])
            if impact == 0:
                df.iloc[i] = [v_range[i], 0, 0, 0]
            elif impact == 1:
                df.iloc[i] = [v_range[i], t, vx, vy]
    return df

if __name__ == "__main__":
    macc = 1 | units.MSun
    mdon = 1.2 | units.MSun
    racc = 1 | units.RSun
    a = 1.3 | units.au
    vfr = 1.4

    threebody_fall(macc, mdon, racc, a, vfr)