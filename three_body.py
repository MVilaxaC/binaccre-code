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

def threebody_fall(macc, mdon, racc, a, vfr, save=False):
    # System's units
    mass_unit = macc + mdon
    length_unit = a
    time_unit = (a**3 / (constants.G * (macc + mdon)))**0.5
    G = constants.G * mass_unit * time_unit**2 / length_unit**3
    
    dt = (1 | units.minute) / time_unit
    mu = mdon / mass_unit
    P = 2 * np.pi

    t = 0
    
    # Velocity, positions and acceleration at time 0
    vx = 0
    vy_comoving = (1 - vfr) * (constants.G * (macc + mdon) / a)**0.5
    rL1 = distance_to_L1(macc, mdon, a, vy_comoving) 
    x = (rL1 / length_unit) - mu
    vy = (1 - vfr)*(1 - mu) - x
    #vy = (1 - vfr) - x
    y = 0
    ax = omega_dx(x, y, mu) + 2*vy
    ay = omega_dy(x, y, mu) - 2*vx

    # Distance between parcel and accretor
    r = x + mu

    impact = 0

    data = {'t' : [t],
            'x' : [x],
            'y' : [y],
            'vx' : [vx],
            'vy' : [vy],
            't [min]' : [(t * time_unit).value_in(units.minute)],
            'x [RSun]' : [(x * length_unit).value_in(units.RSun)],
            'y [RSun]' : [(y * length_unit).value_in(units.RSun)],
            'vx [km s-1]' : [(vx * length_unit / time_unit).value_in(units.km * units.s**-1)],
            'vy [km s-1]' : [(vx * length_unit / time_unit).value_in(units.km * units.s**-1)]}
    df_traj = pd.DataFrame(data=data)

    count = 0

    while r >= racc / length_unit:
        #print('Time = {} hours,\tr = {} RSun'.format((t*time_unit).value_in(units.hour), (r*length_unit).value_in(units.RSun)))
        
        t += dt

        # Update positions, velocities and acceleration
        x += vx * dt
        y += vy * dt
        vx += ax * dt
        vy += ay * dt
        ax = omega_dx(x, y, mu) + 2*vy
        ay = omega_dy(x, y, mu) - 2*vx

        r = ((x + mu)**2 + y**2)**0.5
        
        if (save == True) & (count % 50 == 0):
            
            new_row = pd.Series({'t' : t,
                                 'x' : x,
                                 'y' : y,
                                 'vx' : vx,
                                 'vy' : vy,
                                 't [min]' : (t * time_unit).value_in(units.minute),
                                 'x [RSun]' : (x * length_unit).value_in(units.RSun),
                                 'y [RSun]' : (y * length_unit).value_in(units.RSun),
                                 'vx [km s-1]' : (vx * length_unit / time_unit).value_in(units.km * units.s**-1),
                                 'vy [km s-1]' : (vx * length_unit / time_unit).value_in(units.km * units.s**-1)})
            df_traj = pd.concat([df_traj, new_row.to_frame().T], ignore_index=True)
            df_traj.to_csv('./data/{:=05.2f}q_{:=09.5f}a_{:=06.3f}vfr_3body_trajectory.csv'.format(mdon / macc, a.value_in(units.au), vfr))

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
        count +=1
    
    if (save == True):
        new_row = pd.Series({'t' : t,
                            'x' : x,
                            'y' : y,
                            'vx' : vx,
                            'vy' : vy,
                            't [min]' : (t * time_unit).value_in(units.minute),
                            'x [RSun]' : (x * length_unit).value_in(units.RSun),
                            'y [RSun]' : (y * length_unit).value_in(units.RSun),
                            'vx [km s-1]' : (vx * length_unit / time_unit).value_in(units.km * units.s**-1),
                            'vy [km s-1]' : (vx * length_unit / time_unit).value_in(units.km * units.s**-1)})
        df_traj = pd.concat([df_traj, new_row.to_frame().T], ignore_index=True)
        df_traj.to_csv('./data/{:=05.2f}q_{:=09.5f}a_{:=06.3f}vfr_3body_trajectory.csv'.format(mdon / macc, a.value_in(units.au), vfr))

    return impact, t, x, y, vx, vy, ax, ay, mass_unit, length_unit, time_unit

def impact_data_v(macc, mdon, racc, a, vmin, vmax, vn):
    v_range = np.linspace(vmin, vmax, vn)

    data = {'vfr' : np.zeros(vn),
            't [hour]' : np.zeros(vn),
            'x [RSun]' : np.zeros(vn),
            'y [RSun]' : np.zeros(vn),
            'vx [km s-1]' : np.zeros(vn),
            'vy [km s-1]' : np.zeros(vn)}
    df = pd.DataFrame(data=data)
    
    for i in range(vn):
        print('vfr = {}'.format(v_range[i]))
        impact, t, x, y, vx, vy, ax, ay, m_unit, l_unit, t_unit = threebody_fall(macc, mdon, racc, a, v_range[i])
        if impact == 0:
            df.iloc[i] = [v_range[i], 0, 0, 0, 0, 0]
        elif impact == 1:
            df.iloc[i] = [v_range[i],
                          (t*t_unit).value_in(units.hour),
                          (x*l_unit).value_in(units.RSun),
                          (y*l_unit).value_in(units.RSun),
                          (vx*l_unit/t_unit).value_in(units.km * units.s**-1),
                          (vy*l_unit/t_unit).value_in(units.km * units.s**-1)]
    return df

def add_mass_and_spin(df, macc, racc, mtr, P):
    dm_list = []
    spin_list = []
    for i in range(len(df.index)):
        x = df.iloc[i]['x [RSun]'] | units.RSun
        y = df.iloc[i]['y [RSun]'] | units.RSun
        vx = df.iloc[i]['vx [km s-1]'] | units.km * units.s**-1
        vy = df.iloc[i]['vy [km s-1]'] | units.km * units.s**-1
        if df['t [hour]'].iloc[i] == 0:
            dm_list.append(0)
            spin_list.append(0)
        elif df['t [hour]'].iloc[i] >= 0:
            dm = mtr * P
            r = (x**2 + y**2)**0.5
            v = (vx**2 + vy**2)**0.5
            v_crit = (constants.G * (macc + dm) / racc)**0.5
            ang_imp = np.arccos((r**2 + y**2 - x**2)/(2*r*y)) - np.arccos(vy/v)
            v_t = v * np.sin(ang_imp)
            L_tot = racc * v_t * dm

            dm_list.append(dm.value_in(units.kg))
            spin_list.append(L_tot / (racc * (macc + dm) * v_crit))
        
    df['dm acc [kg]'] = dm_list
    df['dv'] = spin_list
    
    return df

def run_test(macc, mdon, racc, a, vmin, vmax, vn, mtr_e):
    P = 2 * np.pi * ((a**3) / (constants.G * (macc + mdon)))**0.5
    mtr = 10**mtr_e | units.MSun * units.yr**-1

    df = impact_data_v(macc, mdon, racc, a, vmin, vmax, vn)
    df = add_mass_and_spin(df, macc, racc, mtr, P)

    df.to_csv('./data/{:=05.2f}q_{:=09.5f}a_3body_rot.csv'.format(mdon / macc, a.value_in(units.au)))

if __name__ == "__main__":
    macc = 1 | units.MSun
    mdon = 1.2 | units.MSun
    racc = 1 | units.RSun
    a = 1.8 | units.au
    vfr = 1.4
    '''
    run_test(macc, 0.80 | units.MSun, racc, 1.3 | units.au, 1., 2., 400, -6)
    run_test(macc, 0.80 | units.MSun, racc, 1.8 | units.au, 1., 2., 400, -6)
    run_test(macc, 0.80 | units.MSun, racc, 2.2 | units.au, 1., 2., 400, -6)
    
    run_test(macc, 0.60 | units.MSun, racc, 1.3 | units.au, 1., 2., 400, -6)
    run_test(macc, 0.60 | units.MSun, racc, 1.8 | units.au, 1., 2., 400, -6)
    run_test(macc, 0.60 | units.MSun, racc, 2.2 | units.au, 1., 2., 400, -6)

    run_test(macc, 1.40 | units.MSun, racc, 1.3 | units.au, 1., 4.5, 400, -6)
    run_test(macc, 1.40 | units.MSun, racc, 1.8 | units.au, 1., 4.5, 400, -6)
    run_test(macc, 1.40 | units.MSun, racc, 2.2 | units.au, 1., 4.5, 400, -6)

    #threebody_fall(macc, mdon, racc, a, vfr)
    '''
    #run_test(macc, 0.50 | units.MSun, racc, a, 1., 2., 100, -6)
    #run_test(macc, 0.75 | units.MSun, racc, a, 1., 2., 100, -6)
    run_test(macc, macc, racc, a, 1., 2., 100, -6)
    #run_test(macc, 1.25 | units.MSun, racc, a, 1.5, 2.5, 100, -6)
    #run_test(macc, 1.50 | units.MSun, racc, a, 1.5, 2.5, 100, -6)
    
    #threebody_fall(macc, mdon, racc, a, 6.0, save=True)