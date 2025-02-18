import os
import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from amuse.units import units, constants, nbody_system
from amuse.datamodel import Particles, Particle, ParticlesSuperset
from amuse.units.optparse import OptionParser

from overflow_fraction import read_file
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap

def max_e(table_name):
    print(table_name.split('.')[0])
    table = pd.read_table(table_name, header=None, names=['filenames'])
    max_e = []
    for f in table['filenames']:
        macc, mdon, a, e, v_fr, v_extra, df = read_file('./data/'+table_name.split('.')[0]+'/'+f)
        max_e.append(df.loc[df['flag impact'] == 0.0]['e p'].max())
        #print(df['e p'].max(), df.loc[df['e p'] == df['e p'].max()]['flag impact'])
        sl = df.loc[(df['flag impact'] == 0.0) & (df['e p'] >= 0.99)]
        if sl.shape[0] > 0:
            print('    '+f+'    {}/{}'.format(sl.shape[0], df.loc[df['flag impact'] == 0.0].shape[0]))
    #print(max_e)

def show_test(filename):
    df = pd.read_csv(filename)
    fig, ax = plt.subplots(figsize = (6,4), dpi=300)

    #ax.plot(df['t [yr]'], df['m acc [MSun]'])
    #ax.plot(df['t [yr]'], df['m don [MSun]'])
    ax.plot(df['t [yr]'], df['a [AU]'])
    plt.show()

def coef(tabname2, tabname3):
    df2 = pd.read_csv(tabname2, index_col=0)
    df3 = pd.read_csv(tabname3, index_col=0)

    # Clean from NaN's
    mask2 = [not x for x in df2['ang imp [pi rad]'].isnull()]
    df2 = df2.iloc[mask2]

    mask3 = [not x for x in df3['ang imp [pi rad]'].isnull()]
    df3 = df3.iloc[mask3]

    # Get zeros and maximum angle of impact
    zero2 = df2.loc[df2['ang imp [pi rad]'] >= 0].iloc[-1]
    zero3 = df3.loc[df3['ang imp [pi rad]'] >= 0].iloc[-1]

    max2 = df2.loc[df2['ang imp [pi rad]'] == df2['ang imp [pi rad]'].max()]
    max3 = df3.loc[df3['ang imp [pi rad]'] == df3['ang imp [pi rad]'].max()]

    amp2 = zero2['vfr'] - max2['vfr']
    amp3 = zero3['vfr'] - max3['vfr']
    ratio = amp3.values[0]/amp2.values[0]

    print(amp3,'\n', zero2 * ratio - max2 * ratio)

    vfr2 = df2['vfr']*ratio + (zero3['vfr'] - zero2['vfr'] * ratio)

    fig, ax = plt.subplots(figsize = (6,4), dpi=300)

    ax.plot(vfr2, df2['ang imp [pi rad]'])
    ax.plot(df3['vfr'], df3['ang imp [pi rad]'])
    ax.set_xlim(right=zero3['vfr'])

    plt.show()
    plt.close()

def plot_zeros(tab_list):
    vfr_list, q_list = [[], []]
    for t in tab_list:
        # Get data from table name
        q = float(t.split('/')[-1].split('q_')[0])
        a = float(t.split('q_')[-1].split('a_')[0]) | units.au
        mdon = q | units.MSun
        macc = 1 | units.MSun

        # Read table and get vfr where angle of impact is 0
        df = pd.read_csv(t, index_col=0)
        P = (2 * np.pi * ((a**3) / (constants.G * (macc + mdon)))**0.5).value_in(units.hour)
        truean = df['t [hour]'] * 2 * np.pi / P

        x = df['x [RSun]'] * np.cos(truean)
        y = df['y [RSun]'] * np.cos(truean)
        r = (x**2 + y**2)**0.5
        omega = ((constants.G * (macc + mdon) / a**3)**0.5).value_in(units.s**-1)
        vx = df['vx [km s-1]'] + (-1 * df['y [RSun]'] * 695700) * omega
        vy = df['vy [km s-1]'] + (df['x [RSun]'] * 695700) * omega
        v = (vx**2 + vy**2)**0.5
        ang_imp = np.arccos(y/r) + np.arccos(vy/v) - np.pi
        df['ang imp [pi rad]'] = ang_imp

        print(df['ang imp [pi rad]'])

        zero = df.loc[df['ang imp [pi rad]'] >= 0].iloc[-1]

        vfr_list.append(zero['vfr'])
        q_list.append(q)

    vfr_arr = np.array(vfr_list)
    q_arr = np.array(q_list)

    # Straight line that crosses first and last point
    q_line = np.linspace(0.25, 1.75, 10)
    #vfr_line = vfr_arr[0] + (q_line - q_arr[0]) * (vfr_arr[-1] - vfr_arr[0])/(q_arr[-1] - q_arr[0])
    z = np.polyfit(q_arr, vfr_arr, 2)
    vfr_pol = np.poly1d(z)
    vfr_line = vfr_pol(q_line)

    print(z)

    fig, ax = plt.subplots(figsize = (6,4), dpi=300)

    ax.scatter(q_list, vfr_list)
    ax.plot(q_line, vfr_line)

    plt.show()
    plt.close()

def plot_zeros_many(tab_list_list):
    
    props = {'fontsize': 11}
    colors = ['firebrick', 'gold', 'limegreen']
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 3)))
    
    fig, ax = plt.subplots(figsize = (4,4), dpi=300)

    for tab_list in tab_list_list:
        c = next(color)
        vfr_list, q_list = [[], []]
        for t in tab_list:
            # Get data from table name
            q = float(t.split('/')[-1].split('q_')[0])
            a = float(t.split('q_')[-1].split('a_')[0]) | units.au
            mdon = q | units.MSun
            macc = 1 | units.MSun

            # Read table and get vfr where angle of impact is 0
            df = pd.read_csv(t, index_col=0)
            P = (2 * np.pi * ((a**3) / (constants.G * (macc + mdon)))**0.5).value_in(units.hour)
            truean = df['t [hour]'] * 2 * np.pi / P

            x = df['x [RSun]'] * np.cos(truean)
            y = df['y [RSun]'] * np.cos(truean)
            r = (x**2 + y**2)**0.5
            omega = ((constants.G * (macc + mdon) / a**3)**0.5).value_in(units.s**-1)
            vx = df['vx [km s-1]'] + (-1 * df['y [RSun]'] * 695700) * omega
            vy = df['vy [km s-1]'] + (df['x [RSun]'] * 695700) * omega
            v = (vx**2 + vy**2)**0.5
            ang_imp = np.arccos(y/r) + np.arccos(vy/v) - np.pi
            df['ang imp [pi rad]'] = ang_imp

            zero = df.loc[df['ang imp [pi rad]'] >= 0].iloc[-1]

            vfr_list.append(zero['vfr'])
            q_list.append(q)

        label = r'$a$ = '+'{:=05.2f}'.format(a.value_in(units.au))

        vfr_arr = np.array(vfr_list)
        q_arr = np.array(q_list)

        # Straight line that crosses first and last point
        q_line = np.linspace(0.25, 1.75, 10)
        pq2 = (-0.02162484 * a.value_in(units.au)**2
               +0.06286028 * a.value_in(units.au)
               +0.05295173)
        pq1 = (0.03202641 * a.value_in(units.au)**2
               -0.0909287 * a.value_in(units.au)
               +1.02136945)
        pq0 = (-0.01422374 * a.value_in(units.au)**2
               +0.06100643 * a.value_in(units.au)
               +0.70129542)
        vfr_line = (pq2 * q_line**2
                    + pq1 * q_line
                    + pq0)
        ax.plot(q_line, vfr_line, color=c)
        ax.scatter(q_list, vfr_list, s=25, color=c, label=label, marker='o', facecolor='none', zorder=3)

    ax.set_xlabel(r'$q$', **props)
    ax.set_ylabel(r'$v_{extra} / v_{per}$ ($\alpha_{imp}$ = 0)', **props)
    
    ax.legend(loc=2)
    
    plt.subplots_adjust(left=0.175, right=0.95, top=0.95, bottom=0.125)

    #plt.show()
    plt.savefig('./plots/coefficients_zero_fit.png')
    plt.close()

def coefficients_zeros():
    '''
    [ 0.0981241   0.95728716  0.75656566]
    [ 0.09603573  0.96146389  0.76502208]
    [ 0.08658009  0.97633478  0.76666667]
    '''
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (4,8), dpi=300, sharex=True)

    a = np.array([1.3, 1.8, 2.2])

    pa13 = np.array([0.0981241, 0.95728716, 0.75656566])
    pa18 = np.array([0.09603573, 0.96146389, 0.76502208])
    pa22 = np.array([0.08658009, 0.97633478, 0.76666667])

    p1 = np.array([pa13[0], pa18[0], pa22[0]])
    p2 = np.array([pa13[1], pa18[1], pa22[1]])
    p3 = np.array([pa13[2], pa18[2], pa22[2]])

    z1 = np.polyfit(a, p1, 2)
    z2 = np.polyfit(a, p2, 2)
    z3 = np.polyfit(a, p3, 2)

    print(z1)
    print(z2)
    print(z3)

    p1_pol = np.poly1d(z1)
    p2_pol = np.poly1d(z2)
    p3_pol = np.poly1d(z3)

    a_line = np.linspace(1.2, 2.4, 100)

    p1_line = p1_pol(a_line)
    p2_line = p2_pol(a_line)
    p3_line = p3_pol(a_line)

    ax1.scatter(a, p1, color='k')
    ax1.plot(a_line, p1_line, color='k')

    ax2.scatter(a, p2, color='k')
    ax2.plot(a_line, p2_line, color='k')

    ax3.scatter(a, p3, color='k')
    ax3.plot(a_line, p3_line, color='k')

    ax1.set_ylabel(r'$p_{2}$')
    ax2.set_ylabel(r'$p_{1}$')
    ax3.set_ylabel(r'$p_{0}$')
    ax3.set_xlabel(r'$a$ [AU]')

    plt.subplots_adjust(hspace=0, left=0.2, right=0.95, top=0.975, bottom=0.075)

    plt.savefig('./plots/coefficients_zero_a.png')
    plt.close()

def plot_amp_many(tab_list_list):
        
    props = {'fontsize': 11}
    colors = ['firebrick', 'gold', 'limegreen', 'royalblue', 'hotpink']
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))
    
    fig, ax = plt.subplots(figsize = (4,4), dpi=300)

    for tab_list in tab_list_list:
        c = next(color)
        amp_list, a_list = [[], []]
        for t in tab_list:
            # Get data from table name
            q = float(t.split('/')[-1].split('q_')[0])
            a = float(t.split('q_')[-1].split('a_')[0]) | units.au
            mdon = q | units.MSun
            macc = 1 | units.MSun

            # Read table and get vfr where angle of impact is 0
            df = pd.read_csv(t, index_col=0)
            P = (2 * np.pi * ((a**3) / (constants.G * (macc + mdon)))**0.5).value_in(units.hour)
            truean = df['t [hour]'] * 2 * np.pi / P

            x = df['x [RSun]'] * np.cos(truean)
            y = df['y [RSun]'] * np.cos(truean)
            r = (x**2 + y**2)**0.5
            omega = ((constants.G * (macc + mdon) / a**3)**0.5).value_in(units.s**-1)
            vx = df['vx [km s-1]'] + (-1 * df['y [RSun]'] * 695700) * omega
            vy = df['vy [km s-1]'] + (df['x [RSun]'] * 695700) * omega
            v = (vx**2 + vy**2)**0.5
            ang_imp = np.arccos(y/r) + np.arccos(vy/v) - np.pi
            df['ang imp [pi rad]'] = ang_imp

            zero = df.loc[df['ang imp [pi rad]'] >= 0].iloc[-1]
            max = df.loc[df['ang imp [pi rad]'] == df['ang imp [pi rad]'].max()]

            amp = zero['vfr'] - max['vfr']

            amp_list.append(amp.tolist()[0])
            a_list.append(a.value_in(units.au))

        label = r'$q$ = '+'{:=05.2f}'.format(q)

        amp_arr = np.array(amp_list)
        a_arr = np.array(a_list)

        # Straight line that crosses first and last point
        a_line = np.linspace(1.2, 2.3, 100)
        '''
        pq2 = (-0.02162484 * a.value_in(units.au)**2
               +0.06286028 * a.value_in(units.au)
               +0.05295173)
        pq1 = (0.03202641 * a.value_in(units.au)**2
               -0.0909287 * a.value_in(units.au)
               +1.02136945)
        pq0 = (-0.01422374 * a.value_in(units.au)**2
               +0.06100643 * a.value_in(units.au)
               +0.70129542)
        vfr_line = (pq2 * q_line**2
                    + pq1 * q_line
                    + pq0)
        '''
        z = np.polyfit(a_arr, amp_arr, 2)
        print(z)
        amp_pol = np.poly1d(z)
        amp_line = amp_pol(a_line)

        ax.plot(a_line, amp_line, color=c)
        ax.scatter(a_arr, amp_arr, s=25, color=c, label=label, marker='o', facecolor='none', zorder=3)

    ax.set_xlabel(r'$a$ [AU]', **props)
    ax.set_ylabel(r'$A$', **props)
    
    ax.legend(loc=2)
    
    plt.subplots_adjust(left=0.175, right=0.95, top=0.95, bottom=0.125)

    #plt.show()
    plt.savefig('./plots/coefficients_amp_fit.png')
    plt.close()

def coefficients_amp(tab_list_list):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (4,8), dpi=300, sharex=True)

    props = {'fontsize': 11}
    
    p2_list, p1_list, p0_list = [[], [], []]

    q_list = []
    for tab_list in tab_list_list:
        amp_list, a_list = [[], []]
        for t in tab_list:
            # Get data from table name
            q = float(t.split('/')[-1].split('q_')[0])
            a = float(t.split('q_')[-1].split('a_')[0]) | units.au
            mdon = q | units.MSun
            macc = 1 | units.MSun

            # Read table and get vfr where angle of impact is 0
            df = pd.read_csv(t, index_col=0)
            P = (2 * np.pi * ((a**3) / (constants.G * (macc + mdon)))**0.5).value_in(units.hour)
            truean = df['t [hour]'] * 2 * np.pi / P

            x = df['x [RSun]'] * np.cos(truean)
            y = df['y [RSun]'] * np.cos(truean)
            r = (x**2 + y**2)**0.5
            omega = ((constants.G * (macc + mdon) / a**3)**0.5).value_in(units.s**-1)
            vx = df['vx [km s-1]'] + (-1 * df['y [RSun]'] * 695700) * omega
            vy = df['vy [km s-1]'] + (df['x [RSun]'] * 695700) * omega
            v = (vx**2 + vy**2)**0.5
            ang_imp = np.arccos(y/r) + np.arccos(vy/v) - np.pi
            df['ang imp [pi rad]'] = ang_imp

            mask = [not x for x in df['ang imp [pi rad]'].isnull()]
            df = df.iloc[mask]

            zero = df.loc[df['ang imp [pi rad]'] >= 0].iloc[-1]
            max = df.loc[df['ang imp [pi rad]'] == df['ang imp [pi rad]'].max()]

            amp = zero['vfr'] - max['vfr']

            amp_list.append(amp)
            a_list.append(a.value_in(units.au))

        z = np.polyfit(a_list, amp_list, 2)

        q_list.append(q)
        #print(q)
        p2_list.append(z[0].tolist()[0])
        p1_list.append(z[1].tolist()[0])
        p0_list.append(z[2].tolist()[0])
    
    amp_arr = np.array(amp_list)
    a_arr = np.array(a_list)
    q_arr = np.array(q_list)

    p2 = np.array(p2_list)
    p1 = np.array(p1_list)
    p0 = np.array(p0_list)
    
    # Get polynome fittyng of all data except q=1
    z2 = np.polyfit([q_list[0], q_list[1], q_list[2], q_list[3], q_list[4]],
                    [p2_list[0], p2_list[1], p2_list[2], p2_list[3], p2_list[4]],
                    1)
    z1 = np.polyfit([q_list[0], q_list[1], q_list[2], q_list[3], q_list[4]],
                    [p1_list[0], p1_list[1], p1_list[2], p1_list[3], p1_list[4]],
                    1)
    z0 = np.polyfit([q_list[0], q_list[1], q_list[2], q_list[3], q_list[4]],
                    [p0_list[0], p0_list[1], p0_list[2], p0_list[3], p0_list[4]],
                    1)
    print(z2, z1, z0)
    '''
    z2 = np.polyfit(q_arr, p2, 4)
    z1 = np.polyfit(q_arr, p1, 4)
    z0 = np.polyfit(q_arr, p0, 4)
    '''
    p2_pol = np.poly1d(z2)
    p1_pol = np.poly1d(z1)
    p0_pol = np.poly1d(z0)

    q_line = np.linspace(0.45, 1.55, 101)

    p2_line = p2_pol(q_line)
    p1_line = p1_pol(q_line)
    p0_line = p0_pol(q_line)

    ax1.scatter(q_arr, p2, color='k')
    ax1.plot(q_line, p2_line, color='k')

    ax2.scatter(q_arr, p1, color='k')
    ax2.plot(q_line, p1_line, color='k')

    ax3.scatter(q_arr, p0, color='k')
    ax3.plot(q_line, p0_line, color='k')

    ax1.set_ylabel(r'$p_{2}$')
    ax2.set_ylabel(r'$p_{1}$')
    ax3.set_ylabel(r'$p_{0}$')
    ax3.set_xlabel(r'$q$')

    plt.subplots_adjust(hspace=0, left=0.2, right=0.95, top=0.975, bottom=0.075)

    plt.savefig('./plots/coefficients_amp_q.png')
    plt.close()

def correct_table(two_df_lists_list):
    n = int(len(two_df_lists_list) * len(two_df_lists_list[0]))
    data = {'a [AU]' : np.zeros(n),
            'q' : np.zeros(n),
            'v zero' : np.zeros(n),
            'v Max' : np.zeros(n),
            'A' : np.zeros(n)}
    df = pd.DataFrame(data=data)

    j = 0
    for two_df_list in two_df_lists_list:
        #a = float(two_df_list[0].split('q_')[1].split('a_')[0])
        for i in range(0, len(two_df_list)):
            a = float(two_df_list[i].split('q_')[1].split('a_')[0])
            q = float(two_df_list[i].split('vfr_')[1].split('q_')[0])
            two_files = pd.read_table(two_df_list[i], header=None, names=['filenames'])
            vfr_list, v_list, ang_list = [[], [], []]
            for filename in two_files['filenames']:
                two_df = pd.read_csv('./data/'+two_df_list[i].split('.dat')[0]+'/'+filename+'.csv')
                vfr_list.append(float((filename.split('e_')[1]).split('vfr_')[0]))
                if two_df.iloc[0]['flag impact'] == 1:
                    v_list.append(two_df.iloc[0]['v imp [km s-1]'])
                    ang_list.append(two_df.iloc[0]['ang imp [rad]'] / np.pi)
                else:
                    v_list.append(np.nan)
                    ang_list.append(np.nan)
            
            data2 = {'vfr' : vfr_list, 'v imp [km s-1]' : v_list, 'ang imp [pi rad]' : ang_list}
            df2 = pd.DataFrame(data=data2)
            mask2 = [not x for x in df2['ang imp [pi rad]'].isnull()]
            df2 = df2.iloc[mask2]
        
            zero2 = df2.loc[df2['ang imp [pi rad]'] >= 0].iloc[-1]
            max2 = df2.loc[df2['ang imp [pi rad]'] == df2['ang imp [pi rad]'].max()]

            df.loc[j, 'a [AU]'] = a
            df.loc[j, 'q'] = q
            df.loc[j, 'v zero'] = zero2['vfr']
            df.loc[j, 'v Max'] = max2['vfr'].values[0]
            df.loc[j, 'A'] = zero2['vfr'] - max2['vfr'].values[0]

            j += 1
        
    print(df)

    df.to_csv('./data/correction_table_plots.csv')

def plot_amp_many_model(tablename):
    df = pd.read_csv(tablename)
    a_arr = df['a [AU]'].unique()
    q_arr = df['q'].unique()

    props = {'fontsize': 11}
    colors = ['firebrick', 'gold', 'limegreen', 'royalblue', 'hotpink']
    cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    color = iter(cmap(np.linspace(0, 1, 5)))
    
    fig, ax = plt.subplots(figsize = (4,4), dpi=300)

    a_line = np.linspace(1.2, 2.3, 100)
    for i in range(len(q_arr)):
        c = next(color)
        df_slice = df.loc[df['q'] == q_arr[i]]
        
        z = np.polyfit(df_slice['a [AU]'], df_slice['A'], 2)
        print(z)
        amp_pol = np.poly1d(z)
        amp_line = amp_pol(a_line)

        label = r'$q$ = '+'{:=05.2f}'.format(q_arr[i])

        if (q_arr[i] > 0.5) & (q_arr[i] <= 1.0):
            linestyle = 'solid'
            facecolor = c
        else:
            linestyle = 'dashed'
            facecolor = 'none'
        ax.plot(a_line, amp_line, color=c, linestyle=linestyle)
        ax.scatter(a_arr, df_slice['A'], s=25, color=c, label=label, marker='o', facecolor=facecolor, zorder=3)

    ax.set_xlabel(r'$a$ [AU]', **props)
    ax.set_ylabel(r'$A$', **props)
    
    ax.legend(loc=1)
    
    plt.subplots_adjust(left=0.175, right=0.95, top=0.95, bottom=0.125)

    plt.savefig('./plots/coefficients_amp_fit_model.png')
    plt.close()

def coefficients_amp_model(tablename):
    df = pd.read_csv(tablename)
    a_arr = df['a [AU]'].unique()
    q_arr = df['q'].unique()

    props = {'fontsize': 11}

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (4,8), dpi=300, sharex=True)

    a_line = np.linspace(1.2, 2.3, 100)
    p2_list, p1_list, p0_list = [[], [], []]
    for i in range(len(q_arr)):
        df_slice = df.loc[df['q'] == q_arr[i]]
        
        z = np.polyfit(df_slice['a [AU]'], df_slice['A'], 2)

        p2_list.append(z[0])
        p1_list.append(z[1])
        p0_list.append(z[2])

    p2 = np.array(p2_list)
    p1 = np.array(p1_list)
    p0 = np.array(p0_list)

    z2 = np.polyfit(q_arr, p2, 2)
    z1 = np.polyfit(q_arr, p1, 2) 
    z0 = np.polyfit(q_arr, p0, 2)

    print(z2, z1, z0)

    p2_pol = np.poly1d(z2)
    p1_pol = np.poly1d(z1)
    p0_pol = np.poly1d(z0)

    q_line = np.linspace(0.45, 1.55, 101)

    p2_line = p2_pol(q_line)
    p1_line = p1_pol(q_line)
    p0_line = p0_pol(q_line)

    ax1.scatter(q_arr, p2, color='k')
    ax1.plot(q_line, p2_line, color='k')

    ax2.scatter(q_arr, p1, color='k')
    ax2.plot(q_line, p1_line, color='k')

    ax3.scatter(q_arr, p0, color='k')
    ax3.plot(q_line, p0_line, color='k')

    ax1.set_ylabel(r'$p_{2}$')
    ax2.set_ylabel(r'$p_{1}$')
    ax3.set_ylabel(r'$p_{0}$')
    ax3.set_xlabel(r'$q$')

    plt.subplots_adjust(hspace=0, left=0.2, right=0.95, top=0.975, bottom=0.075)

    plt.savefig('./plots/coefficients_amp_q_model.png')
    plt.close()

    
if __name__ == "__main__":
    #max_e('e_001_800a_0_80vfr_000_00vexp.dat')
    #show_test('./data/evolution_table_2.csv')
    #coef('./data/1.200q_02.200a_2body.csv', './data/1.200q_02.200a_3body.csv')
    #plot_zeros(['./data/00.50q_001.80000a_3body_rot.csv', './data/00.75q_001.80000a_3body_rot.csv', './data/01.00q_001.80000a_3body_rot.csv', './data/01.25q_001.80000a_3body_rot.csv', './data/01.50q_001.80000a_3body_rot.csv'])
    #plot_zeros_many([['./data/00.50q_001.30000a_3body_rot.csv', './data/00.75q_001.30000a_3body_rot.csv', './data/01.00q_001.30000a_3body_rot.csv', './data/01.25q_001.30000a_3body_rot.csv', './data/01.50q_001.30000a_3body_rot.csv'], ['./data/00.50q_001.80000a_3body_rot.csv', './data/00.75q_001.80000a_3body_rot.csv', './data/01.00q_001.80000a_3body_rot.csv', './data/01.25q_001.80000a_3body_rot.csv', './data/01.50q_001.80000a_3body_rot.csv'], ['./data/00.50q_002.20000a_3body_rot.csv', './data/00.75q_002.20000a_3body_rot.csv', './data/01.00q_002.20000a_3body_rot.csv', './data/01.25q_002.20000a_3body_rot.csv', './data/01.50q_002.20000a_3body_rot.csv']])
    #coefficients_zeros()
    #coefficients_amp([['./data/00.50q_001.30000a_3body_rot.csv', './data/00.50q_001.80000a_3body_rot.csv', './data/00.50q_002.20000a_3body_rot.csv'], ['./data/00.75q_001.30000a_3body_rot.csv', './data/00.75q_001.80000a_3body_rot.csv', './data/00.75q_002.20000a_3body_rot.csv'], ['./data/01.00q_001.30000a_3body_rot.csv', './data/01.00q_001.80000a_3body_rot.csv', './data/01.00q_002.20000a_3body_rot.csv'], ['./data/01.25q_001.30000a_3body_rot.csv', './data/01.25q_001.80000a_3body_rot.csv', './data/01.25q_002.20000a_3body_rot.csv'], ['./data/01.50q_001.30000a_3body_rot.csv', './data/01.50q_001.80000a_3body_rot.csv', './data/01.50q_002.20000a_3body_rot.csv']])
    #plot_amp_many([['./data/00.50q_001.30000a_3body_rot.csv', './data/00.50q_001.80000a_3body_rot.csv', './data/00.50q_002.20000a_3body_rot.csv'], ['./data/00.75q_001.30000a_3body_rot.csv', './data/00.75q_001.80000a_3body_rot.csv', './data/00.75q_002.20000a_3body_rot.csv'], ['./data/01.00q_001.30000a_3body_rot.csv', './data/01.00q_001.80000a_3body_rot.csv', './data/01.00q_002.20000a_3body_rot.csv'], ['./data/01.25q_001.30000a_3body_rot.csv', './data/01.25q_001.80000a_3body_rot.csv', './data/01.25q_002.20000a_3body_rot.csv'], ['./data/01.50q_001.30000a_3body_rot.csv', './data/01.50q_001.80000a_3body_rot.csv', './data/01.50q_002.20000a_3body_rot.csv']])
    #correct_table([['vfr_00.50q_001.300a_00.0000e_000.00vexp.dat', 'vfr_00.75q_001.300a_00.0000e_000.00vexp.dat', 'vfr_01.00q_001.300a_00.0000e_000.00vexp.dat', 'vfr_01.25q_001.300a_00.0000e_000.00vexp.dat', 'vfr_01.50q_001.300a_00.0000e_000.00vexp.dat'],
    #              ['vfr_00.50q_001.800a_00.0000e_000.00vexp.dat', 'vfr_00.75q_001.800a_00.0000e_000.00vexp.dat', 'vfr_01.00q_001.800a_00.0000e_000.00vexp.dat', 'vfr_01.25q_001.800a_00.0000e_000.00vexp.dat', 'vfr_01.50q_001.800a_00.0000e_000.00vexp.dat'],
    #              ['vfr_00.50q_002.200a_00.0000e_000.00vexp.dat', 'vfr_00.75q_002.200a_00.0000e_000.00vexp.dat', 'vfr_01.00q_002.200a_00.0000e_000.00vexp.dat', 'vfr_01.25q_002.200a_00.0000e_000.00vexp.dat', 'vfr_01.50q_002.200a_00.0000e_000.00vexp.dat']])
    #plot_amp_many_model('./data/correction_table.csv')
    #coefficients_amp_model('./data/correction_table.csv')
    correct_table([['vfr_01.20q_001.300a_00.0000e_000.00vexp.dat', 'vfr_01.20q_001.800a_00.0000e_000.00vexp.dat', 'vfr_01.20q_002.200a_00.0000e_000.00vexp.dat']])