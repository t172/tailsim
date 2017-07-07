#!/usr/bin/python

from __future__ import print_function
import sys
import argparse
import math
import numpy as np
import matplotlib as mpl
from scipy.optimize import curve_fit
from scipy.misc import factorial

assert sys.version_info >= (2,7)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=argparse.FileType('r'), help='unit_scale.dat input file')
    parser.add_argument('-t', '--title', nargs='?', type=str, default=None, help='plot title string')
    parser.add_argument('-o', '--outfile', nargs='?', type=argparse.FileType('w'), default=None, help='output image file')
    parser.add_argument('-b', '--batch', action='store_true', help='batch mode (no interactive plots)')
    cmdline = parser.parse_args()

    if cmdline.batch:
        mpl.use('agg')
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt

    threshold1 = 0.9
    threshold2 = 0.99

    selected1_slopes = []
    selected1_intercepts = []
    selected2_slopes = []
    selected2_intercepts = []
    other_slopes = []
    other_intercepts = []
    fit = {}
    for line in cmdline.infile:
        fields = line.split()
        host = fields[0]
        correlation = float(fields[2])
        slope = float(fields[3])
        intercept = float(fields[4])

        if math.isnan(slope):
            continue

        if correlation >= threshold2:
            selected2_slopes.append(slope)
            selected2_intercepts.append(intercept)
        elif correlation >= threshold1:
            selected1_slopes.append(slope)
            selected1_intercepts.append(intercept)
        else:
            other_slopes.append(slope)
            other_intercepts.append(intercept)

            # if host not in fit:
            #     fit[host] = []
            # fit[host].append((correlation, slope, intercept))

    print('Regressions included:', len(other_slopes) + len(selected1_slopes) + len(selected2_slopes))

    # for host in sorted(fit.keys()):
    #     if len(fit[host]) <= 1:
    #         continue
    #     print(host, end='')
    #     for line in fit[host]:
    #         print(' ', line[1], end='')
    #     print()

    fig, (axl, axr) = plt.subplots(1, 2)
    fig.set_size_inches(9,4)
    fig.subplots_adjust(wspace=0.05)

    if cmdline.title is not None:
        plt.suptitle(cmdline.title)
    axl.set_title('Slope')
    axr.set_title('Intercept')

    # lvalues = [other_slopes, selected1_slopes, selected2_slopes]
    # rvalues = [other_intercepts, selected1_intercepts, selected2_intercepts]
    # colors = ['c', 'b', 'm']
    # labels = ['correlation $< '+ str(threshold1) + '$', \
    #           '$' + str(threshold1) + '\leq$ correlation $<' + str(threshold2) + '$', \
    #           'correlation $\geq ' + str(threshold2) + '$']
    lvalues = [selected2_slopes, selected1_slopes, other_slopes]
    rvalues = [selected2_intercepts, selected1_intercepts, other_intercepts]
    colors = ['m', 'b', 'c']
    labels = ['correlation $\geq ' + str(threshold2) + '$', \
              '$' + str(threshold1) + '\leq$ correlation $<' + str(threshold2) + '$', \
              'correlation $< '+ str(threshold1) + '$']

    slope_range = [-20, 80]
    intercept_range = [-600, 1800]
    num_bins = 100

    axl.hist(lvalues, bins=np.linspace(*(slope_range + [num_bins])), stacked=True, normed=True, color=colors, label=labels, alpha=0.7)
    axr.hist(rvalues, bins=np.linspace(*(intercept_range + [num_bins])), stacked=True, normed=True, color=colors, label=labels, alpha=0.7)

    # means = [2, 9, 18]
    # for i in xrange(len(means)):
    #     axl.axvline(x=means[i], linestyle='dashed', c='k', linewidth=1, label=('$\mu$ = ' + str(means) if i == 0 else ''))

    # stddevs = [1.5, 2, 3]
    # for i in xrange(len(means)):
    #     gauss = mlab.normpdf(bins, means[i], stddevs[i])
    #     plt.plot(bins, gauss, 'k-', linewidth=1, label='m = ' + str(means[i]) + ', s = ' + str(stddevs[i]))
    
    axl.set_xlabel('Slope (s/unit)')
    axl.set_ylabel('Probability Density')
    axl.yaxis.set_major_locator(plt.NullLocator())
    axl.set_xlim(slope_range)

    axr.yaxis.set_major_locator(plt.NullLocator())
    axr.set_xlabel('Intercept (s)')
    #axr.set_ylabel('Probability Density')
    axr.set_xlim(intercept_range)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=2)
    #axl.legend(loc='upper right')

    if cmdline.outfile is not None:
        plt.savefig(cmdline.outfile)

    if not cmdline.batch:
        plt.show()

if __name__=='__main__':
    main()
#EOF
