#!/bin/python

from __future__ import print_function
import sys
import argparse
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

assert sys.version_info >= (2,7)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=argparse.FileType('r'), help='unit_scale.dat input file')
    parser.add_argument('-t', '--title', nargs='?', type=str, default=None, help='plot title string')
    parser.add_argument('-o', '--outfile', nargs='?', type=argparse.FileType('w'), default=None, help='output image file')
    parser.add_argument('-b', '--batch', action='store_true', help='batch mode (no interactive plots)')
    cmdline = parser.parse_args()

    values = []
    for line in cmdline.infile:
        fields = line.split()
        values += [float(v) for v in fields[3:]]

    q1, q3 = np.percentile(values, [25, 75])
    iqr = abs(q3 - q1)
    print('q1 =', q1)
    print('q3 =', q3)
    print('IQR =', iqr)
    slimmed = [x for x in values if x >= q1 - 1.5*iqr and x <= q3 + 1.5*iqr]

    print()
    print('Before removing outliers: ({} values)'.format(len(values)))
    print('mean =', np.mean(values))
    print('median =', np.median(values))
    print('stddev =', np.std(values))
    print()
    print('After removing outliers: ({} values)'.format(len(slimmed)))
    print('mean =', np.mean(slimmed))
    print('median =', np.median(slimmed))
    print('stddev =', np.std(slimmed))
    print()
    print('Slimmed', len(values) - len(slimmed), 'values')

    mean = np.mean(slimmed)
    stddev = np.std(slimmed)
    n, bins, _ = plt.hist(values, bins=np.arange(0.5,1.5,1.0/1000), normed=True, alpha=0.6, label='Observed')
    gauss = mlab.normpdf(bins, mean, stddev)
    plt.plot(bins, gauss, 'k-', linewidth=1, label=('Gaussian\n$\mu$ = ' + ("%.6f" % mean) + '\n$\sigma$ = ' + ("%.6f" %stddev)))
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    plt.legend(loc='upper right')
    plt.xlim([0.5,1.5])
    plt.axvline(x=mean, linestyle='dashed', c='k', linewidth=1, label='$\mu$ = ' + ("%.6f" % mean))
    if cmdline.title is not None:
        plt.title(cmdline.title)
    if cmdline.outfile is not None:
        plt.savefig(cmdline.outfile, bbox_inches='tight')
    if not cmdline.batch:
        plt.show()

if __name__=='__main__':
    main()
#EOF
