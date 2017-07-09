#!/usr/bin/python

from __future__ import print_function
import sys
import argparse
import math
import numpy as np
import matplotlib as mpl
parser = argparse.ArgumentParser()
parser.add_argument('infile', type=argparse.FileType('r'), nargs='+', help='unit_scale.dat input file')
parser.add_argument('-t', '--title', nargs='?', type=str, default=None, help='plot title string')
parser.add_argument('-o', '--outfile', nargs='?', type=argparse.FileType('w'), default=None, help='output image file')
parser.add_argument('-b', '--batch', action='store_true', help='batch mode (no interactive plots)')
parser.add_argument('--bins', type=int, default=100, help='number of histogram bins')
cmdline = parser.parse_args()
if cmdline.batch:
    mpl.use('agg')
import matplotlib.mlab as mlab
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

assert sys.version_info >= (2,7)

threshold1 = 0.9
threshold2 = 0.99

colors = ['m', 'b', 'c']
labels = ['correlation $\geq ' + str(threshold2) + '$', \
          '$' + str(threshold1) + '\leq$ correlation $<' + str(threshold2) + '$', \
          'correlation $< '+ str(threshold1) + '$']

def read_infile(infile):
    global threshold1, threshold2

    selected1_slopes = []
    selected1_intercepts = []
    selected2_slopes = []
    selected2_intercepts = []
    other_slopes = []
    other_intercepts = []
    fit = {}
    for line in infile:
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

    print('Regressions included:', len(other_slopes) + len(selected1_slopes) + len(selected2_slopes))
    return (selected2_slopes, selected1_slopes, other_slopes), (selected2_intercepts, selected1_intercepts, other_intercepts)

def plot_infile(infile, axl, axr, num_bins=100):
    global colors, labels
    lvalues, rvalues = read_infile(infile)

    slope_range = [-20, 80]
    intercept_range = [-600, 1800]

    # Plot histograms
    axl.hist(lvalues, bins=np.linspace(*(slope_range + [num_bins])), stacked=True, normed=True, color=colors, label=labels, alpha=0.7)
    axr.hist(rvalues, bins=np.linspace(*(intercept_range + [num_bins])), stacked=True, normed=True, color=colors, label=labels, alpha=0.7)

    # Remove y-ticks
    axl.yaxis.set_major_locator(plt.NullLocator())
    axr.yaxis.set_major_locator(plt.NullLocator())

    # Explicit x-range
    axl.set_xlim(slope_range)
    axr.set_xlim(intercept_range)

    run_name = infile.name[:-4] if infile.name.endswith('.dat') else infile.name
    axl.set_ylabel(run_name + ' ({} hosts)'.format(sum([len(s) for s in lvalues])))

    # axl.axvline(x=0, linestyle='solid', c='#aaaaaa', linewidth=1)
    # axr.axvline(x=0, linestyle='solid', c='#aaaaaa', linewidth=1)

def main():
    global cmdline, colors, labels

    num_files = len(cmdline.infile)
    fig, matrix = plt.subplots(num_files, 2)
    if num_files == 1:
        matrix = [matrix]
    fig.set_size_inches(9, 2 + 2*num_files)
    fig.subplots_adjust(wspace=0.05, hspace=0.06)

    if cmdline.title is not None:
        plt.suptitle(cmdline.title)
    matrix[0][0].set_title('Slope')
    matrix[0][1].set_title('Intercept')

    for row in xrange(num_files):
        left = matrix[row][0]
        right = matrix[row][1]
        infile = cmdline.infile[row]
        plot_infile(infile, left, right, cmdline.bins)
        if row == num_files - 1:
            left.set_xlabel('Slope (s/unit)')
            right.set_xlabel('Intercept (s)')
        else:
            # Not the last row
            left.xaxis.set_ticklabels([])
            right.xaxis.set_ticklabels([])

    legend_handles = [mpatches.Patch(color=colors[i]) for i in reversed(range(len(labels)))]
    legend_labels = [lbl for lbl in reversed(labels)]
    plt.figlegend(legend_handles, legend_labels, loc='lower center', ncol=len(labels))

    if cmdline.outfile is not None:
        plt.savefig(cmdline.outfile)

    if not cmdline.batch:
        plt.show()

if __name__=='__main__':
    main()
#EOF
