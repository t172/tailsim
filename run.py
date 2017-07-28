#!/bin/env python

from __future__ import print_function

import sys, os, errno
import argparse
import threading
import numpy as np
from random import random
from tailsim import *

assert sys.version_info >= (2,7)

num_units = 2000000
num_hosts = 1000
host_capacity = 3

runtimes = [LinearRuntime( 60.0*Time.second, 200.0*Time.second)]*(num_hosts // 3) + \
           [LinearRuntime(110.0*Time.second, 250.0*Time.second)]*(num_hosts // 3 + num_hosts % 3) + \
           [LinearRuntime(180.0*Time.second, 300.0*Time.second)]*(num_hosts // 3)

line_color = '#404080ff'
fill_color = '#c0c8f7ff'

class TestMaster(Master):
    def start(self):
        super(TestMaster, self).start()
        self.schedule_event(self.now + 0.5*Time.hour, self.assign_available)
        self.schedule_event(self.now, self.measure)

    def measure(self):
        data = getattr(self, 'data', [])
        data.append((self.now, sum(host.running() for host in self.hosts), sum(host.available() for host in self.hosts), self.units_remaining()))
        self.data = data

        if len(self.sim._schedule) > 0:
            self.schedule_event(self.now + 2*Time.minute, self.measure)

class SimThread(threading.Thread):
    def __init__(self, permit, bitesize, mtbe, selection='random'):
        super(SimThread, self).__init__()
        self.permit = permit
        self.bitesize = bitesize
        self.mtbe = mtbe
        self.selection = selection

    def run(self):
        try:
            self.sim, self.master = setup_sim(bitesize=self.bitesize, mtbe=self.mtbe, selection='select_' + self.selection)
            self.sim.start()
            store_result((self.bitesize, Time.to_hour(self.sim.now), sum(host.evicted_tasks for host in self.master.hosts), self.selection))
        finally:
            self.permit.release()

def read_output(infile):
    """Load existing output file from previous run"""
    global results
    results = []
    for line in infile:
        if line[0] == '#':
            continue
        fields = line.split()
        results.append((int(fields[0]), float(fields[1]), int(fields[2]), fields[3]))

results = []
lock = threading.Lock()
def store_result(result):
    global results, lock, cmdline
    outfmt = ' '.join((str(r) for r in result))
    with lock:
        results.append(result)
        if cmdline.outfile is not None:
            print(outfmt, file=cmdline.outfile)
        if cmdline.logfile is not None:
            print(outfmt, file=cmdline.logfile)

def read_evictions(infile, logfile=None):
    global num_hosts
    if infile is not None:
        if logfile is not None:
            print('Sampling MTBE from', infile.name, file=logfile)
        evictions = []
        for line in infile:
            fields = line.split()
            evictions.append(None if fields[3] == 'inf' else int(float(fields[3])*Time.hour))
        mtbe = random.sample(evictions, num_hosts)
    else:
        poisson_lambda_hr = 4
        if logfile is not None:
            print(u'Using random MTBE from Poisson \u03BB =', poisson_lambda_hr, 'hr', file=logfile)
        mtbe = [int(x) for x in np.random.poisson(lam=poisson_lambda_hr*Time.hour, size=num_hosts)]
    return mtbe

def setup_sim(bitesize, mtbe, selection='select_random'):
    global num_units, num_hosts, host_capacity, runtimes, cmdline
    sim = Simulation(None)
    units = [WorkUnit(np.random.normal(loc=1.0, scale=0.055)) for n in xrange(num_units)]
    hosts = [Host(str(n), host_capacity, runtimes[n], mtbe[n], restart_delay=np.random.poisson(lam=20*Time.minute)) for n in xrange(num_hosts)]
    master = TestMaster(units, hosts)
    master.bitesize = bitesize
    master.select_host = getattr(master, selection, master.select_host)

    sim.add_actor(master)
    for host in hosts:
        sim.add_actor(host)
    return sim, master

def process_cmdline():
    # Process command line
    parser = argparse.ArgumentParser()
    output_options = parser.add_mutually_exclusive_group(required=False)
    parser.add_argument('-v', '--verbose', action='store_true', help='more informative output')
    parser.add_argument('--eviction', nargs='?', type=argparse.FileType('r'), help='use eviction report for eviction histogram')
    parser.add_argument('-b', '--batch', action='store_true', help='batch mode (no interactive plots)')
    parser.add_argument('--bitesize', nargs='*', type=int, default=1, help='task size')
    parser.add_argument('-d', '--dump', nargs='?', type=argparse.FileType('w'), default=None, help='dump data to file')
    parser.add_argument('--title', nargs='?', type=str, default=None, help='plot title string')
    parser.add_argument('-o', '--outfile', nargs='?', type=argparse.FileType('w'), default=None, help='data output file')
    parser.add_argument('-l', '--logfile', nargs='?', type=argparse.FileType('w'), default=sys.stderr, help='simulation log file')
    parser.add_argument('-t', '--trials', nargs='?', type=int, default=1, help='number of trials (for each bitesize)')
    parser.add_argument('--frames', nargs='?', type=int, default=1, help='number of animation frames')
    parser.add_argument('-j', '--threads', nargs='?', type=int, default=1, help='number of processing threads to use')
    parser.add_argument('--font', nargs='?', type=str, default=None, help='font to use for some plots')
    parser.add_argument('--tail', action='store_true', help='plot tail instead')
    parser.add_argument('--style', action='store', type=int, choices=range(5), help='plotting style')
    parser.add_argument('--outdir', nargs='?', type=str, default='.', help='destination directory for frames')
    parser.add_argument('-p', '--plot', nargs='?', type=argparse.FileType('r'), default=None, help='plot existing output file')
    return parser.parse_args()

def plot_tail(infile):
    """Plots a dumped tail"""

    # Read file
    xs, ys = {}, {}
    for line in infile:
        fields = line.split()
        run = int(fields[0])
        time = float(fields[1])
        running = int(fields[2])
        if run not in xs:
            xs[run] = []
            ys[run] = []
        xs[run].append(time)
        ys[run].append(running)
    min_time = min(min(r) for r in xs.values())
    max_time = max(max(r) for r in xs.values())
    max_y = max(max(r) for r in ys.values())

    # Create output directory
    if cmdline.outdir:
        try: os.makedirs(cmdline.outdir)
        except OSError as e:
            if e.errno != errno.EEXIST: raise

    # Plot it
    frame_width = 1920
    frame_height = 1080
    dpi = 96.0
    axis_width = 2
    mpl.rcParams['axes.linewidth'] = axis_width
    using_gui = (cmdline.frames == 1 and not cmdline.batch)
    timestep = (max_time - min_time)/float(cmdline.frames)
    shown_time = min_time
    for frame in xrange(1, cmdline.frames + 1):
        print('frame {}/{}'.format(frame, cmdline.frames), file=sys.stderr)
        shown_time += timestep
        #fig = plt.figure()
        fig, ax = plt.subplots()
        fig.set_size_inches(frame_width/float(dpi), frame_height/float(dpi))
        #ax = plt.Axes(fig, [0, 0, 1, 1])
        #fig.add_axes(ax)
        #ax.set_axis_off()
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.get_xaxis().set_ticks([min_time, max_time])
        ax.set_xticklabels(['', ''])
        ax.xaxis.set_tick_params(width=axis_width)
        ax.get_yaxis().set_ticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_bounds(min_time, max_time)
        plt.xlim([min_time, max_time])
        plt.ylim([-0.03*max_y, 1.05*max_y])
        #plt.axis('off')
        plt.tick_params(left='off', bottom='on')
        fontprop = fm.FontProperties(fname=cmdline.font) if cmdline.font else None
        plt.xlabel('Time', labelpad=20, fontsize=36, fontproperties=fontprop)
        plt.ylabel('Tasks Running', labelpad=40, fontsize=36, fontproperties=fontprop)

        global line_color
        for run in xs:
            times = [x for x in xs[run] if x <= shown_time]
            plt.plot(times, ys[run][:len(times)], color=line_color, lw=6, label='Run {}'.format(run))
        if len(xs) > 1:
            plt.legend(loc='upper right')
        if cmdline.frames > 1:
            outfile = cmdline.outdir + ('/' if cmdline.outdir and cmdline.outdir != '' else '') + 'frame-{}.png'.format(frame)
            plt.savefig(outfile, format='png', dpi=dpi)
        elif cmdline.outfile:
            plt.savefig(cmdline.outfile, format='png', bbox_inches='tight', dpi=dpi)
        if not using_gui:
            plt.close(plt.gcf())
    if using_gui:
        plt.show()

def plot_results():
    global results, cmdline
    results = sorted(results)

    # Separate results by selection
    by_selection = {}
    for row in results:
        selection = row[3]
        if selection not in by_selection:
            by_selection[selection] = []
        by_selection[selection].append(row)

    # Plot
    axis_width = 2
    frame_width = 1920
    frame_height = 1080
    dpi = 96.0
    mpl.rcParams['axes.linewidth'] = axis_width
    global line_color, fill_color
    fig, ax = plt.subplots()
    fig.set_size_inches(frame_width/float(dpi), frame_height/float(dpi))
    #fig.set_size_inches(9, 4)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(width=axis_width)
    #ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}'.format(y) if int(y)==y else '{:.1f}'.format(y)))
    min_time = float('inf')
    max_time = 0
    xticks, yticks = [], []
    xtlbls, ytlbls = [], []
    for selection in by_selection:
        this_selection = by_selection[selection]
        bites = {}
        for value in this_selection:
            if value[0] not in bites:
                bites[value[0]] = []
            bites[value[0]].append(value[1]/24.0)  # convert hours to days
        x = sorted(bites.keys())
        means = [np.mean(bites[v]) for v in x]
        mins = [min(bites[x[i]]) for i in xrange(len(x))]
        maxs = [max(bites[x[i]]) for i in xrange(len(x))]
        min_time = min(min(mins), min_time)
        max_time = max(max(maxs), max_time)
        #stds = [np.std(bites[v]) for v in x]
        #lows = [abs(means[i] - min(bites[x[i]])) for i in xrange(len(x))]
        #highs = [abs(max(bites[x[i]]) - means[i]) for i in xrange(len(x))]
        if cmdline.style == 0:
            pass
        elif cmdline.style == 1:
            for v in x:
                plt.scatter([v]*len(bites[v]), bites[v], marker='.', color=line_color, alpha=0.5, label=selection)
                # xticks += [v]
                # xtlbls += ['']
                # yticks += bites[v]
                # ytlbls += ['']*len(bites[v])
        elif cmdline.style == 2:
            shaded = '#aaaaaa'
            plt.plot(x, means, '-o', color=line_color, alpha=1.0, label=selection)
            ax.spines['bottom'].set_color(shaded)
            ax.spines['left'].set_color(shaded)
            ax.tick_params(axis='both', colors=shaded)
        elif cmdline.style == 3:
            shaded = '#ffffff'
            plt.fill_between(x, mins, maxs, color=fill_color, alpha=0.5)
            plt.plot(x, means, '-', lw=3, solid_capstyle='round', c=line_color, alpha=1.0, label=selection)
            # ax.spines['bottom'].set_visible(False)
            # ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color(shaded)
            ax.spines['left'].set_color(shaded)
            ax.tick_params(axis='both', colors=shaded)
        elif cmdline.style == 4:
            plt.fill_between(x, mins, maxs, color=fill_color, alpha=0.5)
            plt.plot(x, means, '-', lw=3, solid_capstyle='round', c=line_color, alpha=1.0, label=selection)
            added_xticks = [x[i] for i in xrange(len(x)) if means[i] == min(means)]
            xticks += added_xticks
            xtlbls += ['Minimum']*len(added_xticks)

    min_x = min(bites.keys())
    max_x = max(bites.keys())
    range_x = max_x - min_x
    ax.tick_params(width=axis_width, pad=10)

    plt.xlim([min_x - 0.05*range_x, max_x + 0.05*range_x])
    # plt.xticks(xticks + ([min_x, max_x] if cmdline.style < 3 else []))
    # ax.set_xticklabels(xtlbls + (['', ''] if cmdline.style < 3 else []))
    plt.xticks(xticks + [min_x, max_x])
    ax.set_xticklabels(xtlbls + ['', ''])
    ax.spines['bottom'].set_bounds(min_x, max_x)
    for tick in ax.get_xaxis().get_major_ticks():
        tick.label.set_fontsize(20)

    plt.ylim([min_time - 0.05*(max_time - min_time), max_time + 0.05*(max_time - min_time)])
    # plt.yticks(yticks + ([min_time, max_time] if cmdline.style < 3 else []))
    # ax.set_yticklabels(ytlbls + (['', ''] if cmdline.style < 3 else []))
    plt.yticks(yticks + [min_time, max_time])
    ax.set_yticklabels(ytlbls +  ['', ''])
    ax.spines['left'].set_bounds(min_time, max_time)
    
    if cmdline.title is not None:
        plt.title(cmdline.title)
    fontprop = fm.FontProperties(fname=cmdline.font) if cmdline.font else None
    plt.xlabel('Task Size', labelpad=20, fontsize=36, fontproperties=fontprop)
    plt.ylabel('Time Required', labelpad=40, fontsize=36, fontproperties=fontprop)

    #plt.xticks([min(x), max(x)] + [x[i] for i in xrange(len(x)) if means[i] == min(means)] + range(100, max(x), 100))
    #plt.yticks([min_time, max_time, means[0]])

    if len(by_selection) > 1:
        plt.legend(loc='upper center', title='Host Selection')
    if cmdline.outfile:
        plt.savefig(cmdline.outfile, format='png')#, bbox_inches='tight')
    if not cmdline.batch:
        plt.show()

def run_sims(mtbe):
    global cmdline
    threads = []
    permit = threading.BoundedSemaphore(cmdline.threads)
    for selection in ['random']: #['random']: #, 'round_robin']:
        for bitesize in cmdline.bitesize:
            for trial in xrange(cmdline.trials):
                permit.acquire()  # thread will release()
                thread = SimThread(permit, bitesize, mtbe, selection)
                thread.start()
                threads.append(thread)
    for thread in threads:
        thread.join()
    if not cmdline.batch and cmdline.tail:
        plt.clf()
        if cmdline.title:
            plt.title(cmdline.title)
        for i in xrange(len(threads)):
            thread = threads[i]
            x = [t[0]/3.6e9 for t in thread.master.data]
            y = [t[1] for t in thread.master.data]
            plt.plot(x, y, '-')
            if cmdline.dump:
                for j in xrange(len(x)):
                    print(i, x[j], y[j], file=cmdline.dump)
        plt.show()

def main():
    global cmdline, num_hosts
    logfile = cmdline.logfile

    if cmdline.plot is not None:
        if cmdline.tail:
            plot_tail(cmdline.plot)
            return
        else:
            read_output(cmdline.plot)
    else:
        mtbe = read_evictions(cmdline.eviction)
        #mtbe = [61121476039, 61121476039, 84252137406, 10020916374, 17008213591, 10369871537, 12320470514, 25091668433, 21498389138, 11207226817, 22948154325, 30499662501, 4362042659, 25580753094, 86625936935, 24489614850, 5639065775, 7110069808, 8318211443, 240975633429, 240975633429, 4282745812, 9721765283, 33483656043, 83552748947, 2746107460, 34007276053, 1375637689, 23052002697, 52227467471]
        if len(mtbe) < num_hosts:
            mtbe = mtbe * (num_hosts//len(mtbe) + 1)
        # mtbe = [m*10.0 if m else None for m in mtbe]
        print('MTBE =', mtbe, file=logfile)
        print('mean MTBE =', Time.to_hour(np.mean([x for x in mtbe if x is not None])), 'hr', '({} None)'.format(len([x for x in mtbe if x is None])))
        run_sims(mtbe)
    if not cmdline.batch and not cmdline.tail:
        plot_results()

if __name__ == '__main__':
    import matplotlib as mpl
    global cmdline
    cmdline = process_cmdline()
    if cmdline.batch:
        mpl.use('agg')  # use X-independent backend
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import matplotlib.font_manager as fm
    plt.rcParams['svg.fonttype'] = 'none'

    main()
#EOF
