#!/bin/env python

from __future__ import print_function

import sys
import argparse
import threading
import numpy as np
from random import random
from tailsim import *

assert sys.version_info >= (2,7)

num_units = 200
num_hosts = 30
host_capacity = 5

runtimes = [LinearRuntime( 60.0*Time.second, 2000.0*Time.second)]*(num_hosts // 3) + \
           [LinearRuntime(110.0*Time.second, 2500.0*Time.second)]*(num_hosts // 3 + num_hosts % 3) + \
           [LinearRuntime(180.0*Time.second, 3000.0*Time.second)]*(num_hosts // 3)

class TestMaster(Master):
    def start(self):
        super(TestMaster, self).start()
        self.schedule_event(self.now, self.assign_available)

class SimThread(threading.Thread):
    def __init__(self, permit, bitesize, mtbe, selection='random'):
        super(SimThread, self).__init__()
        self.permit = permit
        self.bitesize = bitesize
        self.mtbe = mtbe
        self.selection = selection

    def run(self):
        try:
            sim, master = setup_sim(bitesize=self.bitesize, mtbe=self.mtbe, selection='select_' + self.selection)
            sim.start()
            store_result((self.bitesize, Time.to_hour(sim.now), sum(host.evicted_tasks for host in master.hosts), self.selection))
        finally:
            self.permit.release()

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
    sim = Simulation(cmdline.logfile)
    units = [WorkUnit(np.random.normal(loc=1.0, scale=0.055)) for n in xrange(num_units)]
    hosts = [Host(str(n), host_capacity, runtimes[n], mtbe[n]) for n in xrange(num_hosts)]
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
    #parser.add_argument('-d', '--dump', nargs='?', type=argparse.FileType('w'), default=None, help='dump data to file')
    parser.add_argument('--title', nargs='?', type=str, default=None, help='plot title string')
    parser.add_argument('-o', '--outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout, help='data output file')
    parser.add_argument('-l', '--logfile', nargs='?', type=argparse.FileType('w'), default=None, help='simulation log file')
    parser.add_argument('-t', '--trials', nargs='?', type=int, help='number of trials (for each bitesize)')
    parser.add_argument('-j', '--threads', nargs='?', type=int, default=1, help='number of processing threads to use')
    return parser.parse_args()

def main():
    global cmdline
    logfile = cmdline.logfile

    #mtbe = read_evictions(cmdline.eviction)
    mtbe = [None, 61121476039, 84252137406, 10020916374, 17008213591, 10369871537, 12320470514, 25091668433, 21498389138, 11207226817, 22948154325, 30499662501, 4362042659, 25580753094, 86625936935, 24489614850, 5639065775, 7110069808, 8318211443, 240975633429, None, 4282745812, 9721765283, 33483656043, 83552748947, 2746107460, 34007276053, 1375637689, 23052002697, 52227467471]
    print('MTBE =', mtbe, file=logfile)
    print('mean MTBE =', Time.to_hour(np.mean([x for x in mtbe if x is not None])), 'hr', '({} None)'.format(len([x for x in mtbe if x is None])))

    threads = []
    permit = threading.BoundedSemaphore(cmdline.threads)
    for selection in ['random', 'round_robin']:
        for bitesize in cmdline.bitesize:
            for trial in xrange(cmdline.trials):
                permit.acquire()  # thread will release()
                thread = SimThread(permit, bitesize, mtbe, selection)
                thread.start()
                threads.append(thread)

    for thread in threads:
        thread.join()

    global results
    results = sorted(results)

    by_selection = {}
    for row in results:
        selection = row[3]
        if selection not in by_selection:
            by_selection[selection] = []
        by_selection[selection].append(row)

    for selection in by_selection:
        this_selection = by_selection[selection]
        bites = {}
        for v in this_selection:
            if v[0] not in bites:
                bites[v[0]] = []
            bites[v[0]].append(v[1])
        x = sorted(bites.keys())
        means = [np.mean(bites[v]) for v in x]
        mins = [min(bites[x[i]]) for i in xrange(len(x))]
        maxs = [max(bites[x[i]]) for i in xrange(len(x))]
        stds = [np.std(bites[v]) for v in x]
        lows = [abs(means[i] - min(bites[x[i]])) for i in xrange(len(x))]
        highs = [abs(max(bites[x[i]]) - means[i]) for i in xrange(len(x))]
        #plt.errorbar(x, means, yerr=[lows, highs], fmt='-o', elinewidth=1, label=selection)
        #plt.errorbar(x, means, yerr=stds, fmt='-o', elinewidth=1, label=selection)
        #plt.fill_between(x, [means[i] - stds[i] for i in xrange(len(x))], [means[i] + stds[i] for i in xrange(len(x))], alpha=0.2)
        plt.fill_between(x, mins, maxs, alpha=0.2)
        plt.plot(x, means, '-o', alpha=0.5, label=selection)
        # plt.scatter([v[0] for v in this_selection], [v[1] for v in this_selection], marker='o')
        # plt.plot(x, y, label=selection)

    plt.xlabel('Task Size (units)')
    plt.ylabel('Simulation Time (hr)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(by_selection))
    plt.show()

if __name__ == '__main__':
    import matplotlib as mpl
    global cmdline
    cmdline = process_cmdline()
    if cmdline.batch:
        mpl.use('agg')  # use X-independent backend
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt

    main()
#EOF
