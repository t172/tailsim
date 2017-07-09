from __future__ import print_function

import sys
#from random import random
#from scipy.stats import expon
import numpy as np
from tailsim import *

class TestMaster(Master):
    # def make_tasks(self, time=None):
    #     self.sim.log('Making tasks')
    #     bitesize = 10
    #     while len(self.remaining_units) > 0:
    #         bite = self.remaining_units[0:bitesize]
    #         self.remaining_units = self.remaining_units[bitesize:]
    #         task = self.create_task(bite)
    #         self.submit_task(task)
    #     self.assign_available()
        
    def start(self):
        super(TestMaster, self).start()
        self.schedule_event(self.now, self.assign_available)

def setup():
    num_units = 1000
    num_hosts = 10
    tasks_per_host = 3

    # Running time of a Task
    slope = 20*Time.second
    intercept = 100*Time.second

    # Mean Time Between Evictions
    mtbe = 200*Time.second

    global sim, master
    sim = Simulation()
    units = [WorkUnit(np.random.normal(loc=1.0, scale=0.045)) for n in xrange(num_units)]
    hosts = [Host(str(n), tasks_per_host, LinearRuntime(slope, intercept), mtbe) for n in xrange(num_hosts)]
    master = TestMaster(units, hosts)
    master.bitesize = 10

    sim.add_actor(master)
    for host in hosts:
        sim.add_actor(host)
    return sim

def main():
    global sim, master
    sim = setup()
    sim.start()

    print()
    print(len(master.processed_units), 'units processed')
    print(len(master.remaining_units), 'units remaining')
    print()

    for task in master._graveyard:
        if len(task.history) >= master.max_retries:
            print(task)

if __name__ == '__main__':
    main()
#EOF
