#!/bin/python

from __future__ import print_function
import sys

class SimError(Exception):
    """A basic simulation error"""
    def __init__(self, message=None):
        self.message = message

class WorkUnit:
    """
    A model of single unit of work.  Not all WorkUnits are created
    equally, and some may take more time than others; therefore each
    WorkUnit has an intrinisic time-scaling factor.
    """
    def __init__(self, scale):
        self.scale = scale

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        if value < 0:
            raise ValueError("Unit scale cannot be negative")
        self._scale = value

class Task:
    """
    A collection of WorkUnits, which runs on a Host.
    """
    def __init__(self, units):
        self.units = units

    @property
    def scale(self):
        """Returns the total time scaling for all this Task's WorkUnits"""
        return sum(unit.scale for unit in units)

class Host:
    """
    A model of a Host.  A Host runs Tasks, but has a maximum number of
    Tasks that it can run at once (a capacity).  A Host has intrinsic
    properties that influence the running time of Tasks (slope and
    intercept).  A Host may go down, evicting all Tasks that it is
    currently running.
    """
    def __init__(self, name, num_slots):
        self.name = name
        self.capacity = num_slots
        self._tasks = []

    @property
    def capacity(self, max_slots):
        if max_slots < 0:
            raise ValueError("Number of slots cannot be negative")
        self._capacity = max_slots

    def run(self, task):
        """Run a task on this Host"""
        if len(self.tasks) >= self.capacity:
            raise SimError("Host is full; cannot add Task: " + repr(task))
        self.tasks.append(task)

    def available(self):
        """Returns the number of available slots on this Host"""
        return self.capacity - len(self.tasks)

class Master:
    """
    A Master starts with some WorkUnits.  It creates Tasks and tries
    to run them on Hosts.  When a Master gives a Task to a Host, the
    Host may either complete it or return it to Master unfinished.
    """
    def __init__(self, initial_units):
        self.units = initial_units

def main():
    pass

if __name__=='__main__':
    main()
#EOF
