#!/bin/python

from __future__ import print_function
import sys

assert sys.version_info >= (2,7)

class Time:
    # Simple time units.  (Multiply when storing, divide when retrieving.)
    microsecond = 1.0
    millisecond = 1e3
    second = 1e6
    hour = 3.6e9

class SimError(Exception):
    """A basic simulation error"""
    def __init__(self, time, message=None):
        self.time = time
        self.message = message

    def __str__(self):
        return "Simulation Error at {}: {}".format(self.time, self.message)
    def __repr__(self):
        return "{}({},{})".format(self.__class__.__name__, self.time, self.message)

class SimEvent:
    """
    A SimEvent is any action that changes the state of the simulation.
    It has a scheduled simulation time (when the event should occur)
    and an action (a lambda expression) that will be called with the
    current simulation time when the event will be applied.
    """
    def __init__(self, scheduled_time, action):
        self._time = scheduled_time
        self._action = action

    def __call__(self, time):
        """Perform the action of this event"""
        self._action(time)

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self._time, self._action)

class SimActor:
    """
    A SimActor is anything in the simulation that can create
    SimEvents.
    """
    def __init__(self, sim):
        self._sim = sim

class Simulation:
    """
    A Simulation maintains a simulation timeline, has a collection of
    SimActors that submit SimEvents, and performs the SimEvents
    submitted.
    """
    def __init__(self):
        self._actors = []
        self._events = []

    def add_actor(self, actor):
        """Adds an actor to this simulation"""
        if not isinstance(actor, SimActor):
            raise ValueError("Must be a simulation actor (SimActor): {}".format(actor))
        self._actors.append(actor)

    def register(self, event):
        """Registers a SimEvent in the simulation"""

        self._events
        

# class ScheduledEvent(SimEvent):
#     def __init__(self, time, action):
#         super(ScheduledEvent, self).__init__(time, action)
        
# class RandomEvent(SimEvent):
#     def __init__(self, time, action):
#         super(RandomEvent, self).__init__(time, action)

class WorkUnit:
    """
    A model of a single unit of work.  Not all WorkUnits are created
    equally, and some may require more running time than others (even
    on the same Host).  To model this, each WorkUnit has an intrinisic
    time-scaling factor, or "scale" for short.
    """
    def __init__(self, scale):
        if scale < 0:
            raise ValueError("Unit scale cannot be negative: {}".format(scale))
        self._scale = scale
        self._processed = None

    @property
    def scale(self):
        """The time-scaling factor for this WorkUnit"""
        return self._scale

    @property
    def processed(self):
        """The time when this WorkUnit was processed, or None"""
        return self._processed

    def process(self, time):
        """Mark this WorkUnit as processed"""
        if self._processed != None:
            raise SimError(time, "{} already processed".format(self))
        if time is None:
            raise ValueError("No processing time given")
        self._processed = time

    def __repr__(self):
        return "{}({},{})".format(self.__class__.__name__, self._scale, self._processed)

class TaskRun:
    """
    A TaskRun is a record of an attempt to run a Task on a Host.  It
    is a simple state machine that enforces state transitions and
    records when a Task started and finished running, and on which
    Host.  A Task maintains its history as a collection of TaskRuns.
    """
    WAITING = 0  # waiting placement on a host
    RUNNING = 1  # running on a host
    EVICTED = 2  # evicted from a host that went down
    DONE = 3     # task completed

    # Valid state transitions:
    # WAITING --(run)--> RUNNING ---(evict)----> EVICTED
    #                    RUNNING --(complete)--> DONE

    @classmethod
    def state_str(cls, state):
        if state == cls.WAITING:
            return 'WAITING'
        elif state == cls.RUNNING:
            return 'RUNNING'
        elif state == cls.EVICTED:
            return 'EVICTED'
        elif state == cls.DONE:
            return 'DONE'
        else:
            raise ValueError("Invalid {} state: {}".format(self.__class__.__name__, state))

    def __init__(self):
        self._state = TaskRun.WAITING
        self._start = None
        self._finish = None
        self._host = None

    @property
    def state(self):
        """The current state of this TaskRun"""
        return self._state

    @property
    def host(self):
        """The Host on which this TaskRun was attempted"""
        return self._host

    def lifetime(self, now=None):
        """The time this TaskRun has run on the Host"""
        finish = self._finish if self._finish != None else now
        return finish - self._start if self._start != None and finish != None else 0

    def finished(self):
        """Whether or not this TaskRun is finished"""
        return self._finish != None

    def run(self, time, host):
        """Start running on a given host"""
        if self._state != TaskRun.WAITING:
            self.invalid_transition_to(TaskRun.RUNNING, time)
        self._state = TaskRun.RUNNING
        self._start = time
        self._host = host

    def complete(self, time):
        if self._state != TaskRun.RUNNING:
            self.invalid_transition_to(TaskRun.DONE, time)
        self._state = TaskRun.DONE
        self._finish = time

    def evict(self, time):
        if self._state != TaskRun.RUNNING:
            self.invalid_transition_to(TaskRun.EVICTED, time)
        self._state = TaskRun.EVICTED
        self._finish = time

    def invalid_transition_to(self, to_state, time):
        raise SimError(time, "Invalid state transition: {} to {}".format(TaskRun.state_str(self._state), TaskRun.state_str(to_state)))

    def __repr__(self):
        s = "{}({}".format(self.__class__.__name__, TaskRun.state_str(self._state))
        if self._state != TaskRun.WAITING:
            s += ":{}, {}, {}".format(self._host.host_id if isinstance(self._host, Host) else self._host, self._start, self._finish)
        return s + ')'

class Task:
    """
    A collection of WorkUnits, which runs on a Host and maintains its
    history of TaskRuns.  A Task has a task ID, for the convenience of
    the Master that created it.  A Task's size is the number of
    WorkUnits it has, and a Task's scale is the sum of the time
    scaling factors of all its WorkUnits.  The scale is used to model
    how much running time it requires.
    """
    def __init__(self, task_id, units):
        self._task_id = task_id
        self._units = units
        self._scale = sum(unit.scale for unit in units)
        self._current = TaskRun()
        self._history = [self._current]

    @property
    def size(self):
        """
        The number of WorkUnits in this Task.  Note that this does not
        take into account the time scaling of the WorkUnits.
        """
        return len(self._units)

    @property
    def scale(self):
        """The total time scaling for all this Task's WorkUnits"""
        return self._scale

    @property
    def units(self):
        """The collection of WorkUnits of this Task"""
        return self._units

    @property
    def task_id(self):
        """The assigned ID for this Task"""
        return self._task_id

    @property
    def state(self):
        """The current state of this Task (see class TaskRun)"""
        return self._current.state

    @property
    def history(self):
        return self._history[:]

    def run(self, time, host):
        """Starts running this Task on a given Host"""
        if self.state == TaskRun.DONE:
            raise SimError(time, "Can't rerun a completed task: {}".format(self))
        if self._current.finished():
            self._current = TaskRun()
            self._history.append(self._current)
        self._current.run(time, host)

    def complete(self, time):
        """Completes this Task"""
        self._current.complete(time)
        for unit in self._units:
            unit.process(time)

    def evict(self, time):
        """Informs this Task that it has been evicted"""
        self._current.evict(time)

    def __repr__(self):
        return "{}(task_id={}, size={}, scale={}, {})".format(self.__class__.__name__, self._task_id, self.size, self.scale, self._history)

class LinearRuntime:
    """
    A simple linear model of the running time of a Task, with a slope
    and y-intercept.
    """
    def __init__(self, slope, intercept):
        self._slope = slope
        self._intercept = intercept

    def runtimeOf(self, task):
        """Calculates the running time of a given Task"""
        return self._slope*task.scale + self._intercept

    def __repr__(self):
        return "{}({}x + {})".format(self.__class__.__name__, self._slope, self._intercept)

class Host:
    """
    A model of a Host.  A Host runs Tasks, but has a maximum number of
    Tasks that it can run at once (a capacity).  A Host has intrinsic
    properties that influence the running time of Tasks.  A Host may
    go down, evicting all Tasks that it is currently running, and
    return them to the Master that submitted them.  (Note: There is no
    representation of Host resources (CPU, memory, etc.) to be matched
    to Task requirements.)
    """
    def __init__(self, host_id, num_slots, runtime_model):
        self._host_id = host_id
        if num_slots < 0:
            raise ValueError("Number of Task slots on a Host cannot be negative")
        self._capacity = num_slots
        self._tasks = []
        self._runtime_model = runtime_model
        self._callback = {}

    @property
    def host_id(self):
        return self._host_id

    @property
    def capacity(self):
        return self._capacity

    def run(self, time, task, callback):
        """
        Runs a task on this Host.  When the Task is no longer running on
        this Host (for whatever reason), the callback is called.
        """
        if not self.available():
            raise SimError(time, "Host is full; cannot run Task: {}".format(task))
        self._tasks.append(task)
        self._callback[task] = callback

    def complete(self, time, task):
        """Complete a Task"""
        task.complete(time)
        self.return_task(time, task)

    def evict(self, time):
        """Evicts all Tasks currently running on this Host"""
        for task in self._tasks:
            task.evict(time)
            self.return_task(time, task)

    def return_task(self, time, task):
        """
        Returns a task to its submitter using the callback provided
        (passing the simulation time) and remove it from this Host.
        """
        if task in self._callback:
            self._callback[task](time)
            del self._callback[task]
        self._tasks.remove(task)

    def available(self):
        """Returns the number of available slots on this Host"""
        return self._capacity - len(self._tasks)

    def __repr__(self):
        return "{}(host_id={}, capacity={}, tasks={})".format(self.__class__.__name__, self.host_id, self.capacity, self._tasks)

class Master:
    """
    A Master starts with WorkUnits and Hosts.  It creates Tasks from
    the given WorkUnits and submits them to Hosts, trying to process
    all the WorkUnits.  A Host may or may not process the Task before
    returning it to the Master.
    """
    def __init__(self, initial_units, hosts):
        self._remaining_units = initial_units[:]
        self._processed_units = []
        if len(hosts) == 0:
            raise ValueError("No hosts given")
        self._hosts = hosts[:]
        self._num_tasks_created = 0
        self._graveyard = []
        self.max_retries = 10
        self.select_host = self.select_first_host

    @property
    def max_retries(self):
        """
        The maximum number of attempts to run a Task before reabsorbing
        its WorkUnits back into the remaning units pool.
        """
        return self._max_retries

    @max_retries.setter
    def max_retries(self, value):
        if value <= 0:
            raise ValueError("Invalid number of retries: {}".format(value))
        self._max_retries = value

    def create_task(self, units):
        """Creates a new Task with a collection of WorkUnits"""
        self._num_tasks_created += 1
        return Task(self._num_tasks_created, units)

    def destroy_task(self, task):
        """
        Absorbs the WorkUnits from a Task, returning them to the
        appropriate pool and places the Task in the graveyard.
        """
        for unit in task.units:
            if unit.processed:
                self._processed_units.append(unit)
            else:
                self._remaining_units.append(unit)
        self._graveyard.append(task)

    def submit_task(self, time, task):
        """Selects a Host, and submits a given Task to it."""
        host = self.select_host(task)
        if host is None:
            raise SimError(time, "No Host for Task")
        host.run(time, task, lambda t : self.receive_task(t, task))

    def receive_task(self, time, task):
        """
        Receives a Task returned from a Host, and resubmits or destroys
        it.  (This is used as the callback when submitting to Hosts.)
        """
        if len(task.history) < self.max_retries:
            self.submit_task(time, task)
        else:
            self.destroy_task(task)

    # How to select a Host for a given Task is an important strategy,
    # so multiple implementations are given.  Whatever select_host
    # points to will be called to make this decision.

    def select_first_host(self, task):
        """Selects the first Host that can run the given Task."""
        for host in self._hosts:
            if host.available():
                return host
        return None

    def select_round_robin(self, task):
        """Selects the next available Host in a round-robin fashion"""
        num_hosts = len(self._hosts)
        last_index = getattr(self, '_last_round_robin', num_hosts - 1)
        for h in xrange(1, num_hosts + 1):
            index = (last_index + h) % num_hosts
            host = self._hosts[index]
            if host.available():
                self._last_round_robin = index
                return host
        return None

    def select_least_busy(self, task):
        """Selects the Host with the lowest number of Tasks"""
        self.select_by_busyness(task, True)

    def select_most_busy(self, task):
        """Selects the Host with the highest number Tasks"""
        self.select_by_busyness(task, False)

    def select_by_busyness(self, task, order):
        """Selects an available Host based on the number of free slots."""
        availability = [(host, host.available()) for host in self._hosts if host.available()]
        if len(availability) == 0:
            return None
        return sorted(availability, key=lambda p: p[1], reverse=order)[0]

def main():
    pass

if __name__=='__main__':
    main()
#EOF
