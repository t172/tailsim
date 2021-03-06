#!/bin/python

from __future__ import print_function

import sys
import random
from collections import deque
from scipy.stats import expon

assert sys.version_info >= (2,7)

class Time:
    # Simple time units.  (Multiply when storing, divide when retrieving.)
    microsecond = 1.0
    millisecond = 1e3
    second = 1e6
    minute = 6e7
    hour = 3.6e9

    @classmethod
    def to_hour(cls, value):
        return value/float(Time.hour)
    @classmethod
    def to_second(cls, value):
        return value/float(Time.second)
    @classmethod
    def to_millisecond(cls, value):
        return value/float(Time.millisecond)
    @classmethod
    def to_microsecond(cls, value):
        return value/float(Time.microsecond)

class SimulationError(Exception):
    """A logical error in the simulation."""
    def __init__(self, time, message=None):
        super(SimulationError, self).__init__()
        self.time = time
        self.message = message

    def __str__(self):
        return "Simulation error (time {}): {}".format(self.time, self.message)
    def __repr__(self):
        return "{}({},{})".format(self.__class__.__name__, repr(self.time), repr(self.message))

class Event(object):
    """
    An Event is any action that changes the state of the simulation.
    It has a scheduled simulation time (when the event should occur)
    and an action (a lambda expression) that is called with the
    current time when the event is applied.
    """
    def __init__(self, actor, scheduled_time, action):
        self.actor = actor
        self.time = scheduled_time
        self.action = action

    def __call__(self):
        """
        Perform the action of this Event, applying it to the state of the
        simulation.
        """
        self.action()

    def __repr__(self):
        return "{}({}, actor={}, action={})".format(self.__class__.__name__, self.time, self.actor.__class__.__name__, self.action)

class Actor(object):
    """
    An Actor influences a Simulation by changing its state through the
    generation (and application) of Events.  A Simulation first
    informs an Actor that it is bound to the Simulation and that it is
    ready to receive Events.  The Actor then schedules Events with the
    Simulation, which applies them at the appropriate simulation time.
    An Actor may request that an Event be canceled if it has not yet
    been applied.
    """
    def __init__(self):
        self._simulation = None

    @property
    def sim(self):
        """The Simulation this Actor is bound to"""
        return self._simulation

    @property
    def now(self):
        """The current time of the Simulation this Actor is bound to.""" 
        return self.sim.now

    def bind_to(self, simulation):
        """
        Binds this Actor to the given Simulation.  Any subsequent Events
        generated should be scheduled with the Simulation.
        """
        if not isinstance(simulation, Simulation):
            raise ValueError("{} must be a Simulation".format(simulation))
        self._simulation = simulation

    def schedule_event(self, time, action):
        """
        Schedules an Event with the Simulation this Actor is bound to.
        The return value is the argument to pass to cancel_event() to
        cancel this action.
        """
        return self.sim.schedule(Event(self, time, action))

    def cancel_event(self, handle):
        """
        Request that the Simulation cancel an event specified by the
        argument.  The argument must be something returned by
        schedule_event().
        """
        self.sim.cancel(handle)

    def start(self):
        """
        Called by the Simulation to notify the Actor that the Simulation
        is about to start.  This allows the Actor to schedule its
        initial Events.  (A subclass should override if interested.)
        """
        pass

    def commit_to(self, future_time):
        """
        Gets a commitment from this Actor before advancing the simulation
        time.  This is a negotiation between the Actor and Simulation,
        and the Actor is not required to commit to as much time as
        requested.  The Simulation calls this function before
        advancing the simulation time to get the Actor's promise that
        it will schedule no more events before the given time.  This
        is a "last call" for the Actor to do so, and is useful for
        stochastic Actors to insert random events.  The returned value
        is the time this Actor is willing to commit to, which may be
        before the requested future_time, but may not be in the past
        (before the current simulation time).  An Actor may schedule
        new events at the current time (now), but notice that never
        committing to a time in the future prevents time from
        progressing, and the simulation will stop.  A Simulation
        should never request a commitment beyond any Actor's next
        event.  This base class implementation always commits the
        Actor to whatever time is requested.  A subclass (such as a
        stochastic Actor) must override this if it is interested in
        inserting new (e.g. random) events or negotiating time
        progression.
        """
        return future_time, False

class Simulation(object):
    """
    A Simulation maintains a simulation timeline, has a collection of
    Actors that submit Events, and applies the Events submitted at the
    appropriate simulation time.
    """
    def __init__(self, logfile=sys.stderr):
        self.now = 0
        self.actors = []
        self._schedule = deque()
        self._commit_time = self.now  # latest unanimous commitment
        self.logfile = logfile

    def add_actor(self, actor):
        """Adds an actor to this Simulation"""
        if not isinstance(actor, Actor):
            raise ValueError("Must be a simulation actor (Actor): {}".format(actor))
        self.actors.append(actor)
        actor.bind_to(self)
        actor.commitment = self.now

    def schedule(self, event):
        """
        Schedules an Event to be applied to the Simulation.  Returns the
        argument to pass to cancel() should the Actor decide to cancel
        this Event.
        """
        if event.time < self.now:
            raise SimulationError(self.now, "Can't schedule Event in the past: {}".format(event))
        if event.time < event.actor.commitment:
            raise SimulationError(self.now, "{} violates {}'s commitment of time {}".format(event, event.actor, event.actor.commitment))

        # Insert new event in order
        if len(self._schedule) == 0:
            self._schedule.appendleft(event)
        else:
            rotation = 0
            while rotation < len(self._schedule) and event.time < self._schedule[rotation].time:
                rotation += 1
            self._schedule.rotate(-rotation)
            self._schedule.appendleft(event)
            self._schedule.rotate(rotation)

        # Here, we simply return the event itself, but this should be
        # considered an implementation detail.  What the return value
        # really is should be opaque to the Actor, but it must
        # identify the event sufficiently to cancel() it.
        return event

    def cancel(self, handle):
        """Attempt to cancel a scheduled Event"""
        # In this implementation, schedule() returns the event itself,
        # so we just have to try to remove it from the schedule.
        try:
            self._schedule.remove(handle)
        except ValueError:
            raise SimulationError(self.now, "Cannot cancel event {}".format(handle))

    def cancel_all_from(self, actor):
        """
        Cancel all scheduled events from the given Actor.  Returns the
        number of events canceled.
        """
        num_canceled = 0
        new_schedule = deque()

        while True:
            try:
                event = self._schedule.pop()
            except IndexError:
                break
            if event.actor == actor:
                num_canceled += 1
            else:
                new_schedule.appendleft(event)
        self._schedule = new_schedule
        return num_canceled

    def log(self, msg):
        """Logs the given message"""
        if self.logfile is not None:
            print(self.now, msg, file=self.logfile)

    def take_next(self):
        """
        Gets a commitment to an event, removes it from the schedule, and
        yields it.
        """
        while True:
            if self.stop_requested:
                self.log("Stopping")
                return

            # First event may be within commitment
            try:
                if self._commit_time >= self._schedule[-1].time:
                    yield self._schedule.pop()
                    continue
            except IndexError:
                break

            # Try to extend commitment
            dirty = True
            while dirty:
                commitments = []
                for actor in self.actors:
                    commitment, dirty = actor.commit_to(self._schedule[-1].time)
                    if dirty:
                        break
                    commitments.append(commitment)
            self._commit_time = min(commitments)

            # Stop if no consensus to next event
            try:
                if self._commit_time < self._schedule[-1].time:
                    self.log("Can't get commitment to next event: {}".format(self._schedule[-1]))
                    return
                else:
                    yield self._schedule.pop()
            except IndexError:
                break

        # Finished
        self.log("No events left")
        return

    def start(self):
        """
        Starts the Simulation.  The Simulation stops when there are no
        remaining scheduled Events.
        """
        self.log("{} has {} actors and {} scheduled events".format(self.__class__.__name__, len(self.actors), len(self._schedule)))
        self.stop_requested = False
        if len(self.actors) == 0:
            self.log("No Actors, quitting")
            return

        self.log("Sending start notifications to Actors")
        for actor in self.actors:
            actor.start()

        self.log("Starting with {} scheduled events".format(len(self._schedule)))
        for event in self.take_next():
            if event.time < self.now:
                raise SimulationError(self.now, "Event is stale: {}".format(event))
            self.now = event.time
            event()
        self.log("Done")

    def stop(self, reason=None):
        """Requests the Simulation to stop."""
        if reason:
            self.log('Stop requested: ' + str(reason))
        self.stop_requested = True

class WorkUnit(object):
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
            raise SimulationError(time, "{} already processed".format(self))
        if time is None:
            raise ValueError("No processing time given")
        self._processed = time

    def __repr__(self):
        return "{}({},{})".format(self.__class__.__name__, self._scale, self._processed)

class TaskRun(object):
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
        raise SimulationError(time, "Invalid state transition: {} to {}".format(TaskRun.state_str(self._state), TaskRun.state_str(to_state)))

    def __repr__(self):
        s = "{}({}".format(self.__class__.__name__, TaskRun.state_str(self._state))
        if self._state != TaskRun.WAITING:
            s += ":{}, {}, {}".format(self._host.host_id if isinstance(self._host, Host) else self._host, self._start, self._finish)
        return s + ')'

class Task(object):
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
    def is_done(self):
        return self.state == TaskRun.DONE

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
            raise SimulationError(time, "Can't rerun a completed task: {}".format(self))
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

class RuntimeModel(object):
    def runtime_of(self, task):
        """
        Calculates the running time of a given task according to this
        RuntimeModel.
        """
        raise NotImplementedError("A subclass should implement this method")

class LinearRuntime(RuntimeModel):
    """
    A simple linear model of the running time of a Task, with a slope
    and y-intercept.
    """
    def __init__(self, slope, intercept):
        super(LinearRuntime, self).__init__()
        self._slope = slope
        self._intercept = intercept

    def runtime_of(self, task):
        """Calculates the running time of a given Task"""
        return int(self._slope*task.scale + self._intercept)

    def __repr__(self):
        return "{}({}x + {})".format(self.__class__.__name__, self._slope, self._intercept)

class Host(Actor):
    """
    A model of a Host.  A Host runs Tasks, but has a maximum number of
    Tasks that it can run at once (a capacity).  A Host has intrinsic
    properties that influence the running time of Tasks.  A Host may
    go down, evicting all Tasks that it is currently running, and
    return them to the Master that submitted them.  (Note: There is no
    representation of Host resources (CPU, memory, etc.) to be matched
    to Task requirements.)
    """
    def __init__(self, host_id, num_slots, runtime_model, mtbe=None, restart_delay=0):
        """
        Initializes a Host with the given ID, number of slots with which
        to run Tasks, a runtime model that determines the running time
        of a Task, and an optional mean time between evictions.
        Eviction is modeled as a Poisson process, with the parameter
        'mtbe' being the mean time between evictions (scale
        of an exponential distribution; where scale = 1/lambda).  The
        restart_delay is the time before a Host comes 'up' after going
        'down'.
        """
        # Note that when this initializer is called, the Host will not
        # be bound to a Simulation.  To initialize with a starting
        # simulation time, for example, we need to override bind_to().
        super(Host, self).__init__()
        self.up = False  # start() will go_up()
        self.host_id = host_id
        self.runtime_model = runtime_model
        self.mtbe = mtbe
        self.restart_delay = restart_delay
        self._tasks = []
        self._callback = {}
        self._announce_list = []
        self.evicted_tasks = 0       # kill counter

        if num_slots < 0:
            raise ValueError("Number of Task slots on a Host cannot be negative")
        self.capacity = num_slots
        self._available = num_slots  # number allocated to this Host
        self._running = 0            # number currently running

        if restart_delay < 0:
            raise ValueError("Restart delay cannot be negative")
        self.restart_delay = restart_delay

    def start(self):
        super(Host, self).start()

        # Go up, but do not announce.  This is to try to model the
        # hosts having started sometime in the past.  This is to allow
        # for the master to wait to assign its initial tasks.  Hosts
        # can randomly go up and down before the master begins.
        # Otherwise, there is a brief period at the beginning when
        # evictions on all hosts are improbable
        self.go_up(announce=False)

    # def bind_to(self, simulation):
    #     """
    #     Overrides Actor.bind_to() to initialize with the initial
    #     simulation time.
    #     """
    #     super(Host, self).bind_to(simulation)
    #     self.last_eviction = simulation.now

    def assign(self, task, callback):
        """
        Assigns a task to this Host.  This schedules Events to run() and
        complete() the Task.  This allocates a slot for the given
        task, but it is not counted as running until the run() event
        is processed.
        """
        if self._available <= 0:
            raise SimulationError(self.now, "Host {} cannot be assigned any more tasks".format(repr(self.host_id)))
        self._available -= 1

        start_time = self.now
        finish_time = start_time + self.runtime_model.runtime_of(task)

        self.schedule_event(start_time, lambda: self.run(task, callback))
        self.schedule_event(finish_time, lambda: self.complete(task))

    def run(self, task, callback):
        """
        Runs a task on this Host.  When the Task is no longer running on
        this Host (for whatever reason), the callback is called.
        """
        self.sim.log("Running Task {} on Host {}".format(task.task_id, self.host_id))
        if not self.up:
            raise SimulationError(self.now, "Host {} is DOWN, cannot run Task: {}".format(repr(self.host_id), task))
        if self._running >= self.capacity:
            raise SimulationError(self.now, "Host {} is at capacity; cannot run Task: {}".format(repr(self.host_id), task))
        self._tasks.append(task)
        self._callback[task] = callback
        task.run(self.now, self)
        self._running += 1

    def complete(self, task):
        """Completes a Task"""
        task.complete(self.now)
        self.sim.log("Task {} completed on Host {}".format(task.task_id, self.host_id))
        self.return_task(task)
        self._running -= 1
        self._available += 1

    def evict_all(self):
        """Evicts all Tasks currently running on this Host"""
        for task in self._tasks[:]:
            task.evict(self.now)
            self.evicted_tasks += 1
            self.return_task(task)
        self._running = 0
        self._available = self.capacity

    def return_task(self, task):
        """
        Returns a task to its submitter using the callback provided and
        remove it from this Host.
        """
        if task in self._callback:
            self._callback[task](task)
            del self._callback[task]
        self._tasks.remove(task)

    def available(self):
        """Returns the number of available slots on this Host"""
        if self.up:
            return self._available
            #return self.capacity - len(self._tasks)
        else:
            return 0

    def running(self):
        """Returns the number of tasks running on this Host"""
        return self._running

    def assigned(self):
        """
        Returns the number of tasks assigned on this Host.  Note that this
        may be different from the number running, if the assigned
        tasks have yet to start running.
        """
        return self.capacity - self._available

    def register_announcement(self, master):
        """
        This function is called when a Master requests this Host to
        announce itself whenever it becomes available again
        (i.e. going 'up' after being 'down').
        """
        self._announce_list.append(master)

    def go_up(self, announce=True):
        """Bring this Host 'up', allowing it to process Tasks"""
        self.sim.log("Bringing {} UP".format(repr(self.host_id)))
        if self.up:
            raise SimulationError(self.now, "{} is already up.".format(self))

        self.up = True
        self.up_since = self.now
        self._available = self.capacity
        self._running = 0
        self.last_eviction = self.now
        if announce:
            for master in self._announce_list:
                master.announce(self)

    def go_down(self):
        """
        Bring this Host 'down', evicting all current Tasks, preventing it
        from accepting new ones, and schedule an event to bring the
        Host back up.
        """
        self.sim.log("Bringing {} DOWN".format(repr(self.host_id)))
        if not self.up:
            raise SimulationError(self.now, "{} is already down: {}".format(self.__class__.__name__, self))

        # Change state
        self.up = False
        self.evict_all()
        self._available = 0
        self._running = 0
        
        # Cancel all events and schedule restart
        self.sim.cancel_all_from(self)
        self.schedule_event(self.now + self.restart_delay, self.go_up)

    def commit_to(self, future_time):
        """Time progression negotiation."""
        if not self.up or future_time < self.last_eviction or self.mtbe is None:
            return future_time, False

        # Randomly insert another eviction event
        next_eviction = self.last_eviction + int(expon.rvs(scale=self.mtbe))
        # if next_eviction < self.now:
        #     import pdb; pdb.set_trace()
        self.schedule_event(next_eviction, self.go_down)
        self.last_eviction = next_eviction

        # Only commit as much as necessary
        return min(future_time, next_eviction), True

    def __repr__(self):
        return "{}(host_id={}, capacity={}, tasks={})".format(self.__class__.__name__, self.host_id, self.capacity, [task.task_id for task in self._tasks])

class Master(Actor):
    """
    A Master starts with WorkUnits and Hosts.  It creates Tasks from
    the given WorkUnits and submits them to Hosts, trying to process
    all the WorkUnits.  A Host may or may not process the Task before
    returning it to the Master.
    """
    def __init__(self, initial_units, hosts):
        super(Master, self).__init__()
        self.remaining_units = initial_units[:]
        self.processed_units = []
        self.num_units = len(initial_units)
        self.queue = deque()
        self._num_tasks_created = 0
        self._graveyard = []
        self.max_retries = 10

        # Connect to hosts
        if len(hosts) == 0:
            raise ValueError('No hosts given')
        self.hosts = hosts[:]
        for host in hosts:
            host.register_announcement(self)

        # Default host selection technique
        self.select_host = self.select_first_host

    def start(self):
        super(Master, self).start()
        self.sim.log("{} starting with {} units and {} hosts".format(self.__class__.__name__, self.num_units, len(self.hosts)))

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
        self.sim.log('Making Task {} with {} units'.format(self._num_tasks_created, len(units)))
        return Task(self._num_tasks_created, units)

    def destroy_task(self, task):
        """
        Absorbs the WorkUnits from a Task, returning them to the
        appropriate pool and places the Task in the graveyard.
        """
        for unit in task.units:
            if unit.processed:
                self.processed_units.append(unit)
            else:
                self.remaining_units.append(unit)
        self._graveyard.append(task)

        # Check for end of job
        num_left = self.units_remaining()
        self.sim.log("{} units remaining".format(num_left))
        if num_left == 0:
            self.sim.stop("All units processed")

    def announce(self, host):
        """
        A host announces to this Master that it is now up and available to
        receive tasks.
        """
        self.assign_available()

    def assign_task(self, task):
        """Selects a Host and assigns it the given Task."""
        host = self.select_host(task)
        if host is None:
            raise SimulationError(self.now, "No Host for Task")
        host.assign(task, self.receive_task)

    def assign_available(self):
        """Assign as many tasks to Hosts as possible"""
        if len(self.queue) == 0:
            if self.remaining_units > 0:
                self.make_tasks()

        if len(self.queue) == 0:
            for host in self.hosts:
                if host.assigned():
                    return
            # Nothing in queue, nothing assigned, no remaining_units, so nothing left to do
            self.sim.stop("Nothing queued, nothing to assign")
            return

        # Remove from queue and assign to hosts
        num_available = sum((host.available() for host in self.hosts))
        while num_available > 0 and len(self.queue) > 0:
            self.assign_task(self.queue.pop())
            num_available -= 1

    def make_tasks(self):
        """Populate the queue with tasks"""
        bitesize = getattr(self, 'bitesize', 1)
        num_made = 0
        while len(self.remaining_units) > 0:
            bite = self.remaining_units[0:bitesize]
            self.remaining_units = self.remaining_units[bitesize:]
            self.submit_task(self.create_task(bite))
        return num_made

    def submit_task(self, task):
        """Submits a task to the queue"""
        self.queue.appendleft(task)

    def receive_task(self, task):
        """
        Receives a Task returned from a Host, and resubmits or destroys
        it.  (This is the default callback when submitting to Hosts.)
        """
        if not task.is_done and len(task.history) < self.max_retries:
            self.submit_task(task)
        else:
            self.destroy_task(task)
        self.assign_available()

    def units_remaining(self):
        """Returns the number of work units remaining"""
        return self.num_units - len(self.processed_units)

    # How to select a Host for a given Task is an important strategy,
    # so multiple implementations are given.  Whatever select_host
    # points to will be called to make this decision.

    def select_random(self, task):
        """Select an available host at random."""
        try:
            return random.choice([host for host in self.hosts if host.available()])
        except IndexError:
            return None

    def select_first_host(self, task):
        """Selects the first Host that can run the given Task."""
        for host in self.hosts:
            if host.available():
                return host
        return None

    def select_round_robin(self, task):
        """Selects the next available Host in a round-robin fashion"""
        num_hosts = len(self.hosts)
        last_index = getattr(self, '_last_round_robin', num_hosts - 1)
        for h in xrange(1, num_hosts + 1):
            index = (last_index + h) % num_hosts
            host = self.hosts[index]
            if host.available():
                self._last_round_robin = index
                return host
        return None

    def select_least_busy(self, task):
        """Selects the Host with the lowest number of Tasks"""
        return self.select_by_busyness(task, True)

    def select_most_busy(self, task):
        """Selects the Host with the highest number Tasks"""
        return self.select_by_busyness(task, False)

    def select_by_busyness(self, task, order):
        """Selects an available Host based on the number of free slots."""
        try:
            return sorted([(host.available(), host) for host in self.hosts if host.available()], reverse=order)[0][1]
        except IndexError:
            return None

def main():
    pass

if __name__=='__main__':
    main()
#EOF
