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

class SimulationError(Exception):
    """A logical error in the simulation."""
    def __init__(self, time, message=None):
        super(SimulationError, self).__init__()
        self.time = time
        self.message = message

    def __str__(self):
        return "Simulation Error at time {}: {}".format(self.time, self.message)
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

    def __call__(self, time):
        """
        Perform the action of this Event, applying it to the state of the
        simulation.  The current simulation time is passed as the
        argument.
        """
        self.action(time)

    def __repr__(self):
        return "{}({}, actor={}, action={})".format(self.__class__.__name__, self.time, self.actor, self.action)

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
    def simulation(self):
        """The Simulation this Actor is bound to"""
        if self._simulation is None:
            raise SimulationError(None, "{} is not registered with any Simulation: {}".format(self.__class__.__name__, self))
        return self._simulation

    @simulation.setter
    def simulation(self, simulation):
        self.bind_to(simulation)

    def bind_to(self, simulation):
        """
        Binds this Actor to the given Simulation.  Any subsequent Events
        generated may be scheduled with the Simulation specified.
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
        event = Event(self, time, action)
        return self.simulation.schedule(event)

    def cancel_event(self, handle):
        """
        Request that the Simulation cancel an event specified by the
        argument.  The argument must be something returned by
        schedule_event().
        """
        self.simulation.cancel(handle)

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
        return future_time

    def start(self):
        """
        Called by the Simulation to notify the Actor that the Simulation
        is about to start.  This allows the Actor to schedule its
        initial Events.  (A subclass should override if interested.)
        """
        pass

class Simulation(object):
    """
    A Simulation maintains a simulation timeline, has a collection of
    Actors that submit Events, and applies the Events submitted at the
    appropriate simulation time.
    """
    def __init__(self):
        self.now = 0
        self.actors = []
        self._schedule = []
        self._canceled_events = []
        self._past_events = []
        self._commit_of = {}  # times Actors have committed to
        self._commit_time = self.now  # latest unanimous commitment

    def add_actor(self, actor):
        """Adds an actor to this Simulation"""
        if not isinstance(actor, Actor):
            raise ValueError("Must be a simulation actor (Actor): {}".format(actor))
        self.actors.append(actor)
        actor.bind_to(self)
        self._commit_of[actor] = self.now

    def schedule(self, event):
        """
        Schedules an Event to be applied to the Simulation.  Returns the
        argument to pass to cancel() should the Actor decide to cancel
        this Event.
        """
        if event.time < self.now:
            raise SimulationError(self.now, "Can't schedule Event in the past: {}".format(event))
        if event.time < self._commit_of[event.actor]:
            raise SimulationError(self.now, "{} violates {}'s commitment of time {}".format(event, event.actor, self._commit_of[event.actor]))
        self._schedule.append(event)
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
        self._canceled_events.append(handle)
        self.log("Canceled {}".format(handle))

    def cancel_all_from(self, actor):
        """
        Cancel all scheduled events from the given Actor.  Returns the
        number of events canceled.
        """
        num_canceled = 0
        new_schedule = []
        for event in self._schedule:
            if event.actor == actor:
                self._canceled_events.append(event)
                self.log("Canceled {}".format(event))
                num_canceled += 1
            else:
                new_schedule.append(event)
        self._schedule = new_schedule
        return num_canceled

    def log(self, msg):
        """Logs the given message"""
        print(self.now, msg, file=sys.stderr)

    def take_next(self):
        """
        Gets a commitment to an event, removes it from the schedule, and
        yields it.
        """
        while True:
            # First event may be within commitment
            try:
                self._schedule = sorted(self._schedule, key=lambda e: e.time)
                next_event = self._schedule[0]
                if self._commit_time >= next_event.time:
                    yield self._schedule.pop(0)
                    continue
            except IndexError:
                break

            # Try to extend commitment
            for actor in self.actors:
                new_commit = actor.commit_to(next_event.time)
                if new_commit < self._commit_of[actor]:
                    raise SimulationError(self.now, "{} cannot lower commitment from {} to {}".format(actor, self._commit_of[actor], new_commit))
                self._commit_of[actor] = new_commit
            self._commit_time = min(self._commit_of.values())

            # Stop if no consensus to next event
            try:
                self._schedule = sorted(self._schedule, key=lambda e: e.time)
                if self._commit_time < self._schedule[0].time:
                    self.log("Can't get commitment to next event: {}".format(self._schedule[0]))
                    return
                else:
                    yield self._schedule.pop(0)
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
        if len(self.actors) > 0:
            self.log("Sending start notifications to Actors")
            for actor in self.actors:
                actor.start()
        self.log("Starting at time {} with {} scheduled events".format(self.now, len(self._schedule)))
        for event in self.take_next():
            self.now = event.time
            event(self.now)
        self.log("Done")

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

class LinearRuntime(object):
    """
    A simple linear model of the running time of a Task, with a slope
    and y-intercept.
    """
    def __init__(self, slope, intercept):
        self._slope = slope
        self._intercept = intercept

    def runtime_of(self, task):
        """Calculates the running time of a given Task"""
        return self._slope*task.scale + self._intercept

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
    def __init__(self, host_id, num_slots, runtime_model):
        super(Host, self).__init__()
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
            raise SimulationError(time, "Host is full; cannot run Task: {}".format(task))
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

class Master(Actor):
    """
    A Master starts with WorkUnits and Hosts.  It creates Tasks from
    the given WorkUnits and submits them to Hosts, trying to process
    all the WorkUnits.  A Host may or may not process the Task before
    returning it to the Master.
    """
    def __init__(self, initial_units, hosts):
        super(Master, self).__init__()
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
        """Selects a Host, and submits the given Task to it."""
        host = self.select_host(task)
        if host is None:
            raise SimulationError(time, "No Host for Task")
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
