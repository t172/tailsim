# tailsim
A discrete event simulator for modeling running times of a workload in an opportunistic batch system.  This was originally intended to model the long tail behavior of Lobster workflows using WorkQueue, with variability in host performance, host eviction rates, and individual task sizes.

A `Master` starts with a pool of predefined `WorkUnit`s, from which it creates `Task`s.  These are dispatched to `Host`s when available.  The `Host` runs the `Task`s, and "evicts" randomly, using a Poisson process.  Performance (the time to process `WorkUnit`s) and eviction rate are intrinsic `Host` attributes.  `Task`s are returned to the `Master` as completed or uncompleted.  The `WorkUnit`s of uncompleted `Task`s are returned to the `Master`'s unprocessed pool to be redispatched.  The simulation ends when all `WorkUnit`s are completed or there are no more simulation events scheduled.

This work was a done by Bryan Harris as a part of the Data Intensive Scientific Computing (DISC) REU at Notre Dame in the summer of 2017.  See https://disc.crc.nd.edu/index.php/disc-summer-2017
