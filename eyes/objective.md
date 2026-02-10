### Background
In the current state of covert military operations, a heavy emphasis is being placed on the usage of kamikaze drones, or drones that are commissioned with the express purpose of blowing up/delivering a payload.

### Overall Issue
When drones are released to reach a target, accuracy is not ensured. Highest levels of accuracy can be achieved when data on drone location is transmitted to a ground station monitor, but this is faulty: captured data could render the mission useless and resources wasted, or even worse, in the hands of the enemy.

### Solution (So-Far)
Drones are programmed before launch with hyper-specific instructions - a certain angle of launch is taken, the drone releases it's payload after a certain amount of distance is covered, a mechanism is released after a set amount of time, etc.

### Why This Fails
The hyper paramaterization of these drones makes these weapons entirely ineffective when any sort of obstacle is identified. If the drone has a faulty motor, the payload releases early. More obviously, if weather conditions are bad or anything less than ideal, the confidence of a successful mission drops dramatically.

### Eyes
Eyes is an onboard, perception-based validation system that allows autonomous drones to confirm whether they are in the correct location without transmitting explicit positional data. Instead of relying on rigid, pre-programmed distance, timing, or motor-usage heuristics, Eyes leverages onboard sensing to compare what the drone observes in real time against preloaded environmental priors (e.g., satellite imagery, terrain features, or structural layouts). By shifting from hyper-parameterized execution to perceptual confirmation, Eyes enables drones to adapt to drift, weather, and minor system failures while maintaining operational security. The system operates entirely on the vehicle, requiring no external communication during flight. Eyes replaces the lengthy assumptions typically given to kamikaze drones with situational awareness, allowing these systems to verify where they are, not just assume they arrived or comply with faulty hardcoded conditions.