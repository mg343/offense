# boids policy notes
working log of what I notice during Boids policy trials.

## trial 1 policy

knobs (main.py, post alias-fix + fmax bump):
- n=150  k=7  see=90px  sep_r=22px
- vmax=4.4  vmin=2.2  fmax=0.55  (~60fps)
- w: sep=1.75  ali=1.05  coh=0.95  |  goal=0.42  obs=2.4  threat=3.1  wall=1.8
- obs_pad=55  arrive_r=70  wall_m=48
- noise=0.04  force = sum(w*steer) then cap to fmax  → weights only buy direction, mag always ≤0.55
- obs: (520,270,r68), (710,530,r82), (900,310,r52)

what it is: reynolds sep/ali/coh on k-NN, plus global seek + radial/tangent avoid. no health, no hit flag. group hold is the thing that’s working.

## post-trial 1 observations

extrema / heading
- units in the extrema of the constellation lag on heading correction, or else snap heading at very high intervals with high variability
- shouldn’t: they already have a target (goal=0.42 every frame). look at: 1-sided k-NN (all 7 neighbors inward), vmin locking a leftover heading, noise=0.04 vs fmax=0.55

obstacles vs goal — needs a reweighting
- currently biased towards getting into a target position → units completely disregarding obstacles (plow / clip circles)
- paper weights obs=2.4 vs goal=0.42 but post-cap the seek still wins; obs is not treated as hard fail
- must reweight: target nav  vs  stick-with-group (doing well)  vs  obstacle avoidance (underweighted)
- analog for later: this "could manifest itself in a future RL policy" as hit / death / health — hitting an obstacle is the ultimate failure. no such metric in trial 1

## trial 2 policy

knobs:
- n=150  k=7  see=90px  sep_r=22px
- vmax=4.4  vmin=2.2  fmax=0.55  (~60fps)
- w: sep=1.75  ali=1.05  coh=0.95  |  goal=0.22  obs=5.0  threat=5.0  wall=1.8
- obs_pad=70  arrive_r=70  wall_m=48
- noise=0.01  heading_keep=0.6
- force = cap(flock+goal+wall, fmax) + cap(dodge, fmax)  → dodge has its own 0.55, seek off while dodge>0
- obs: same (520,270,r68), (710,530,r82), (900,310,r52)

vs t1: group w unchanged. goal 0.42→0.22, obs 2.4→5.0, threat 3.1→5.0, pad 55→70, noise 0.04→0.01. dodge no longer shares the cap with seek. heading blend for extrema chatter.

## post-trial 2 observations

static obs — reunite works
- preplanned circles: very few units near contact. split/rejoin around the negative of the obstacle is doing the job
- T-threat does not get the same behavior. threat vx=6.8 vs vmax=4.4 (~1.55x). infl=r28+pad70=98px. closing ~6.8, dwell in pad ~14 frames vs static approach ~vmax 3 / pad 70 → ~23 frames. not enough time to carve
- immediate: slow T (try ~vmax, e.g. 3–4). not a different policy — same avoid() already. moving vs still / large vs small should be one law (deform to the negative of the body), not a special case for T. next: scale pad/force with closing speed and radius so the same rule covers both

cohesion / splinters
- small groups peel off and stay gone too long before reconvening. coh=0.95 vs sep=1.75, k=7 see=90: a splinter of ≥8 is a complete neighborhood, local coh never sees the main body
- naive: bump coh. real hole may be range not weight — k-NN has no flock-level pull. look at coh vs sep ratio and/or see, not coh in isolation

scale / don’t overfit this window
- tuning discrete px knobs (see=90, pad=70, w=5.0) to 1280x800 will not transfer to drones or another policy
- want relationships not values: sep_r/see, w_coh/w_sep, pad/r_obst, v_obst/vmax, maybe body-length + time-to-collision for moving bodies
- nature: topological k already scales; metric radii and raw weights don’t. avoid baking another example-specific stack of numbers

## trial 3 policy

knobs:
- n=150  k=7  body=sep_r=22  see=5*body=110
- vmax=4.4  vmin=2.2  fmax=0.55
- w: sep=1.75  ali=1.05  coh=0.7*sep=1.225  flock=0.2*sep=0.35  |  goal=0.22  obs=5.0  threat=5.0  wall=1.8
- pad = 1.0*r + 18*closing  (closing = approach speed vs body, 0 if still/receding)
- threat r=1.25*body=27.5  v_obst=vmax=4.4
- noise=0.01  heading_keep=0.6  arrive_r=70  wall_m=48
- dodge cap unchanged: cap(flock+goal+wall)+cap(dodge)

vs t2: T 6.8→4.4 (v_obst/vmax=1). same avoid() for still/moving — pad from r + ttc not a T special case. splinters: flock-mean term (k-NN can’t see past a pack of ≥8) + coh/sep=0.7 + see 90→110. radii as multiples of body.

## post-trial 3 observations



