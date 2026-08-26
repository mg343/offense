# 2d boids. each bird only looks at k nearest. click = goal, t = threat, r = reset, space = pause
#importing libraries
import math
import numpy as np
import pygame

#defining variables
num_boids = 150  # dense enough to read as a flock. num_boids = np.random.randint(40, 300)
screen_width = 1280  # common laptop window
screen_height = 800  # 16:10-ish with 1280
neighbor_count = 7  # birds only track ~6-7 neighbors
perception_radius = 90  # local view, not the whole flock
separation_radius = 22  # personal space so they don't collapse
max_speed = 4.4  # px/frame at 60fps, readable glide
min_speed = 2.2  # stop them stalling into a clump
max_force = 0.55  # turn radius v^2/a has to fit inside obstacle_pad or they plow through
weight_separation, weight_alignment, weight_cohesion = 1.75, 1.05, 0.95  # group hold is working, leave it
weight_goal, weight_obstacle, weight_threat, weight_wall = 0.22, 5.0, 5.0, 1.8  # hit = fail; obs out of the shared cap
obstacle_pad = 70  # start dodge sooner so seek can't line them into a circle
noise_std = 0.01  # 0.04 was snapping extrema headings
heading_keep = 0.6  # blend prior heading; extrema were flipping every frame
arrive_radius = 70  # close enough to count as reaching the goal
wall_margin = 48  # keep the flock off the window edge
threat_radius = 28  # big enough to split the volume. threat_radius = np.random.uniform(18, 40)

obstacles = [(520, 270, 68), (710, 530, 82), (900, 310, 52)]  # staggered so they split/squeeze. obstacles = [(np.random.uniform(300, screen_width-300), np.random.uniform(150, screen_height-150), np.random.uniform(40, 90)) for _ in range(3)]
start = np.array([140.0, screen_height / 2])  # left-side spawn. start = np.array([np.random.uniform(80, 220), np.random.uniform(120, screen_height-120)])
dest = np.array([screen_width - 140.0, screen_height / 2])  # right-side ferry. dest = np.array([np.random.uniform(screen_width-220, screen_width-80), np.random.uniform(120, screen_height-120)])
goal = dest.copy()

np.random.seed(7)  # repeatable. np.random.seed()
positions = start + np.random.randn(num_boids, 2) * 38  # * np.random.uniform(20, 60)
positions[:, 0] = np.clip(positions[:, 0], 40, 280)
velocities = np.array([3.0, 0.0]) + np.random.randn(num_boids, 2) * 0.55
threat = None  # x, y, vx, vy
paused = False


#steer toward a desired velocity
def steer(desired, velocity):
    n = np.linalg.norm(desired)
    if n < 1e-8:
        return np.zeros(2)
    return desired / n * max_speed - velocity


#clamp vector magnitudes
def cap(vecs, limit):
    mag = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs * np.where(mag > limit, limit / (mag + 1e-8), 1.0)


#repel + slide around a circle
def avoid(position, velocity, cx, cy, radius, pad=obstacle_pad):
    away = np.array([position[0] - cx, position[1] - cy])
    dist = np.hypot(away[0], away[1]) + 1e-8
    infl = radius + pad
    if dist >= infl:
        return np.zeros(2)
    normal = away / dist
    falloff = ((infl - dist) / infl) ** 2
    tangent = np.array([-normal[1], normal[0]])
    if np.dot(velocity, tangent) < 0:
        tangent = -tangent
    return normal * falloff * 1.55 + tangent * falloff * 0.85


#put the flock back at start
def reset():
    global positions, velocities, goal, threat
    positions = start + np.random.randn(num_boids, 2) * 38
    positions[:, 0] = np.clip(positions[:, 0], 40, 280)
    velocities = np.array([3.0, 0.0]) + np.random.randn(num_boids, 2) * 0.55
    goal = dest.copy()
    threat = None


#open the window
pygame.init()
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()
font = pygame.font.Font(None, 22)

running = True
while running:
    #handle input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            elif event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_r:
                reset()
            elif event.key == pygame.K_t:
                threat = [-40.0, float(np.random.uniform(180, screen_height - 180)), 6.8, 0.0]
        elif event.type == pygame.MOUSEBUTTONDOWN:
            goal = np.array(event.pos, dtype=float)

    if not paused:
        #nearest neighbors
        dist_sq = np.sum((positions[:, None, :] - positions[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(dist_sq, 1e18)
        nearest = np.argpartition(dist_sq, neighbor_count, axis=1)[:, :neighbor_count]
        acceleration = np.zeros_like(positions)
        dodge_acc = np.zeros_like(positions)
        see2, sep2 = perception_radius ** 2, separation_radius ** 2

        #local boid forces
        for i in range(num_boids):
            position, velocity = positions[i], velocities[i]
            separation = np.zeros(2)
            alignment = np.zeros(2)
            cohesion = np.zeros(2)
            n_sep = n_vis = 0
            for j in nearest[i]:
                if dist_sq[i, j] >= see2:
                    continue
                delta = position - positions[j]
                n_vis += 1
                alignment += velocities[j]
                cohesion += positions[j]
                if 1e-12 < dist_sq[i, j] < sep2:
                    separation += delta / (math.sqrt(dist_sq[i, j]) + 1e-8)
                    n_sep += 1
            force = np.zeros(2)
            if n_sep:
                force += weight_separation * steer(separation / n_sep, velocity)
            if n_vis:
                force += weight_alignment * steer(alignment / n_vis, velocity)
                force += weight_cohesion * steer(cohesion / n_vis - position, velocity)
            #dodge is death-priority: own budget, and no seek while in the pad
            dodge = np.zeros(2)
            for obst_x, obst_y, obst_r in obstacles:
                dodge += weight_obstacle * avoid(position, velocity, obst_x, obst_y, obst_r)
            if threat is not None:
                dodge += weight_threat * avoid(position, velocity, threat[0], threat[1], threat_radius)
            if np.linalg.norm(dodge) < 1e-6:
                force += weight_goal * steer(goal - position, velocity)
            dodge_acc[i] = dodge
            #push off window edges
            if position[0] < wall_margin:
                force[0] += weight_wall * (wall_margin - position[0]) / wall_margin
            elif position[0] > screen_width - wall_margin:
                force[0] -= weight_wall * (position[0] - (screen_width - wall_margin)) / wall_margin
            if position[1] < wall_margin:
                force[1] += weight_wall * (wall_margin - position[1]) / wall_margin
            elif position[1] > screen_height - wall_margin:
                force[1] -= weight_wall * (position[1] - (screen_height - wall_margin)) / wall_margin
            acceleration[i] = force + np.random.randn(2) * noise_std

        #integrate
        acceleration = cap(acceleration, max_force) + cap(dodge_acc, max_force)
        new_vel = cap(velocities + acceleration, max_speed)
        velocities = cap(heading_keep * velocities + (1.0 - heading_keep) * new_vel, max_speed)
        speed = np.linalg.norm(velocities, axis=1, keepdims=True)
        velocities = np.where(speed < min_speed, velocities * min_speed / (speed + 1e-8), velocities)
        positions = positions + velocities
        positions[:, 0] = np.clip(positions[:, 0], 8, screen_width - 8)
        positions[:, 1] = np.clip(positions[:, 1], 8, screen_height - 8)

        #move threat off-screen then drop it
        if threat is not None:
            threat[0] += threat[2]
            threat[1] += threat[3]
            if threat[0] > screen_width + 60:
                threat = None

        #ferry back the other way once we arrive
        if np.linalg.norm(positions.mean(0) - goal) < arrive_radius:
            goal = start.copy() if np.linalg.norm(goal - dest) < 1 else dest.copy()

    #draw
    screen.fill((11, 14, 24))
    for obst_x, obst_y, obst_r in obstacles:
        pygame.draw.circle(screen, (22, 28, 42), (int(obst_x), int(obst_y)), int(obst_r))
        pygame.draw.circle(screen, (58, 72, 98), (int(obst_x), int(obst_y)), int(obst_r), 2)
    pygame.draw.circle(screen, (232, 176, 72), (int(goal[0]), int(goal[1])), 8, 2)
    if threat is not None:
        pygame.draw.circle(screen, (220, 72, 72), (int(threat[0]), int(threat[1])), threat_radius, 2)
    for i in range(num_boids):
        heading = math.atan2(velocities[i, 1], velocities[i, 0])
        x, y = positions[i]
        tip = (x + math.cos(heading) * 7, y + math.sin(heading) * 7)
        left = (x + math.cos(heading + 2.55) * 4.5, y + math.sin(heading + 2.55) * 4.5)
        right = (x + math.cos(heading - 2.55) * 4.5, y + math.sin(heading - 2.55) * 4.5)
        pygame.draw.polygon(screen, (214, 226, 245), [tip, left, right])
    screen.blit(font.render("click goal  t threat  r reset  space pause", True, (170, 180, 200)), (14, 12))
    pygame.display.flip()
    clock.tick(60)  # 60fps is the usual desktop sim rate

pygame.quit()