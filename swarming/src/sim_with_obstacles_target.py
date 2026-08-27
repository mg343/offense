# inspired by https://www.youtube.com/watch?v=4LWmRuB-uNU&pp=ygUMbXVybXVyYXRpb25z
import math
import random
import pygame

WIDTH, HEIGHT = 1280, 800
NUM_BIRDS = 150
obstacle_radius = 50

class Bird:

    cohesion = 0.005
    bird_seperation = 0.7  # personal space is logically high
    bird_seperation_radius = 25.0
    obstacle_seperation_tolerance = 1  # hitting an object is worse than crowding a neighbor
    obstacle_seperation_radius = 5 * bird_seperation_radius  # flinch much earlier than bird-bird space
    alignment = 0.05
    view = 200.0
    max_speed = 4.0
    heading_keep = 0.7  # less work to keep current heading
    seek = 1  # strength of steering toward target
    seek_arrive = 0.25 * bird_seperation_radius  # would rather err on keeping the swarm close to the target than letting it get distracted
    target = (random.uniform(80, WIDTH - 80), random.uniform(80, HEIGHT - 80))  # loft each bird already knows

    def __init__(self, x, y):
        self.x = x
        self.y = y
        heading = random.uniform(0, 2 * math.pi)
        self.vx = math.cos(heading) * self.max_speed
        self.vy = math.sin(heading) * self.max_speed

    def see(self, flock, obstacles):
        nearby = []
        for other in flock:
            if other is self:
                continue
            if math.hypot(other.x - self.x, other.y - self.y) < self.view:
                nearby.append(other)

        seen_obstacles = []
        for ox, oy in obstacles:
            if math.hypot(self.x - ox, self.y - oy) - obstacle_radius < self.view:
                seen_obstacles.append((ox, oy))
        return nearby, seen_obstacles

    def cohesion_update(self, nearby):
        if not nearby:
            return
        cx = sum(b.x for b in nearby) / len(nearby)
        cy = sum(b.y for b in nearby) / len(nearby)
        self.vx += (cx - self.x) * self.cohesion
        self.vy += (cy - self.y) * self.cohesion

    def seperation_update(self, nearby, seen_obstacles):
        for other in nearby:
            dx = self.x - other.x
            dy = self.y - other.y
            dist = math.hypot(dx, dy)
            if 0 < dist < self.bird_seperation_radius:
                self.vx += dx / dist * self.bird_seperation
                self.vy += dy / dist * self.bird_seperation

        for ox, oy in seen_obstacles:
            dx, dy = self.x - ox, self.y - oy
            dist = math.hypot(dx, dy) or 1e-6
            if dist - obstacle_radius < self.obstacle_seperation_radius:
                self.vx += dx / dist * self.obstacle_seperation_tolerance
                self.vy += dy / dist * self.obstacle_seperation_tolerance

    def seek_update(self, seen_obstacles):
        if seen_obstacles:
            return
        dx = self.target[0] - self.x
        dy = self.target[1] - self.y
        dist = math.hypot(dx, dy)
        if dist < self.seek_arrive:
            return
        self.vx += dx / dist * self.seek
        self.vy += dy / dist * self.seek

    def alignment_update(self, nearby):
        if not nearby:
            return
        mvx = sum(b.vx for b in nearby) / len(nearby)
        mvy = sum(b.vy for b in nearby) / len(nearby)
        self.vx += (mvx - self.vx) * self.alignment
        self.vy += (mvy - self.vy) * self.alignment

    def update(self, flock, obstacles):
        old_vx, old_vy = self.vx, self.vy
        nearby, seen_obstacles = self.see(flock, obstacles)
        self.cohesion_update(nearby)
        self.seperation_update(nearby, seen_obstacles)
        self.alignment_update(nearby)
        self.seek_update(seen_obstacles)
        speed = math.hypot(self.vx, self.vy)
        if speed > self.max_speed:
            self.vx = self.vx / speed * self.max_speed
            self.vy = self.vy / speed * self.max_speed
        k = self.heading_keep
        self.vx = k * old_vx + (1 - k) * self.vx
        self.vy = k * old_vy + (1 - k) * self.vy
        self.x += self.vx
        self.y += self.vy
        if self.x < 0 or self.x > WIDTH:
            self.vx *= -1
            self.x = max(0, min(WIDTH, self.x))
        if self.y < 0 or self.y > HEIGHT:
            self.vy *= -1
            self.y = max(0, min(HEIGHT, self.y))
            
        for ox, oy in obstacles:
            dx, dy = self.x - ox, self.y - oy
            dist = math.hypot(dx, dy) or 1e-6
            if dist < obstacle_radius:
                nx, ny = dx / dist, dy / dist
                self.x = ox + nx * obstacle_radius
                self.y = oy + ny * obstacle_radius
                inward = self.vx * nx + self.vy * ny
                if inward < 0:
                    self.vx -= 2 * inward * nx
                    self.vy -= 2 * inward * ny

# pygame start
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# spawn birds
center_x, center_y = WIDTH / 2, HEIGHT / 2
radius = HEIGHT / 4
birds = []
for i in range(NUM_BIRDS):
    angle = 2 * math.pi * i / NUM_BIRDS
    x = center_x + radius * math.cos(angle)
    y = center_y + radius * math.sin(angle)
    birds.append(Bird(x, y))

obstacles = []  # left click to place (x, y)
# right click to reposition target

running = True
while running:
    # pygame render
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                obstacles.append(event.pos)
            elif event.button == 3:
                Bird.target = event.pos

    # loop
    for bird in birds:
        bird.update(birds, obstacles)

    # graphics
    screen.fill((11, 14, 24))
    for ox, oy in obstacles:
        pygame.draw.circle(screen, (22, 28, 42), (int(ox), int(oy)), obstacle_radius)
        pygame.draw.circle(screen, (58, 72, 98), (int(ox), int(oy)), obstacle_radius, 2)
    pygame.draw.circle(screen, (232, 176, 72), (int(Bird.target[0]), int(Bird.target[1])), 8, 2)
    for bird in birds:
        heading = math.atan2(bird.vy, bird.vx)
        tip = (bird.x + math.cos(heading) * 8, bird.y + math.sin(heading) * 8)
        left = (bird.x + math.cos(heading + 2.5) * 5, bird.y + math.sin(heading + 2.5) * 5)
        right = (bird.x + math.cos(heading - 2.5) * 5, bird.y + math.sin(heading - 2.5) * 5)
        pygame.draw.polygon(screen, (214, 226, 245), [tip, left, right])
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
