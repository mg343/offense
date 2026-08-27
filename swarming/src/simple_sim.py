# inspired by https://www.youtube.com/watch?v=4LWmRuB-uNU&pp=ygUMbXVybXVyYXRpb25z
import math
import random
import pygame

WIDTH, HEIGHT = 800, 600
NUM_BIRDS = 150

class Bird:
    cohesion = 0.005
    seperation = 0.7 #personal space is logically high
    alignment = 0.05
    seperation_radius = 25.0
    view = 200.0
    max_speed = 4.0
    heading_keep = 0.7 #less work to keep current heading

    def __init__(self, x, y):
        self.x = x
        self.y = y
        heading = random.uniform(0, 2 * math.pi)
        self.vx = math.cos(heading) * self.max_speed
        self.vy = math.sin(heading) * self.max_speed

    def see(self, flock):
        nearby = []
        for other in flock:
            if other is self:
                continue
            if math.hypot(other.x - self.x, other.y - self.y) < self.view:
                nearby.append(other)
        return nearby

    def cohesion_update(self, nearby):
        if not nearby:
            return
        cx = sum(b.x for b in nearby) / len(nearby)
        cy = sum(b.y for b in nearby) / len(nearby)
        self.vx += (cx - self.x) * self.cohesion
        self.vy += (cy - self.y) * self.cohesion

    def seperation_update(self, nearby):
        for other in nearby:
            dx = self.x - other.x
            dy = self.y - other.y
            dist = math.hypot(dx, dy)
            if 0 < dist < self.seperation_radius:
                self.vx += dx / dist * self.seperation
                self.vy += dy / dist * self.seperation

    def alignment_update(self, nearby):
        if not nearby:
            return
        mvx = sum(b.vx for b in nearby) / len(nearby)
        mvy = sum(b.vy for b in nearby) / len(nearby)
        self.vx += (mvx - self.vx) * self.alignment
        self.vy += (mvy - self.vy) * self.alignment

    def update(self, flock):
        old_vx, old_vy = self.vx, self.vy
        nearby = self.see(flock)
        self.cohesion_update(nearby)
        self.seperation_update(nearby)
        self.alignment_update(nearby)
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

running = True
while running:
    # pygame render
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # loop
    for bird in birds:
        bird.update(birds)

    # graphics
    screen.fill((11, 14, 24))
    for bird in birds:
        heading = math.atan2(bird.vy, bird.vx)
        tip = (bird.x + math.cos(heading) * 8, bird.y + math.sin(heading) * 8)
        left = (bird.x + math.cos(heading + 2.5) * 5, bird.y + math.sin(heading + 2.5) * 5)
        right = (bird.x + math.cos(heading - 2.5) * 5, bird.y + math.sin(heading - 2.5) * 5)
        pygame.draw.polygon(screen, (214, 226, 245), [tip, left, right])
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
