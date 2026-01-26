import pygame
import requests

pygame.init()

WIDTH, HEIGHT = 500, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gesture Car Game")

clock = pygame.time.Clock()

car_x = 220
car_y = 500

boost = 0

def get_gesture():
    try:
        r = requests.get("http://localhost:5000/gesture", timeout=0.1)
        return r.json()["gesture"]
    except:
        return "none"

font = pygame.font.SysFont(None, 36)

running = True

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # CLEAR SCREEN
    screen.fill((40,40,40))

    gesture = get_gesture()

    # SHOW TEXT
    txt = font.render("Gesture: " + gesture, True, (255,255,255))
    screen.blit(txt,(20,20))

    # CONTROLS
    if gesture == "circle":
        car_x += 5

    if gesture == "updown":
        car_y -= 5

    if gesture == "shake":
        boost = 20

    if boost > 0:
        car_y -= 10
        boost -= 1

    # DRAW CAR
    pygame.draw.rect(screen,(255,0,0),(car_x,car_y,60,100))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
