import pygame
from glob import glob

class Credits:
    def __init__(self, screen, credits_image_folder):
        self.screen = screen
        self.images = [pygame.transform.scale(pygame.image.load(image_path).convert_alpha(), (1920, 1080)) for image_path in glob(credits_image_folder)]
        self.pointer = 0

    def Update(self, mouse_x, mouse_y, pygame_events):
        keys = pygame.key.get_pressed()
        for event in pygame_events:
            if event.type == pygame.KEYDOWN:
                if keys[pygame.K_LEFT] and self.pointer > 0:
                    self.pointer -= 1
                elif keys[pygame.K_RIGHT] and self.pointer < 4:
                    self.pointer += 1

        return "Credits"

    def Draw(self):
        self.screen.blit(self.images[self.pointer], (0, 0))
