import pygame


class Controls:
    def __init__(self, screen, controls_image1, controls_image2):
        self.screen = screen
        self.images = [pygame.image.load(controls_image1).convert_alpha(),
                       pygame.image.load(controls_image2).convert_alpha()]
        self.pointer = 0

    def Update(self, mouse_x, mouse_y, pygame_events):
        keys = pygame.key.get_pressed()
        for event in pygame_events:
            if event.type == pygame.KEYDOWN:
                if keys[pygame.K_LEFT] and self.pointer == 1:
                    self.pointer = 0
                elif keys[pygame.K_RIGHT] and self.pointer == 0:
                    self.pointer = 1
        return "Controls"

    def Draw(self):
        self.screen.blit(self.images[self.pointer], (0, 0))
