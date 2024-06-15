import os
import pygame
import time
import random
import numpy as np
from glob import glob

class Button:
    def __init__(self, screen, pos, hover_image_path=None, image_path=None, border_width=5):
        self.screen = screen
        # pos included the size
        self.pos = pos
        self.rect = pygame.Rect(self.pos)
        self.border_width = border_width

        self.hover_image = None
        if hover_image_path:
            self.hover_image = pygame.transform.scale(pygame.image.load(hover_image_path).convert_alpha(), (300, 100))
        self.image = None
        if image_path is not None:
            self.image = pygame.transform.scale(pygame.image.load(image_path).convert_alpha(), (300, 100))
        self.is_mouse_collided = False

    def Update_Pos(self, dx, dy):
        self.pos[0] += dx
        self.pos[1] += dy
        self.rect.update(self.pos)

    def Collided_With_Mouse(self, mouse_x, mouse_y):
        self.is_mouse_collided = False
        if is_collided := self.rect.collidepoint(mouse_x, mouse_y):
            self.is_mouse_collided = True
        return is_collided

    def Render_Button(self, hover_color=(255, 255, 255)):
        if self.is_mouse_collided:
            # render the hover image here
            if self.hover_image is not None:
                self.screen.blit(self.hover_image, self.pos[:2])
            else:
                pygame.draw.rect(self.screen, hover_color,
                                 (self.pos[0] - self.border_width, self.pos[1] - self.border_width,
                                  self.pos[2] + 2 * self.border_width, self.pos[3] + 2 * self.border_width))
        if self.image is not None:
            if self.hover_image is not None and not self.is_mouse_collided:
                self.screen.blit(self.image, self.pos[:2])
            elif self.hover_image is None and self.image is not None:
                self.screen.blit(self.image, self.pos[:2])


class Main_Menu:
    def __init__(self, screen, screen_resolution):
        self.screen = screen
        self.screen_size = np.array(screen_resolution, dtype=np.uint16)

        self.pressed = None
        start_x = 270
        y_shift = 270
        self.play_button = Button(self.screen,
                                  (start_x, y_shift, 300, 100),
                                  "Data/Start_Menu/play_button_hover.png",
                                  "Data/Start_Menu/play_button.png")
        self.gallery_button = Button(self.screen,
                                     (start_x, y_shift + 150, 300, 100),
                                     image_path="Data/Start_Menu/gallery_button.png")
        self.controls_button = Button(self.screen,
                                      (start_x, y_shift + 300, 300, 100),
                                      image_path="Data/Start_Menu/control_button.png")
        self.credits_button = Button(self.screen,
                                     (start_x, y_shift+  450, 300, 100),
                                     image_path="Data/Start_Menu/credits_button.png"
                                     )
        self.art_images = [pygame.image.load(path) for path in glob("Data/Start_Menu/Art/*.png")]
        if os.path.exists("Images/"):
            self.art_images += [pygame.image.load(path) for path in glob("Images/*.png")]
        self.pointer = random.randrange(0, len(self.art_images))
        self.start_time = time.time()

    def Update(self, mouse_x, mouse_y, pygame_events):

        if time.time() - self.start_time >= 5:
            self.pointer = random.randrange(0, len(self.art_images), 1)
            self.start_time = time.time()
        mouse_down = False
        for event in pygame_events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_down = True
        if self.play_button.Collided_With_Mouse(mouse_x, mouse_y) and mouse_down:
            return "Canvas"
        elif self.gallery_button.Collided_With_Mouse(mouse_x, mouse_y) and mouse_down:
            return "Gallery"
        elif self.controls_button.Collided_With_Mouse(mouse_x, mouse_y) and mouse_down:
            return "Controls"
        elif self.credits_button.Collided_With_Mouse(mouse_x, mouse_y) and mouse_down:
            return "Credits"
        return "Main_Menu"

    def Draw(self):
        self.play_button.Render_Button((100, 60, 255))
        self.gallery_button.Render_Button((100, 60, 255))
        self.controls_button.Render_Button((100, 60, 255))
        self.credits_button.Render_Button((100, 60, 255))

        self.screen.blit(self.art_images[self.pointer], (800, 250))


if __name__ == "__main__":
    import os
    import ctypes

    os.environ["SDL_VIDEO_CENTERED"] = "1"

    pygame.init()
    ctypes.windll.user32.SetProcessDPIAware()
    true_res = (ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1) - 50)
    screen = pygame.display.set_mode((1920, 1080),
                                     pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.HWACCEL | pygame.RESIZABLE,
                                     vsync=1)
    main_menu = Main_Menu(screen, true_res)

    running = True
    while running:
        pygame_events = pygame.event.get()
        mouse_pos = pygame.mouse.get_pos()

        screen.fill((255, 255, 255))
        if (x := main_menu.Update(*mouse_pos, pygame_events)) != "Main_Menu":
            print(x)
        main_menu.Draw()

        for event in pygame_events:
            if event.type == pygame.QUIT:
                running = False
        pygame.display.flip()
