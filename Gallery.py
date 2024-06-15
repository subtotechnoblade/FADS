import pygame
import numpy as np
from glob import glob
from Linked_List import Linked_List


class Update_Button:
    def __init__(self, screen, pos, image_path):
        self.screen = screen
        self.pos = pos
        self.rect = pygame.Rect(self.pos)
        self.border = pygame.Rect(self.pos[0] - 3,
                                  self.pos[1] - 3,
                                  self.pos[2] + 2 * 3,
                                  self.pos[3] + 2 * 3)
        self.sprite = pygame.image.load(image_path).convert_alpha()
        self.is_mouse_collided = False

    def Collided_With_Mouse(self, mouse_x, mouse_y):
        self.is_mouse_collided = False
        if is_collided := self.rect.collidepoint(mouse_x, mouse_y):
            self.is_mouse_collided = True
        return is_collided

    def Render_Button(self, hover_color=(255, 255, 255), width=3):
        if self.is_mouse_collided:
            pygame.draw.rect(self.screen,
                             hover_color,
                             self.border)
        self.screen.blit(self.sprite, self.pos[:2])

class Gallery:
    def __init__(self, screen, screen_resolution, folder_path, shape, tile_size=5):
        self.screen = screen
        self.pointer = 0
        # self.frame = pygame.image.load("Data/Backgrounds/frame.png").convert_alpha()
        self.gallery_images = []
        self.folder_path = folder_path
        self.Load()

        self.tile_size = tile_size
        self.gallery_surf = pygame.Surface((shape[1] * self.tile_size, shape[0] * self.tile_size))
        if self.gallery_images:
            pygame.surfarray.blit_array(self.gallery_surf, self.Scale_Image(self.gallery_images[self.pointer]))
        else:
            pygame.surfarray.blit_array(self.gallery_surf,
                                        np.ones((shape[1] * self.tile_size, shape[0] * self.tile_size, 3)) * 255)
        self.starting_pos = (
        (screen_resolution[0] - shape[1] * self.tile_size) / 2, (screen_resolution[1] - shape[0] * self.tile_size) / 2)
        self.update_button = Update_Button(screen, (870, 900, 160, 80), "Data/Icons/update.png")
    def Load(self):
        self.gallery_images = []
        files = glob(f"{self.folder_path}/*.npz")
        print(len(files))
        self.gallery_images = [0] * len(files)
        for i, path in enumerate(files):
            img = np.load(path, allow_pickle=False)["inputs"]
            # Note this is not transposed, and it is in formal (x, y)
            self.gallery_images[i] = img

    def Set_Canvas(self, canvas):
        self.canvas = canvas

    def Set_Canvas_State(self):
        self.canvas.Load_Internal(f"Paintings/{self.pointer}.npz")

    def Scale_Image(self, arr):
        return np.repeat(np.repeat(arr, self.tile_size, axis=0), self.tile_size, axis=1)

    def Update(self, mouse_x, mouse_y, pygame_events):
        keys = pygame.key.get_pressed()
        mouse_pressed = pygame.mouse.get_pressed()
        for event in pygame_events:
            if event.type == pygame.KEYDOWN:
                if keys[pygame.K_LEFT]:
                    self.pointer -= 1
                elif keys[pygame.K_RIGHT]:
                    self.pointer += 1

            elif event.type == pygame.MOUSEBUTTONDOWN and not self.update_button.Collided_With_Mouse(mouse_x, mouse_y):
                if mouse_pressed[0]:
                    self.pointer -= 1
                elif mouse_pressed[2]:
                    self.pointer += 1
            self.pointer = min(max(self.pointer, 0), len(self.gallery_images) - 1)
            self.pointer = self.pointer if self.pointer < len(self.gallery_images) else len(self.gallery_images) - 1

            if self.update_button.Collided_With_Mouse(mouse_x, mouse_y) and event.type == pygame.MOUSEBUTTONDOWN:
                self.Set_Canvas_State()
                self.canvas.Update(mouse_x, mouse_y, pygame_events)
                return "Canvas"
        self.pointer = min(max(self.pointer, 0), len(self.gallery_images) - 1)
        # if we have changed the pointer, then we update the surface
        if self.gallery_images:
            pygame.surfarray.blit_array(self.gallery_surf, self.Scale_Image(self.gallery_images[self.pointer]))
        # update the gallery surf here
        return "Gallery"

    def Draw(self):
        # self.screen.blit(self.frame, self.starting_pos)
        self.screen.blit(self.gallery_surf, self.starting_pos)
        self.update_button.Render_Button((0, 134, 223))
        # self.screen.blit()


if __name__ == "__main__":
    import os
    import ctypes

    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pygame.init()
    ctypes.windll.user32.SetProcessDPIAware()
    true_res = (ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1))
    screen = pygame.display.set_mode((1920, 1080),
                                     pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.HWACCEL | pygame.RESIZABLE,
                                     vsync=1)
    gallery = Gallery(screen, (1920, 1080), f"Paintings", (120, 210))
    running = True
    while running:
        pygame_events = pygame.event.get()
        mouse_x, mouse_y = pygame.mouse.get_pos()
        gallery.Update(mouse_x, mouse_y, pygame_events)
        gallery.Draw()

        pygame.display.flip()
        for event in pygame_events:
            if event.type == pygame.QUIT:
                running = False
