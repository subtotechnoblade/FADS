import pygame


class Transition:
    def __init__(self, screen, true_resolution):
        self.screen = screen
        self.true_resolution = true_resolution
        self.fade_surf = pygame.Surface(self.true_resolution, pygame.SRCALPHA)
        self.alpha = 0
    def Update(self, new_alpha):
        self.alpha = new_alpha
        self.alpha = min(self.alpha, 255)
        self.fade_surf.set_alpha(self.alpha)

    def Update_Inc(self, da=5):
        self.alpha += da
        self.alpha = min(self.alpha, 255)
        self.fade_surf.set_alpha(self.alpha)
        if self.alpha == 0 and da < 0:
            return True
        elif self.alpha == 255 and da > 0:
            return True

    def Draw(self):
        self.screen.blit(self.fade_surf, (0, 0))


class Fade_Transition(Transition):
    def __init__(self, screen, true_resolution):
        super().__init__(screen, true_resolution)
        self.fade_surf.fill((0, 0, 0))


if __name__ == "__main__":
    import os
    import ctypes

    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pygame.init()
    ctypes.windll.user32.SetProcessDPIAware()
    true_res = (ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1))
    screen = pygame.display.set_mode(true_res,
                                     pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.HWACCEL | pygame.FULLSCREEN,
                                     vsync=1)

    fade_transition = Fade_Transition(screen, true_res)
    running = True
    while running:
        pygame_events = pygame.event.get()
        screen.fill((0, 255, 255))
        fade_transition.Update_Inc()
        fade_transition.Draw()

        pygame.display.flip()
        for event in pygame_events:
            if event.type == pygame.QUIT:
                running = False
