if __name__ == "__main__":

    import os
    import pygame


    class Back_Button:
        def __init__(self, screen, pos, image_path=None):
            self.screen = screen
            self.pos = pos
            self.image = pygame.transform.scale(pygame.image.load(image_path).convert_alpha(), self.pos[2:])
            self.rect = pygame.Rect(pos)

            self.is_mouse_collided = False

        def Collided_With_Mouse(self, mouse_x, mouse_y):
            self.is_mouse_collided = False
            if is_collided := self.rect.collidepoint(mouse_x, mouse_y):
                self.is_mouse_collided = True
            return is_collided

        def Update(self, mouse_x, mouse_y):
            self.Collided_With_Mouse(mouse_x, mouse_y)

        def Render_Button(self):
            if self.is_mouse_collided:
                pygame.draw.rect(self.screen, (100, 100, 255),
                                 (self.rect[0] - 3, self.rect[1] - 3, self.rect[2] + 6, self.rect[3] + 6))
            # pygame.draw.rect(self.screen, (100, 200, 0), self.pos)
            self.screen.blit(self.image, self.pos[:2])


    def Canvas_Init(screen, connection):
        saved_folder_path = "Save"
        os.makedirs(saved_folder_path, exist_ok=True)

        save_path = "Paintings"
        os.makedirs(save_path, exist_ok=True)

        image_folder_path = "Images"
        os.makedirs(image_folder_path, exist_ok=True)

        return Canvas(screen=screen,
                      start_pos=(50, 100),
                      shape=(120, 210),
                      tile_size=5,
                      brush_radius=20,
                      saved_folder_path=saved_folder_path,
                      pipe_connection=connection)


    from Transition import Fade_Transition
    from Load_Screen import Run_Loading_Screen
    from Start_Menu import Main_Menu
    from Canvas import Canvas
    from Gallery import Gallery
    from Controls import Controls
    from Credits import Credits

    os.environ["SDL_VIDEO_CENTERED"] = "1"
    import ctypes
    import multiprocessing as mp

    if os.path.exists("Data/tmp/signal.txt"):
        os.remove("Data/tmp/signal.txt")
    if os.path.exists("Data/tmp/downloaded_signal.txt"):
        os.remove("Data/tmp/downloaded_signal.txt")

    # connection1, connection2 = mp.Pipe(duplex=True)
    # Note true dimensions are 1050 by 600

    if not Run_Loading_Screen(None):
        raise RuntimeError(f"Exited while warming up, everything is normal")

    ctypes.windll.user32.SetProcessDPIAware()
    # true_res = (ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1))
    true_res = 1920, 1080
    pygame.init()
    screen = pygame.display.set_mode((true_res[0], true_res[1] - 1),
                                     pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.HWACCEL | pygame.RESIZABLE,
                                     vsync=1)

    pygame.display.set_caption("Fine Arts Drawing Simulator")
    pygame_icon = pygame.transform.scale(pygame.image.load("Data/Icons/FADS_icon.png").convert_alpha(), (32, 32))

    pygame.display.set_icon(pygame_icon)
    pygame.mouse.set_visible(True)

    background_image = pygame.image.load("Data/Backgrounds/nebularesized.png").convert_alpha()
    state = "Main_Menu"
    changed_state = state
    # if not os.path.exists("Paintings_b/"):  # For testing, test this at the end. Cuz I'm too lazy
    #     tutorial = True
    #     state = "Tutorial"

    # menu = Main_Menu(screen, true_res)
    # initialization here
    transitioning = None
    fade_transition = Fade_Transition(screen, true_res)
    game_states = {
        "Main_Menu": Main_Menu(screen, true_res),
        "Canvas": Canvas_Init(screen, None),
        "Gallery": Gallery(screen, true_res, "Paintings/", (120, 210)),
        "Controls": Controls(screen, "Data/Controls/1.png", "Data/Controls/2.png"),
        "Credits": Credits(screen, "Data/Credits/*.png")
    }
    game_states["Gallery"].Set_Canvas(game_states["Canvas"])

    back_button = Back_Button(screen, (50, 10, 100, 50), "Data/Icons/backbutton.png")

    clock = pygame.time.Clock()

    running = True
    while running:
        pygame_events = pygame.event.get()
        screen.blit(background_image, (0, 0))

        mouse_x, mouse_y = pygame.mouse.get_pos()
        if changed_state == state:
            changed_state = game_states[state].Update(mouse_x, mouse_y, pygame_events)
            if changed_state != state:
                transitioning = False
                if changed_state == "Gallery":
                    game_states[changed_state].Load()

        game_states[state].Draw()
        if state != "Main_Menu":
            back_button.Update(mouse_x, mouse_y)
            for event in pygame_events:
                if event.type == pygame.MOUSEBUTTONDOWN and back_button.is_mouse_collided:
                    changed_state = "Main_Menu"
                    transitioning = False
            back_button.Render_Button()

        if transitioning is not None:
            if not transitioning:
                if fade_transition.Update_Inc():
                    state = changed_state
                    if state == "Canvas":
                        pygame.mouse.set_visible(False)
                    else:
                        pygame.mouse.set_visible(True)
                    fade_transition.Update(255)
                    transitioning = True
            else:
                if fade_transition.Update_Inc(-5):
                    fade_transition.Update(0)
                    transitioning = None
            fade_transition.Draw()
        for event in pygame_events:
            if event.type == pygame.QUIT:
                game_states["Canvas"].Save_State()
                running = False
                # connection2.send("none".encode())
        pygame.display.flip()

    pygame.quit()
