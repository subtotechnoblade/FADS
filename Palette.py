import numpy as np
import pygame
import pygame.gfxdraw
import colour


class Button:
    def __init__(self, pos, screen, border_width=3):
        self.pos = pos
        self.screen = screen
        self.border_width = border_width
        self.border = pygame.Rect(self.pos[0] - self.border_width,
                                  self.pos[1] - self.border_width,
                                  self.pos[2] + 2 * self.border_width,
                                  self.pos[3] + 2 * self.border_width)

        self.rect = pygame.Rect(self.pos)
        self.is_mouse_collided = False

    def Set_Pos(self, new_pos):
        assert new_pos.shape[0] == 4
        self.pos = new_pos
        self.rect.update(self.pos)

    def Update_Pos(self, dx, dy):
        self.pos[0] += dx
        self.pos[1] += dy
        self.rect.update(self.pos)
        self.border.update(self.pos[0] - self.border_width,
                           self.pos[1] - self.border_width,
                           self.pos[2] + 2 * self.border_width,
                           self.pos[3] + 2 * self.border_width)

    def Collided_With_Mouse(self, mouse_x, mouse_y):
        self.is_mouse_collided = False
        if is_collided := self.rect.collidepoint(mouse_x, mouse_y):
            self.is_mouse_collided = True
        return is_collided

    def Render_Button(self, hover_color=(255, 255, 255)):
        raise RuntimeError(f"Render Button method not implemented")


class Misc_Selector(Button):
    def __init__(self, pos, screen, sprite_path, info):
        super().__init__(pos, screen)
        self.info = info

    def Get_Info(self):
        return self.info

    def Render_Button(self, hover_color=(255, 255, 255), width=3):
        if self.is_mouse_collided:
            pygame.draw.rect(self.screen,
                             hover_color,
                             self.border)
        pygame.draw.rect(self.screen, (0, 0, 0), self.rect)


class Color_bucket(Button):
    def __init__(self, pos: np.array,
                 screen,
                 color_pin,
                 color: np.array = np.array([255, 255, 255], dtype=np.uint8)):
        super().__init__(pos, screen)
        self.RGB = color
        self.color_pin = color_pin

    def Set_RGB(self, rgb: np.array, color_pin):
        self.RGB = rgb
        self.color_pin = color_pin

    def Get_Color_Pin(self):
        return self.color_pin

    def Render_Button(self, hover_color=(255, 255, 255)):
        if self.is_mouse_collided:
            pygame.draw.rect(self.screen,
                             hover_color,
                             self.border)
        pygame.draw.rect(self.screen, self.RGB, self.pos)


def Generate_Colour_Wheel(samples=50, value=1.0, clip_circle=True, method='Color'):
    # Not written by me, taken from stack overflow
    xx, yy = np.meshgrid(
        np.linspace(-1, 1, samples), np.linspace(-1, 1, samples))

    S = np.sqrt(xx ** 2 + yy ** 2)
    H = (np.arctan2(xx, yy) + np.pi) / (np.pi * 2)

    HSV = colour.utilities.tstack([H, S, np.ones(H.shape) * value])
    RGB = colour.HSV_to_RGB(HSV)

    if method.lower() == 'matplotlib':
        RGB = colour.utilities.orient(RGB, '90 CW')
    elif method.lower() == 'nuke':
        RGB = colour.utilities.orient(RGB, 'Flip')
        RGB = colour.utilities.orient(RGB, '90 CW')
    RGB = np.flipud(RGB)
    return RGB


class Color_Wheel:
    def __init__(self, radius, pos, screen):
        self.radius = radius
        self.size = radius * 2 + 1
        self.pos = pos  # x, y, where the top left corner is
        self.center = np.array([self.pos[0] + self.radius, self.pos[1] + self.radius])

        self.screen = screen

        self.color_wheel = np.transpose(Generate_Colour_Wheel(self.size, value=1),
                                        axes=(1, 0, 2)) * self.Create_Circular_Mask(True) * 255

        self.is_pressed = False
        self.color_pin = np.array([self.radius + 1, self.radius + 1, 1.0], dtype=np.float32)

        self.slider_array = np.tile(np.arange(1, 0, -1 / 150), reps=(10, 1)) * 255
        self.slider_array = np.ones(3) * self.slider_array[:, :, np.newaxis]

    def Create_Circular_Mask(self, full=False):
        coords = np.arange(self.size)
        mask = (coords - self.radius) ** 2 + (coords.reshape((-1, 1)) - self.radius) ** 2 <= self.radius ** 2
        if not full:
            return mask
        return np.ones(3) * mask[:, :, np.newaxis]

    def Update_Color_Wheel(self):
        self.color_wheel = np.transpose(Generate_Colour_Wheel(self.size, value=self.color_pin[2]),
                                        axes=(1, 0, 2)) * self.Create_Circular_Mask(True) * 255

    def Update_Pos(self, dx, dy):
        self.pos[0] += dx
        self.pos[1] += dy

        self.center[0] += dx
        self.center[1] += dy

    def Pressed_Color_Wheel(self, mouse_x, mouse_y):
        keys = pygame.key.get_pressed()
        mouse_pressed = pygame.mouse.get_pressed()[0]
        if mouse_pressed:
            if (mouse_x - self.center[0]) ** 2 + (mouse_y - self.center[1]) ** 2 <= self.radius ** 2:
                self.is_pressed = True
                if keys[pygame.K_LSHIFT]:
                    self.color_pin[:2] = np.array([self.radius + 1, self.radius + 1], dtype=np.float32)
                    return

                self.color_pin[:2] = np.array([mouse_x - self.pos[0], mouse_y - self.pos[1]])
                return
        if not mouse_pressed:
            self.is_pressed = False

    def Pressed_Slider(self, mouse_x, mouse_y):
        keys = pygame.key.get_pressed()
        mouse_pressed = pygame.mouse.get_pressed()[0]
        if mouse_pressed:
            x, y = self.pos[0] + 160, self.pos[1]
            if x <= mouse_x <= x + 10 and y <= mouse_y <= y + 150:
                if keys[pygame.K_LSHIFT]:
                    self.color_pin[2] = 1
                else:
                    self.color_pin[2] = 1.0 - (mouse_y - self.pos[1]) / 150

                self.Update_Color_Wheel()
                self.is_pressed = True
                return
        if not mouse_pressed:
            self.is_pressed = False

    def Get_Color(self):
        # since the color wheel is inverted because of array blit
        return self.color_wheel[int(self.color_pin[0])][int(self.color_pin[1])]

    def Get_Color_Pin(self):
        return self.color_pin

    def Set_Color_Pin(self, new_pin):
        self.color_pin = new_pin
        self.Update_Color_Wheel()

    def Update(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        self.Pressed_Color_Wheel(mouse_x, mouse_y)
        self.Pressed_Slider(mouse_x, mouse_y)

    def Draw(self):
        # Draw the color wheel
        # not sure if this code is efficient
        color_wheel_surf = pygame.Surface((self.size, self.size), flags=pygame.SRCALPHA, depth=32)
        pygame.surfarray.blit_array(color_wheel_surf, self.color_wheel)
        alpha = pygame.surfarray.pixels_alpha(color_wheel_surf)

        alpha[:] = self.Create_Circular_Mask() * 255

        del alpha
        self.screen.blit(color_wheel_surf, self.pos[:2])

        # draw the border
        pygame.gfxdraw.aacircle(self.screen,
                                *self.center,
                                self.radius,
                                (0, 0, 0))

        # Render the slider
        slider_surf = pygame.Surface((self.slider_array.shape[0], self.slider_array.shape[1]), depth=32)
        pygame.surfarray.blit_array(slider_surf, self.slider_array)
        self.screen.blit(slider_surf, (self.pos[0] + 160, self.pos[1]))

        # draw the color pin and slider pin
        pygame.gfxdraw.aacircle(self.screen,
                                self.pos[0] + int(self.color_pin[0]),
                                self.pos[1] + int(self.color_pin[1]),
                                3,
                                (0, 0, 0))

        inverse_brightness = np.array([255, 255, 255], dtype=np.uint8) - 255 * self.color_pin[2]
        pygame.gfxdraw.aacircle(self.screen,
                                self.pos[0] + 165,
                                self.pos[1] + int((1 - self.color_pin[2]) * 150),
                                3,
                                inverse_brightness)


class Palette:
    def __init__(self, screen, starting_coord: np.array, button_size=75):
        self.screen = screen
        self.border_width = 15

        self.pos = np.array([starting_coord[0],
                             starting_coord[1],
                             8 * (button_size + self.border_width + 3) + 150 + 75 + self.border_width,
                             2 * (button_size + self.border_width)], dtype=np.int16)

        self.color_wheel = Color_Wheel(radius=75,
                                       pos=np.array([self.pos[0] + 7 * (button_size + self.border_width + 3),
                                                     self.pos[1] + self.border_width, 150, 150]),
                                       screen=self.screen)
        self.buttons = []
        for i in range(8):
            button = Color_bucket(pos=np.array(
                [i * (button_size + 3) + starting_coord[0] + self.border_width, starting_coord[1] + self.border_width,
                 button_size,
                 button_size],
                dtype=np.int16),
                screen=self.screen,
                color_pin=np.array([76, 76, 1.0]))

            self.buttons.append(button)

        for i in range(8):
            button = Color_bucket(pos=np.array([i * (button_size + 3) + starting_coord[0] + self.border_width,
                                                starting_coord[1] + button_size + 3 + self.border_width, button_size,
                                                button_size],
                                               dtype=np.int16),
                                  screen=self.screen,
                                  color_pin=np.array([76, 76, 1.0]))
            self.buttons.append(button)

        self.selected_button = self.buttons[0]

        self.smoothing_buttons = []
        # todo "Get all the sprite image paths for the smoothing curve icons
        # and update the Render_button code to use the sprite if Ming ever finishes that"
        # fuck
        for i, (kernel_type, sprite_path) in enumerate(
                [("constant", "FUCK"), ("quadratic", "SHIT"), ("cos", "THIS I GAI")]):
            self.smoothing_buttons.append(
                Misc_Selector(
                    np.array([starting_coord[0] + 175 + button_size + 7 * (button_size + self.border_width + 3),
                              starting_coord[1] + self.border_width + (i * (self.border_width + 25)),
                              25, 25], dtype=np.int16),
                    screen=self.screen,
                    sprite_path=None,
                    info=kernel_type))
        self.kernel_type, self.selected_smoothing_button = "cos", self.smoothing_buttons[2]

        # Todo create a file where we store all the preloaded colors like (red, blue, green, light x)
        # If the file is not there we create one
        # When saving we overwrite the previous file if there is any
        # implement the save when the user quits the game
        # self.color_picker.Set_Coord(self.buttons[0].HSLA)


        self.is_moving = False
        self.prev_mouse_pos = np.array(pygame.mouse.get_pos(), dtype=np.int16)
        self.background_rect = pygame.draw.rect(screen, color=(100, 100, 100),
                                                rect=self.pos.astype(np.int32, copy=False))

    def Update_Pos(self, dx, dy):
        self.pos += np.array([dx, dy, 0, 0], dtype=np.int16)
        self.color_wheel.Update_Pos(dx, dy)
        for button in self.buttons:
            button.Update_Pos(dx, dy)

        for smoothing_button in self.smoothing_buttons:
            smoothing_button.Update_Pos(dx, dy)

    def Get_Color(self):
        return self.selected_button.RGB

    def Get_Kernel_Type(self):
        return self.kernel_type

    def Update_Hover_Selector(self, mouse_x, mouse_y):
        x, y, w, h = self.pos
        if not (x <= mouse_x <= x + w and y <= mouse_y <= y + h):
            return

        for button_id, button in enumerate(self.buttons):
            button.Collided_With_Mouse(mouse_x, mouse_y)

        for smoothing_button in self.smoothing_buttons:
            smoothing_button.Collided_With_Mouse(mouse_x, mouse_y)

    def Update_Current_Selector(self, mouse_x, mouse_y):
        x, y, w, h = self.pos
        if not (x <= mouse_x <= x + w and y <= mouse_y <= y + h):
            return

        for button in self.buttons:
            if button.Collided_With_Mouse(mouse_x, mouse_y):
                self.selected_button = button
                self.color_wheel.Set_Color_Pin(self.selected_button.color_pin)
                return

        for smoothing_button in self.smoothing_buttons:
            if smoothing_button.Collided_With_Mouse(mouse_x, mouse_y):
                self.kernel_type = smoothing_button.Get_Info()
                self.selected_smoothing_button = smoothing_button
                return

    def Update(self, pygame_events: pygame.event):
        self.color_wheel.Update()
        mouse_x, mouse_y = pygame.mouse.get_pos()
        updated_pos = np.array([mouse_x, mouse_y])
        mouse_pressed = pygame.mouse.get_pressed()

        if (mouse_pressed[0] and self.background_rect.collidepoint(mouse_x, mouse_y)
                # if we are not pressing the color wheel
                and not self.color_wheel.is_pressed
                # if the cursor is not in this box, (box difference)
                and not ((self.pos[0] + self.border_width <= mouse_x <= self.pos[0] + self.pos[2] - self.border_width
                          and
                          self.pos[1] + self.border_width <= mouse_y <= self.pos[1] + self.pos[
                              3] - 0.5 * self.border_width))):
            self.is_moving = True
        if not mouse_pressed[0]:
            self.is_moving = False

        if self.is_moving:
            dx, dy = updated_pos - self.prev_mouse_pos
            self.Update_Pos(dx, dy)
        self.prev_mouse_pos = updated_pos

        if not self.is_moving:
            self.Update_Hover_Selector(mouse_x, mouse_y)
            if mouse_pressed[0]:
                self.Update_Current_Selector(mouse_x, mouse_y)

            new_color = self.color_wheel.Get_Color()
            if (*new_color,) != (*self.selected_button.RGB,):
                self.selected_button.Set_RGB(new_color, self.color_wheel.Get_Color_Pin())

    def Draw(self):
        self.background_rect = pygame.draw.rect(self.screen, color=(100, 100, 100), rect=self.pos.astype(np.int32))

        # draw cursor for selecting the color
        pygame.draw.rect(self.screen, color=(0, 134, 223), rect=self.selected_button.border)

        self.color_wheel.Draw()

        for button in self.buttons:
            button.Render_Button(hover_color=(0, 255, 127))

        pygame.draw.rect(self.screen, color=(0, 134, 223), rect=self.selected_smoothing_button.border)
        for smoothing_button in self.smoothing_buttons:
            smoothing_button.Render_Button(hover_color=(0, 255, 127))


# DO NOT EDIT
# DO NOT EDIT
# DO NOT EDIT
# DO NOT EDIT

if __name__ == "__main__":
    import time

    screen_size = (1920, 1080)
    screen = pygame.display.set_mode(screen_size, vsync=True, flags=pygame.RESIZABLE)

    palette = Palette(screen, np.array([500, 400]))
    mouse_coord = np.array(pygame.mouse.get_pos(), dtype=np.float32)
    # pos = np.zeros(2, dtype=np.float32)
    s = time.time()
    count = 0
    running = True
    while running:
        # keys = pygame.key.get_pressed()
        pygame_events = pygame.event.get()
        screen.fill((255, 255, 255))
        palette.Update(pygame_events)
        # delta_mouse = np.array(pygame.mouse.get_pos(), dtype=np.float32) - mouse_coord
        # mouse_coord = np.array(pygame.mouse.get_pos(), dtype=np.float32)
        palette.Draw()

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
