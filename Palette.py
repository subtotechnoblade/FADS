import os
import numpy as np
import pygame
import pygame.gfxdraw
from numba import njit
import colorsys


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
        image = pygame.image.load(sprite_path).convert_alpha()
        self.sprite = pygame.transform.scale(image,
                                             (self.pos[-1], self.pos[-1]))
        self.info = info

    def Get_Info(self):
        return self.info

    def Render_Button(self, hover_color=(255, 255, 255), width=3):
        if self.is_mouse_collided:
            pygame.draw.rect(self.screen,
                             hover_color,
                             self.border)
        # pygame.draw.rect(self.screen, (0, 0, 0), self.rect)
        self.screen.blit(self.sprite, self.pos[:2])


class Color_bucket(Button):
    def __init__(self, pos: np.array,
                 screen,
                 color_pin,
                 color: np.array = np.array([255, 255, 255], dtype=np.float32)):
        super().__init__(pos, screen)
        self.RGB = color
        self.color_pin = color_pin

    def Set_RGB(self, rgb: np.array, color_pin: np.array):
        self.RGB = rgb
        self.color_pin = color_pin

    def Get_Color_Pin(self):
        return self.color_pin

    def Get_Color(self):
        return self.RGB

    def Render_Button(self, hover_color=(255, 255, 255)):
        if self.is_mouse_collided:
            pygame.draw.rect(self.screen,
                             hover_color,
                             self.border)
        pygame.draw.rect(self.screen, self.RGB, self.pos)


def HSV_To_RGB(h, s, v):
    # Written by stackoverflow https://stackoverflow.com/questions/24852345/hsv-to-rgb-color-conversion
    # I have no idea what is going on
    shape = h.shape
    i = np.int_(h * 6.0)
    f = h * 6.0 - i

    q = f
    t = 1.0 - f
    i = np.ravel(i)
    f = np.ravel(f)
    i %= 6

    t = np.ravel(t)
    q = np.ravel(q)
    clist = (1 - np.ravel(s) * np.vstack([np.zeros_like(f), np.ones_like(f), q, t])) * np.ravel(v)

    # 0:v 1:p 2:q 3:t
    order = np.array([[0, 3, 1], [2, 0, 1], [1, 0, 3], [1, 2, 0], [3, 1, 0], [0, 1, 2]])
    rgb = clist[order[i], np.arange(np.prod(shape))[:, None]]

    return rgb.reshape(shape + (3,))


def Generate_Colour_Wheel(samples=50, value=1.0):
    # Not written by me, taken from stack overflow
    xx, yy = np.meshgrid(
        np.linspace(-1, 1, samples, dtype=np.float32), np.linspace(-1, 1, samples, dtype=np.float32))

    S = np.sqrt(xx ** 2 + yy ** 2)
    H = (np.arctan2(xx, yy) + np.pi) / (np.pi * 2)

    HSV = np.array([H, S, np.ones(H.shape) * value], dtype=np.float32)
    RGB = HSV_To_RGB(*HSV)

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
                                        axes=(1, 0, 2)) * self.Create_Circular_Mask(True) * 255.0
        self.is_mouse_over_color_wheel = False
        self.is_mouse_over_brightness_slider = False
        self.is_pressed = False
        self.color_pin = np.array([self.radius, self.radius, 1.0], dtype=np.float32)

        self.slider_array = np.tile(np.arange(1, 0, -1 / 150), reps=(10, 1)) * 255.0
        self.slider_array = np.ones(3) * self.slider_array[:, :, np.newaxis]

        self.Numba_Warmup()

    def Create_Circular_Mask(self, full=False):
        coords = np.arange(self.size)
        mask = (coords - self.radius) ** 2 + (coords.reshape((-1, 1)) - self.radius) ** 2 <= self.radius ** 2
        if not full:
            return mask
        return np.ones(3) * mask[:, :, np.newaxis]

    def Update_Color_Wheel(self):
        self.color_wheel = np.transpose(Generate_Colour_Wheel(self.size, value=self.color_pin[2]),
                                        axes=(1, 0, 2)) * self.Create_Circular_Mask(True) * 255.0

    def Update_Pos(self, dx, dy):
        self.pos[0] += dx
        self.pos[1] += dy

        self.center[0] += dx
        self.center[1] += dy

    def Pressed_Color_Wheel(self, mouse_x, mouse_y, pygame_events):
        keys = pygame.key.get_pressed()
        mouse_pressed = pygame.mouse.get_pressed()[0]
        self.is_mouse_over_color_wheel = False
        if (mouse_x - self.center[0]) ** 2 + (mouse_y - self.center[1]) ** 2 <= self.radius ** 2:
            self.is_mouse_over_color_wheel = True
            if mouse_pressed:
                self.is_pressed = True
                if keys[pygame.K_LSHIFT]:
                    self.color_pin[:2] = np.array([self.radius, self.radius], dtype=np.float32)
                    return

                self.color_pin[:2] = np.array([mouse_x - self.pos[0], mouse_y - self.pos[1]], dtype=np.float32)
            return
        if not mouse_pressed:
            self.is_pressed = False

    def Pressed_Slider(self, mouse_x, mouse_y, pygame_events):
        keys = pygame.key.get_pressed()
        mouse_pressed = pygame.mouse.get_pressed()[0]

        self.is_mouse_over_brightness_slider = False
        x, y = self.pos[0] + 160, self.pos[1]
        if x <= mouse_x <= x + 10 and y <= mouse_y <= y + 150:
            self.is_mouse_over_brightness_slider = True
            if mouse_pressed:

                if keys[pygame.K_LSHIFT]:
                    self.color_pin[2] = 1.0
                else:
                    self.color_pin[2] = 1.0 - (mouse_y - self.pos[1]) / 150

                self.Update_Color_Wheel()
                self.is_pressed = True
                return
        if not mouse_pressed:
            self.is_pressed = False

    def Get_Color(self):
        return self.color_wheel[int(self.color_pin[0])][int(self.color_pin[1])]

    def Numba_Warmup(self):
        self.Find_Closest_Color(self.color_wheel, np.array([255, 255, 0]))

    @staticmethod
    @njit(cache=True, nogil=True)
    def Find_Closest_Color(color_wheel, target_color):
        best_loss = np.inf
        best_position = (-1, -1)
        if np.sum(target_color) == 0:
            return 75, 75

        # we do x first because when we render pygame.blit_array flips it
        for x, row in enumerate(color_wheel):
            for y, value in enumerate(row):
                if True:
                    loss = np.sum(np.abs(target_color - value))
                    if loss <= 1e-2:
                        return x, y
                    if loss < best_loss:
                        best_loss = loss
                        best_position = (x, y)
        return best_position

    def Set_Color(self, target_color: np.array) -> np.array:
        brightness = colorsys.rgb_to_hsv(*target_color)[-1] / 255

        self.color_pin[2] = brightness
        self.Update_Color_Wheel()  # we update the color after we have updated the brightness
        self.color_pin[0], self.color_pin[1] = self.Find_Closest_Color(self.color_wheel, target_color=target_color)
        self.Update_Color_Wheel()

    def Get_Color_Pin(self):
        return self.color_pin

    def Set_Color_Pin(self, new_pin):
        self.color_pin = new_pin
        self.Update_Color_Wheel()

    def Update(self, mouse_x, mouse_y, pygame_events):
        self.Pressed_Color_Wheel(mouse_x, mouse_y, pygame_events)
        self.Pressed_Slider(mouse_x, mouse_y, pygame_events)

    def Draw(self):
        # draw the hovering border if the mouse hovers over the color wheel
        if self.is_mouse_over_color_wheel:
            pygame.gfxdraw.filled_circle(self.screen,
                                         *self.center,
                                         self.radius + 4,
                                         (0, 255, 127))
            pygame.gfxdraw.aacircle(self.screen,
                                    *self.center,
                                    self.radius + 4,
                                    (0, 255, 127))
        if self.is_mouse_over_brightness_slider:
            # 150, 10
            pygame.draw.rect(self.screen, (0, 255, 127), (self.pos[0] + 160 - 4, self.pos[1] - 4, 18, 158))

        # Draw the color wheel
        # not sure if this code is efficient
        color_wheel_surf = pygame.Surface((self.size, self.size), flags=pygame.SRCALPHA, depth=32)
        pygame.surfarray.blit_array(color_wheel_surf, self.color_wheel)
        alpha_mask = pygame.surfarray.pixels_alpha(color_wheel_surf)

        alpha_mask[:] = self.Create_Circular_Mask() * 255

        del alpha_mask
        self.screen.blit(color_wheel_surf, self.pos[:2])

        # draw the color wheel border
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
    def __init__(self, screen, starting_coord: np.array, saved_folder_path, button_size=75):
        self.screen = screen
        self.border_width = 15

        self.background = pygame.image.load("Backgrounds/Palette_background.png").convert_alpha()

        self.pos = np.array([starting_coord[0],
                             starting_coord[1],
                             7.5 * (button_size + self.border_width + 3) + 150 + 75 + 2 * self.border_width,
                             2 * (button_size + self.border_width)], dtype=np.int16)

        self.color_wheel = Color_Wheel(radius=75,
                                       pos=np.array([self.pos[0] + 7 * (button_size + self.border_width + 3),
                                                     self.pos[1] + self.border_width, 150, 150]),
                                       screen=self.screen)
        self.color_buckets = []
        for i in range(8):
            color_bucket = Color_bucket(pos=np.array(
                [i * (button_size + 3) + starting_coord[0] + self.border_width, starting_coord[1] + self.border_width,
                 button_size,
                 button_size],
                dtype=np.int16),
                screen=self.screen,
                color_pin=np.array([75, 75, 1.0], dtype=np.float32),
                color=np.array(self.color_wheel.color_wheel[75][75], dtype=np.float32))

            self.color_buckets.append(color_bucket)

        for i in range(8):
            color_bucket = Color_bucket(pos=np.array([i * (button_size + 3) + starting_coord[0] + self.border_width,
                                                      starting_coord[1] + button_size + 3 + self.border_width,
                                                      button_size,
                                                      button_size],
                                                     dtype=np.int16),
                                        screen=self.screen,
                                        color_pin=np.array([75, 75, 1.0], dtype=np.float32),
                                        color=np.array(self.color_wheel.color_wheel[75][75], dtype=np.float32))
            self.color_buckets.append(color_bucket)

        # Loading Code
        self.saved_folder_path = saved_folder_path
        if os.path.isfile(f"{self.saved_folder_path}/Palette.npz"):
            color_bucket_colors = np.load(f"{self.saved_folder_path}/Palette.npz", allow_pickle=True)["colors"]
        else:
            color_bucket_colors = np.array(
                [[255, 0, 0],
                 [220, 20, 60],
                 [238, 75, 43],
                 [253, 94, 83],
                 [0, 0, 255],
                 [0, 191, 255],
                 [135, 206, 250],
                 [255, 255, 0],
                 [238, 175, 97],
                 [251, 144, 98],
                 ],
                dtype=np.float32)
        for color_bucket_id, color in enumerate(color_bucket_colors):
            best_x, best_y = self.color_wheel.Find_Closest_Color(self.color_wheel.color_wheel, np.array(color))
            self.Set_Color(self.color_buckets[color_bucket_id], self.color_wheel.color_wheel[best_x][best_y])

        self.color_wheel.Set_Color_Pin(self.color_buckets[10].Get_Color_Pin())
        self.selected_color_bucket = self.color_buckets[10]

        self.smoothing_buttons = []

        for i, (kernel_type, sprite_path) in enumerate(
                [("constant", "icons/Constant_Icon.png"),
                 ("linear", "icons/Linear_Icon.png"),
                 ("quadratic", "icons/Quadratic_Icon.png"),
                 ("cos", "icons/Cos_Icon.png")]):
            self.smoothing_buttons.append(
                Misc_Selector(
                    np.array([starting_coord[0] + 175 + button_size + 6.5 * (button_size + self.border_width + 3),
                              starting_coord[1] + self.border_width + (i * (self.border_width + 25)),
                              30, 30], dtype=np.int16),
                    screen=self.screen,
                    sprite_path=sprite_path,
                    info=kernel_type))

        # Set the default kernel type
        self.kernel_type, self.selected_smoothing_button = "quadratic", self.smoothing_buttons[2]

        # Note that if self.is_selected_color_picker is True it should overwrite the smoothing buttons
        self.is_selected_color_picker = False
        self.cached_color = None  # used when we unconfirmed the color picker color
        self.color_picker = Misc_Selector(
            np.array([starting_coord[0] + 175 + button_size + 7 * (button_size + self.border_width + 3),
                      starting_coord[1] + self.border_width,
                      30, 30], dtype=np.int16),
            screen=self.screen,
            sprite_path="icons/Color_Picker_Icon.png",
            info=pygame.transform.scale(pygame.image.load("icons/Color_Picker_Icon.png").convert_alpha(), (30, 30)))

        self.is_moving = False
        self.prev_mouse_pos = np.array(pygame.mouse.get_pos(), dtype=np.int16)
        self.background_rect = pygame.draw.rect(screen, color=(100, 100, 100),
                                                rect=self.pos.astype(np.int32, copy=False))

    def Update_Pos(self, dx, dy):
        self.pos += np.array([dx, dy, 0, 0], dtype=np.int16)
        self.color_wheel.Update_Pos(dx, dy)
        for color_bucket in self.color_buckets:
            color_bucket.Update_Pos(dx, dy)

        for smoothing_button in self.smoothing_buttons:
            smoothing_button.Update_Pos(dx, dy)

        self.color_picker.Update_Pos(dx, dy)

    def Get_Color(self):
        return self.selected_color_bucket.Get_Color()

    def Cache_Color(self):
        self.cached_color = (self.selected_color_bucket.Get_Color(), self.selected_color_bucket.Get_Color_Pin())

    def Set_Color(self, target_color_bucket, target_color: np.array) -> np.array:
        self.color_wheel.Set_Color(target_color)
        target_color_bucket.Set_RGB(self.color_wheel.Get_Color(), np.copy(self.color_wheel.Get_Color_Pin()))

    def Save_Palette(self):
        bucket_colors = np.zeros((16, 3), dtype=np.float32)
        for i, color_bucket in enumerate(self.color_buckets):
            bucket_colors[i] = color_bucket.Get_Color()
        np.savez_compressed(f"{self.saved_folder_path}/Palette.npz", colors=bucket_colors)
        print("Save Successful")

    def Get_Kernel_Type(self):
        return self.kernel_type

    def Update_Hover_Selector(self, mouse_x, mouse_y):
        x, y, w, h = self.pos
        if not (x <= mouse_x <= x + w and y <= mouse_y <= y + h):
            return

        for color_bucket_id, color_bucket in enumerate(self.color_buckets):
            color_bucket.Collided_With_Mouse(mouse_x, mouse_y)

        for smoothing_button in self.smoothing_buttons:
            smoothing_button.Collided_With_Mouse(mouse_x, mouse_y)

        self.color_picker.Collided_With_Mouse(mouse_x, mouse_y)

    def Update_Current_Selector(self, mouse_x, mouse_y):
        x, y, w, h = self.pos
        if not (x <= mouse_x <= x + w and y <= mouse_y <= y + h):
            return
        self.color_picker.Collided_With_Mouse(mouse_x, mouse_y)
        for color_buckets in self.color_buckets:
            if color_buckets.Collided_With_Mouse(mouse_x, mouse_y):
                self.selected_color_bucket = color_buckets
                self.color_wheel.Set_Color_Pin(self.selected_color_bucket.color_pin)
                return

        for smoothing_button in self.smoothing_buttons:
            if smoothing_button.Collided_With_Mouse(mouse_x, mouse_y):
                self.kernel_type = smoothing_button.Get_Info()
                self.selected_smoothing_button = smoothing_button
                return

    def Update(self, mouse_x, mouse_y, mouse_pressed, pygame_events: pygame.event, clickable=True):
        updated_pos = np.array([mouse_x, mouse_y])
        if (mouse_pressed[0]
                # if we are colliding with the rectangle given my self.pos
                and (self.pos[0] <= mouse_x <= self.pos[0] + self.pos[2] and self.pos[1] <= mouse_y <= self.pos[1] +
                     self.pos[3])
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

        if not self.is_moving and not self.is_selected_color_picker:
            self.color_wheel.Update(mouse_x, mouse_y, pygame_events)

            self.Update_Hover_Selector(mouse_x, mouse_y)
            for event in pygame_events:
                if event.type == pygame.MOUSEBUTTONDOWN and mouse_pressed[0]:
                    self.Update_Current_Selector(mouse_x, mouse_y)

            # keyboard shortcut for using color picker
            new_color = self.color_wheel.Get_Color()
            if (*new_color,) != (*self.selected_color_bucket.Get_Color(),):
                self.selected_color_bucket.Set_RGB(new_color, self.color_wheel.Get_Color_Pin())

        # Must update after because it isn't possible to update with not self.is_selected_color_picker in the if statement
        self.color_picker.Collided_With_Mouse(mouse_x, mouse_y)

    def Update_After_Render(self, pygame_events: pygame.event):
        mouse_pressed = pygame.mouse.get_pressed()
        mouse_x, mouse_y = pygame.mouse.get_pos()
        keys = pygame.key.get_pressed()

        if self.is_selected_color_picker:
            target_color = np.array(self.screen.get_at((mouse_x, mouse_y))[:3])
            self.selected_color_bucket.Set_RGB(target_color, None)
        if not self.is_moving:
            for event in pygame_events:
                if event.type == pygame.KEYDOWN and keys[pygame.K_c]:
                    self.is_selected_color_picker = not self.is_selected_color_picker
                    self.Cache_Color()
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN and self.color_picker.Collided_With_Mouse(mouse_x, mouse_y):

                    if mouse_pressed[0] and not self.is_selected_color_picker:
                        self.is_selected_color_picker = not self.is_selected_color_picker
                        self.Cache_Color()
                        return

                # checks for color confirmation or confirmation
                if event.type == pygame.MOUSEBUTTONDOWN and self.is_selected_color_picker:
                    if mouse_pressed[0]:
                        self.Set_Color(self.selected_color_bucket, target_color)
                        self.is_selected_color_picker = not self.is_selected_color_picker
                    elif mouse_pressed[2]:
                        old_color, old_pin = self.cached_color
                        self.selected_color_bucket.Set_RGB(old_color, old_pin)
                        self.color_wheel.Set_Color_Pin(old_pin)
                        self.color_wheel.Update()
                        self.is_selected_color_picker = not self.is_selected_color_picker

    def Draw(self):
        self.screen.blit(self.background, self.pos[:2])
        # self.background_rect = pygame.draw.rect(self.screen, color=(100, 100, 100), rect=self.pos.astype(np.int32))

        # draw cursor for selecting the color
        pygame.draw.rect(self.screen, color=(0, 134, 223), rect=self.selected_color_bucket.border)

        self.color_wheel.Draw()

        for color_bucket in self.color_buckets:
            color_bucket.Render_Button(hover_color=(0, 255, 127))

        pygame.draw.rect(self.screen, color=(0, 134, 223), rect=self.selected_smoothing_button.border)
        for smoothing_button in self.smoothing_buttons:
            smoothing_button.Render_Button(hover_color=(0, 255, 127))

        if self.is_selected_color_picker:
            pygame.draw.rect(self.screen, color=(0, 134, 223), rect=self.color_picker.border)
        self.color_picker.Render_Button(hover_color=(0, 255, 127))


if __name__ == "__main__":
    import time
    import os
    import ctypes

    os.environ["SDL_VIDEO_CENTERED"] = "1"
    ctypes.windll.user32.SetProcessDPIAware()
    true_res = (ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1) - 50)
    screen = pygame.display.set_mode(true_res,
                                     pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.HWACCEL | pygame.RESIZABLE,
                                     vsync=1)

    palette = Palette(screen, np.array([500, 400]), None)
    mouse_coord = np.array(pygame.mouse.get_pos(), dtype=np.float32)

    clock = pygame.time.Clock()
    running = True
    while running:
        # clock.tick(60)
        # keys = pygame.key.get_pressed()
        pygame_events = pygame.event.get()
        screen.fill((255, 255, 255))
        palette.Update(*pygame.mouse.get_pos(), pygame.mouse.get_pressed(), pygame_events)
        # delta_mouse = np.array(pygame.mouse.get_pos(), dtype=np.float32) - mouse_coord
        # mouse_coord = np.array(pygame.mouse.get_pos(), dtype=np.float32)
        palette.Draw()

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                break
    pygame.quit()
