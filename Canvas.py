import os
import sys
import ctypes
from collections import deque

import pygame
import pygame.gfxdraw

import numpy as np
# from numba import njit

from splines import CatmullRom

import pyfftw
import scipy.fftpack
from scipy.fft import set_global_backend
from scipy.signal import fftconvolve

from Palette import Palette
from Linked_List import Linked_List
from Filler import Color_Fill, Line_Fill

# import time

import matplotlib.pyplot as plt

# np.set_printoptions(threshold=np.inf)
scipy.fftpack = pyfftw.interfaces.scipy_fftpack
set_global_backend(pyfftw.interfaces.scipy_fft)
pyfftw.config.NUM_THREADS = os.cpu_count()
pyfftw.interfaces.cache.enable()


# @njit(cache=True, nogil=True, fastmath=True)
def Compute_Circular_Mask(radius):
    size = 2 * radius + 1
    coords = np.arange(size, dtype=np.float32)
    return (coords - radius) ** 2 + (coords.reshape((-1, 1)) - radius) ** 2 <= radius ** 2


# @njit(cache=True, nogil=True, fastmath=True)
def Compute_Distance(radius):
    size = 2 * radius + 1
    coords = np.arange(size, dtype=np.float32)
    return np.sqrt((coords - radius) ** 2 + (coords.reshape((-1, 1)) - radius) ** 2)


class Brush:
    def __init__(self, screen, radius, palette, falloff):
        pygame.font.init()

        self.screen = screen

        self.radius = radius
        self.new_radius = radius
        self.palette = palette

        self.Numba_Warmup()
        self.strength = 1.0
        self.new_strength = self.strength
        self.strength_font = pygame.font.SysFont("ebrima", 25)

        self.setting_brush_size, self.setting_brush_strength = False, False

        self.mouse_pos = np.array(pygame.mouse.get_pos(), dtype=np.int16)
        self.prev_mouse_pos = self.mouse_pos
        self.prev_mouse_press = pygame.mouse.get_pressed()

        self.cursor = pygame.Rect(
            (self.mouse_pos[0] - self.radius, self.mouse_pos[1] - self.radius, 2 * self.radius, 2 * self.radius))

        self.current_kernel_type = self.palette.Get_Kernel_Type()

        quadrant_length = int(self.radius / 4)
        self.kernel = self.Compute_Kernel(quadrant_length) / quadrant_length
        self.kernel *= self.strength
        self.visual_kernel = None

    def Numba_Warmup(self):
        self.Compute_Constant_Kernel(self.radius)
        self.Compute_Linear_Kernel(self.radius)
        self.Compute_Quadratic_Kernel(self.radius)
        self.Compute_Cos_Kernel(self.radius)

    @staticmethod
    def Compute_Constant_Kernel(quadrant_length):
        if quadrant_length == 0:
            return np.ones((1, 1), dtype=np.float32)
        return Compute_Circular_Mask(quadrant_length).astype(np.float32, copy=False) * 1

    @staticmethod
    def Compute_Linear_Kernel(quadrant_length):
        distance = 1 - (Compute_Distance(quadrant_length) / quadrant_length)
        distance *= Compute_Circular_Mask(quadrant_length)
        return distance

    @staticmethod
    # @njit(cache=True, nogil=True, fastmath=True)
    def Compute_Quadratic_Kernel(quadrant_length):
        if quadrant_length == 0:
            return np.ones((1, 1), dtype=np.float32)

        distance_mask = Compute_Distance(quadrant_length) / quadrant_length

        kernel = ((-distance_mask ** 2) + 1)
        kernel *= Compute_Circular_Mask(quadrant_length)
        return kernel

    @staticmethod
    # @njit(cache=True, fastmath=True)
    def Compute_Cos_Kernel(quadrant_length):
        if quadrant_length == 0:
            return np.ones((1, 1), dtype=np.float32)

        distance_mask = Compute_Distance(quadrant_length) / quadrant_length
        # print
        kernel = (0.5 * np.cos(np.pi * distance_mask) + 0.5).astype(np.float32)
        kernel /= kernel[quadrant_length][quadrant_length]
        kernel *= Compute_Circular_Mask(quadrant_length)  # Mask out the necessary pixels to 0

        return kernel

    def Compute_Kernel(self, quadrant_length):
        # print(self.palette.Get_Kernel_Type(), quadrant_length)\

        if "cos" == self.palette.Get_Kernel_Type():
            return self.Compute_Cos_Kernel(quadrant_length)

        if "linear" == self.palette.Get_Kernel_Type():
            return self.Compute_Linear_Kernel(quadrant_length)

        if "quadratic" == self.palette.Get_Kernel_Type():
            return self.Compute_Quadratic_Kernel(quadrant_length)

        if "constant" == self.palette.Get_Kernel_Type():
            return self.Compute_Constant_Kernel(quadrant_length * 0.8)

    def Get_Kernel(self):
        return self.kernel

    def Render_Kernel(self, radius, visual_kernel):
        size = radius * 2 + 1
        kernel_surf = pygame.Surface((size, size), pygame.SRCALPHA, 32)
        kernel_surf.fill(self.palette.Get_Color())
        alpha = pygame.surfarray.pixels_alpha(kernel_surf)
        alpha[:] = visual_kernel

        del alpha  # we must use this, or else this won't work
        self.screen.blit(kernel_surf, (self.mouse_pos[0] - radius, self.mouse_pos[1] - radius))
        del kernel_surf

    def Update(self, keys, pygame_events, tile_size):
        if (keys[pygame.K_r] and keys[
            pygame.K_LSHIFT]) and not self.setting_brush_strength and not self.setting_brush_size:
            self.setting_brush_strength = True
        elif keys[pygame.K_r] and not self.setting_brush_strength and not self.setting_brush_size:
            self.setting_brush_size = True

        # If we are not setting size or strength
        current_mouse_pos = np.array(pygame.mouse.get_pos(), dtype=np.int16)
        if not (self.setting_brush_size or self.setting_brush_strength):
            self.mouse_pos = np.array(pygame.mouse.get_pos(), dtype=np.int16)
            self.cursor.x, self.cursor.y = self.mouse_pos[0] - self.radius, self.mouse_pos[1] - self.radius

        # If we want to change the brush size
        else:
            confirm = None
            for event in pygame_events:
                if event.type == pygame.MOUSEBUTTONUP:
                    if self.prev_mouse_press[0]:
                        confirm = True
                    elif self.prev_mouse_press[2]:
                        confirm = False

            dx, dy = current_mouse_pos - self.prev_mouse_pos
            if self.setting_brush_size:
                scaling = 1
                if keys[pygame.K_LSHIFT]:
                    scaling = 0.25
                self.new_radius += int(dx * scaling)
                self.new_radius = self.new_radius if self.new_radius >= 1 else 1

                if confirm:
                    self.radius = self.new_radius

                    self.cursor.update(self.mouse_pos[0] - self.radius, self.mouse_pos[1] - self.radius,
                                       2 * self.radius, 2 * self.radius)

                    # Todo: add new method to get the current kernel
                    # In order to change the current kernel, and re compute it
                    # make a new method that does that based on the current kernel selection
                    # also implement that (current kernel selection)

                    quadrant_length = int(self.radius / 4)
                    self.kernel = self.Compute_Kernel(quadrant_length) / (quadrant_length if quadrant_length > 0 else 1)

                    self.kernel *= self.strength
                if confirm is not None:
                    self.new_radius = self.radius
                    self.setting_brush_size = False


            # If we want to change the brush strength
            elif self.setting_brush_strength:
                scaling = 1
                if keys[pygame.K_LSHIFT]:
                    scaling = 0.25
                self.new_strength += (dx / 200) * scaling
                self.new_strength = np.clip(self.new_strength, 0, 1)
                # implement strength control code here

                if confirm:
                    self.strength = self.new_strength

                    self.cursor.update(self.mouse_pos[0] - self.radius, self.mouse_pos[1] - self.radius,
                                       2 * self.radius, 2 * self.radius)
                    quadrant_length = int(self.radius / 4)
                    self.kernel = self.Compute_Kernel(quadrant_length) / (quadrant_length if quadrant_length > 0 else 1)

                    self.kernel *= self.strength
                if confirm is not None:
                    self.new_strength = self.strength
                    self.setting_brush_strength = False

        # Update the kernel

        if self.current_kernel_type != (palette_kernel_type := self.palette.Get_Kernel_Type()):
            quadrant_length = int(self.radius / 4)
            self.kernel = (self.Compute_Kernel(quadrant_length) / (
                quadrant_length if quadrant_length > 0 else 1)) * self.strength

            self.current_kernel_type = palette_kernel_type

        self.prev_mouse_pos = np.array(pygame.mouse.get_pos(), dtype=np.int16)
        self.prev_mouse_press = pygame.mouse.get_pressed()

    def Draw_brush(self):
        if not (self.setting_brush_strength or self.setting_brush_size):
            pygame.gfxdraw.aacircle(self.screen,
                                    self.mouse_pos[0],
                                    self.mouse_pos[1],
                                    self.radius,
                                    (0, 0, 0))
            pygame.gfxdraw.aacircle(self.screen,
                                    self.mouse_pos[0],
                                    self.mouse_pos[1],
                                    2,
                                    (0, 0, 0))
        elif self.setting_brush_size:

            pygame.gfxdraw.aacircle(self.screen,
                                    self.mouse_pos[0],
                                    self.mouse_pos[1],
                                    self.new_radius,
                                    (0, 0, 0))
            if self.palette.Get_Kernel_Type() != "constant":
                visual_kernel = self.Compute_Kernel(self.new_radius) * 255 * self.strength
            else:
                visual_kernel = self.Compute_Kernel(self.new_radius * 1.25) * 255 * self.strength

            self.Render_Kernel(self.new_radius, visual_kernel)

        elif self.setting_brush_strength:

            # This is so stupid, I don't know why we mul by 255 and the radius
            if self.palette.Get_Kernel_Type() != "constant":
                kernel = self.Compute_Kernel(100) * 255 * self.new_strength
            else:
                kernel = self.Compute_Kernel(100 * 1.25) * 255 * self.new_strength

            self.Render_Kernel(100, kernel)

            pygame.gfxdraw.aacircle(self.screen,
                                    self.mouse_pos[0],
                                    self.mouse_pos[1],
                                    100,
                                    (0, 0, 0))

            # Smoller Circle
            pygame.gfxdraw.filled_circle(self.screen,
                                         self.mouse_pos[0],
                                         self.mouse_pos[1],
                                         25,
                                         (255, 255, 255)
                                         )
            # Perfectionist circle to cover up the non aa circle above ;(
            pygame.gfxdraw.aacircle(self.screen,
                                    self.mouse_pos[0],
                                    self.mouse_pos[1],
                                    25,
                                    (255, 255, 255))
            # Control Circle
            pygame.gfxdraw.aacircle(self.screen,
                                    self.mouse_pos[0],
                                    self.mouse_pos[1],
                                    int(25 + (100 - 25) * self.new_strength),
                                    (0, 0, 0))

            strength_value = self.strength_font.render(f"{round(self.new_strength, 2)}", True, (0, 0, 0))
            self.screen.blit(strength_value, (self.mouse_pos - np.array([25 - 3, 25 - 7])))


class Canvas:
    def __init__(self, screen, start_pos, shape, tile_size=5, brush_radius=10, load_path="", falloff="linear"):
        self.screen = screen
        self.shape = shape
        self.tile_size = tile_size

        self.pos = np.array(
            [start_pos[0], start_pos[1], shape[1] * tile_size, shape[0] * tile_size],
            dtype=np.uint32)

        self.drawn_curve = None
        self.image = pygame.Surface(self.pos[2:])
        self.image.fill((255, 255, 255))

        if load_path == "":
            self.canvas_pixels = np.ones((*shape[::-1], 3), dtype=np.float32) * 255.0
        else:
            data = np.load(load_path, allow_pickle=True)
            self.canvas_pixels = data["inputs"].astype(np.float32)
            pygame.surfarray.blit_array(self.image,
                                        self.Scale_Image(self.canvas_pixels))

        # start of the undo redo function
        self.buffer = Linked_List()
        self.buffer.Add(np.array(self.canvas_pixels))

        self.mouse_curve = deque()

        self.palette = Palette(self.screen, self.pos[:2] - np.array([0, 225]))

        self.brush = Brush(screen=self.screen, radius=brush_radius, palette=self.palette, falloff=falloff)
        self.transform = np.array([self.tile_size, self.tile_size], dtype=np.uint16)
        self.mask = None

        self.is_drawing = False

        self.fill_threshold = 0

    def Save(self, save_path, name, evaluation):
        np.savez_compressed(f"{save_path}/{name}", inputs=self.canvas_pixels,
                            outputs=np.array([evaluation], dtype=np.float32))

    def Load(self, load_path):
        data = np.load(load_path, allow_pickle=True)
        self.canvas_pixels = data["inputs"]
        print(f"Loaded: {load_path}")
        print(f"Evaluation for this piece is {data['outputs']}")

    def Check_Brush_Collision(self):
        return self.brush.cursor.colliderect(self.pos)

    def Scale_Image(self, arr):
        return np.repeat(np.repeat(arr, self.tile_size, axis=0), self.tile_size, axis=1)

    def Linear_Blend(self, color: np.array, mask: np.array):
        mask = np.ones((*mask.shape, 3), dtype=np.float32) * mask[:, :, np.newaxis]
        return np.around(self.canvas_pixels * (1 - mask) + (color * mask), decimals=5)

    def Alpha_blend(self, color: np.array, mask: np.array):
        # self.canvas_pixels = self.canvas_pixels * (1 - mask) + color * mask
        mask = np.ones((*mask.shape, 3)) * mask[:, :, np.newaxis]
        return np.around((self.canvas_pixels + (color * mask)) / (1 + mask), decimals=5)

    def Gamma_corrected_multiply(self, color: np.array, mask: np.array):
        painted_color = mask * color

        linear_canvas = ((self.canvas_pixels / 255) ** 2.2) * 255
        linear_color = ((painted_color / 255) ** 0.5) * 255

        linear_result = (((1 - mask) * self.canvas_pixels) + (mask * linear_color))
        result = linear_result.clip(0, 255)
        self.canvas_pixels = result

    def CatMull_Rom_interpolation(self, mouse_curve, tile_size, fully=False) -> np.array:
        # assert len(mouse_curve) == 4
        if not fully:
            p0, p1 = mouse_curve[-2:]
        else:
            p0, _, _, p1 = mouse_curve[-4:]

        delta_x, delta_y = p1[0] - p0[0], p1[1] - p0[1]
        x_increments = int(((delta_x ** 2 + delta_y ** 2) ** 0.5))
        x_increments = x_increments if x_increments > 1 else 1

        inter = CatmullRom(mouse_curve[-4:], alpha=0.65)
        if not fully:
            x_inter = np.linspace(inter.grid[2], inter.grid[3], x_increments)
        else:
            x_inter = np.linspace(inter.grid[0], inter.grid[3], x_increments)
        interpolated_points = inter.evaluate(x_inter, 0)

        return interpolated_points.astype(np.int16, copy=False)

    def Compute_Constant_Mask(self, kernel, strength):
        mask = fftconvolve(self.drawn_curve, kernel, "same")
        mask[mask <= 1e-4] = 0
        mask[mask > 0] = strength

        return mask

    def Compute_Convolved_Mask(self, kernel):
        mask = fftconvolve(self.drawn_curve, kernel, "same")
        mask[mask <= 1e-4] = 0
        mask[mask > 1] = 1

        return mask

    def Update(self, pygame_events):
        self.pygame_events = pygame_events
        is_brush_collided = self.Check_Brush_Collision()
        keys = pygame.key.get_pressed()
        # mouse_pos
        mouse_down = False
        mouse_pos = pygame.mouse.get_pos()
        mouse_x, mouse_y = mouse_pos
        mouse_pressed = pygame.mouse.get_pressed()[0]

        for event in pygame_events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_down = True
            if ((event.type == pygame.MOUSEBUTTONDOWN and event.button == 1)
                    and not self.is_drawing and not (
                            self.brush.setting_brush_size or self.brush.setting_brush_strength)):
                if is_brush_collided and (
                        self.pos[0] <= mouse_x <= self.pos[0] + self.pos[2] and self.pos[1] <= mouse_y <= self.pos[1] +
                        self.pos[3]):
                    self.is_drawing = True
                    self.drawn_curve = np.zeros((self.shape[1], self.shape[0]))

            if event.type == pygame.KEYDOWN:
                if keys[pygame.K_LCTRL] and keys[pygame.K_LSHIFT] and keys[pygame.K_z]:
                    self.buffer.Move_Pointer(1)
                    self.canvas_pixels = self.buffer.pointer.snapshot
                    pygame.surfarray.blit_array(self.image,
                                                self.Scale_Image(self.canvas_pixels))

                elif keys[pygame.K_LCTRL] and keys[pygame.K_z]:
                    self.buffer.Move_Pointer(-1)
                    self.canvas_pixels = self.buffer.pointer.snapshot
                    pygame.surfarray.blit_array(self.image,
                                                self.Scale_Image(self.canvas_pixels))
            if event.type == pygame.KEYUP:
                pygame.surfarray.blit_array(self.image,
                                            self.Scale_Image(self.canvas_pixels))

        # must be before brush update
        self.palette.Update(mouse_x, mouse_y, mouse_pressed, pygame_events)
        if self.palette.is_selected_color_picker:
            return

        self.brush.Update(keys, pygame_events, self.tile_size)
        # update when we are not drawing nor selecting brush size/strength
        if ((self.pos[0] <= mouse_x <= self.pos[0] + self.pos[2] - self.tile_size and self.pos[1] <= mouse_y <=
             self.pos[1] + self.pos[3] - self.tile_size) and
                not (self.brush.setting_brush_size or self.brush.setting_brush_strength) and
                keys[pygame.K_f]):
            # attempt to update threshold only when attempting to fill
            for event in pygame_events:
                if event.type == pygame.MOUSEWHEEL:
                    if 0 <= self.fill_threshold + 0.05 * event.y <= 1:
                        self.fill_threshold += 0.05 * event.y
                    break

            # check for line fill first
            if keys[pygame.K_f] and keys[pygame.K_LCTRL]:
                mask = Line_Fill(self.canvas_pixels,
                                 (int((mouse_x - self.pos[0]) / self.tile_size),
                                  int((mouse_y - self.pos[1]) / self.tile_size)),
                                 threshold=self.fill_threshold)

                if mouse_pressed and mouse_down:  # if we confirm the fill
                    self.canvas_pixels = self.Linear_Blend(self.palette.Get_Color(),
                                                           mask * self.brush.strength)
                    # add undo step into the buffer
                    if not np.allclose(a=self.canvas_pixels, b=self.buffer.pointer.snapshot, rtol=5e-2):
                        self.buffer.Add(np.array(self.canvas_pixels))

                pygame.surfarray.blit_array(self.image, self.Scale_Image(self.Linear_Blend(self.palette.Get_Color(),
                                                                                           mask * self.brush.strength)))

            # check for color fill
            elif keys[pygame.K_f]:
                mask = Color_Fill(self.canvas_pixels,
                                  (int((mouse_x - self.pos[0]) / self.tile_size),
                                   int((mouse_y - self.pos[1]) / self.tile_size)),
                                  threshold=self.fill_threshold)
                if mouse_pressed and mouse_down:  # if we confirm the fill
                    self.canvas_pixels = self.Linear_Blend(self.palette.Get_Color(),
                                                           mask * self.brush.strength)
                    # add an undo step into the buffer
                    if not np.allclose(a=self.canvas_pixels, b=self.buffer.pointer.snapshot, rtol=5e-2):
                        self.buffer.Add(np.array(self.canvas_pixels))

                pygame.surfarray.blit_array(self.image,
                                            self.Scale_Image(self.Linear_Blend(self.palette.Get_Color(),
                                                                               mask * self.brush.strength)))

        # update if we want to draw thus clicking and dragging
        elif (is_brush_collided and self.is_drawing and
              not (self.brush.setting_brush_size or self.brush.setting_brush_strength)):
            # put all update drawing code here that doesn't interfere with the brush updates 
            unique_points = len(set(self.mouse_curve))

            if unique_points == 4:
                interpolated_mouse_coords = self.CatMull_Rom_interpolation(np.array(self.mouse_curve), self.tile_size,
                                                                           fully=True)
            elif unique_points > 4:
                interpolated_mouse_coords = self.CatMull_Rom_interpolation(np.array(self.mouse_curve), self.tile_size,
                                                                           fully=False)
            else:
                interpolated_mouse_coords = np.array(self.mouse_curve)

            for x, y in interpolated_mouse_coords:
                if self.pos[0] <= x + 1 <= self.pos[0] + self.pos[2] and self.pos[1] <= y + 1 <= self.pos[1] + self.pos[
                    3]:
                    self.drawn_curve[int((x - self.pos[0]) / self.tile_size)][
                        int((y - self.pos[1]) / self.tile_size)] = 1

                # print(True)
            # Todo implement a way to get the current blending type
            # mask = self.Get_Mask(mouse_coords=interpolated_mouse_coords)
            # self.Alpha_blend(self.palette.Get_color(), mask)
            # self.Linear_blend(self.palette.Get_color(), mask)
            if self.palette.Get_Kernel_Type() != "constant":
                self.mask = self.Compute_Convolved_Mask(self.brush.Get_Kernel())
            else:
                self.mask = self.Compute_Constant_Mask(self.brush.Get_Kernel(), self.brush.strength)

            pygame.surfarray.blit_array(self.image,
                                        self.Scale_Image(self.Linear_Blend(self.palette.Get_Color(), self.mask)))
        if not self.mouse_curve or ((mouse_x, mouse_y) not in self.mouse_curve and self.is_drawing):
            self.mouse_curve.append((mouse_x, mouse_y))

        if (not mouse_pressed) and not (
                self.brush.setting_brush_size or self.brush.setting_brush_strength):
            # confirm drawing code
            if self.is_drawing and self.mask is not None:
                # If we left go from drawing
                # We thus appy the colored mask to the self.canvas_pixels
                self.canvas_pixels = self.Linear_Blend(self.palette.Get_Color(), self.mask)

                if not np.allclose(a=self.canvas_pixels, b=self.buffer.pointer.snapshot, rtol=5e-2):
                    self.buffer.Add(np.array(self.canvas_pixels))

                self.mask = None
            self.is_drawing = False
            self.mouse_curve = deque()

    def Draw(self):
        # Canvas drawing code
        self.screen.blit(self.image, self.pos[:2])
        # ui drawing code
        self.palette.Draw()

        self.palette.Update_After_Render(self.pygame_events)  # Unconventional but must be used

        if not self.palette.is_selected_color_picker:
            self.brush.Draw_brush()
        else:
            mouse_x, mouse_y = pygame.mouse.get_pos()

            # Todo change this to the color picker icon
            pygame.draw.rect(self.screen, (0, 0, 0), (mouse_x, mouse_y, 10, 10), width=5)


if __name__ == "__main__":
    os.environ["SDL_VIDEO_CENTERED"] = "1"
    ctypes.windll.user32.SetProcessDPIAware()
    true_res = (ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1) - 50)
    screen = pygame.display.set_mode(true_res,
                                     pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.HWACCEL | pygame.RESIZABLE,
                                     vsync=1)

    # pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEBUTTONDOWN])

    pygame.display.set_caption("Fine Arts Drawing Simulator")
    clock = pygame.time.Clock()
    pygame.mouse.set_visible(False)

    save_path = "Paintings"
    os.makedirs(save_path, exist_ok=True)

    name = ""  # specify file name here

    if name == "":
        load_path = ""
    else:
        load_path = f"Paintings/{name}.npz"
        if not os.path.isfile(load_path):
            raise FileNotFoundError("Could not find the file with the name given")

    canvas = Canvas(screen=screen,
                    start_pos=(100, 400),
                    shape=(120, 210),
                    tile_size=5,
                    brush_radius=20,
                    load_path=load_path,
                    falloff="linear")
    running = True
    while running:
        clock.tick(60)
        pygame_events = pygame.event.get()
        screen.fill((153, 207, 224))

        canvas.Update(pygame_events)

        canvas.Draw()

        keys = pygame.key.get_pressed()
        for event in pygame_events:
            if event.type == pygame.KEYDOWN:
                if keys[pygame.K_LCTRL] and keys[pygame.K_s]:
                    name = input("Name of file:")
                    valid = False
                    while not valid:
                        try:
                            evaluation = float(input("Evaluation of your art:"))
                            if -1 <= evaluation <= 1:
                                valid = True
                            else:
                                print("Evaluation can only be between -1 and 1")
                        except:
                            print("BRu U gave some letters ")
                    canvas.Save(save_path, name, evaluation)
                elif keys[pygame.K_LCTRL] and keys[pygame.K_l]:
                    valid = False
                    while not valid:
                        name = input("Load file name:")
                        if os.path.exists(f"Paintings/{name}.npz"):
                            valid = True
                        else:
                            print("Name not found in paintings")
                    canvas.Load(f"Paintings/{name}.npz")
                    print()

            if event.type == pygame.QUIT:
                running = False
                # break
        pygame.display.flip()
    # sys.exit()
    pygame.quit()
