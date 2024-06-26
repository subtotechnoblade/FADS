import os
# import sys
import time
from collections import deque

import pygame
import pygame.gfxdraw

import numpy as np
from glob import glob
# from numba import njit

from splines import CatmullRom

from PIL import Image
# import pyfftw
# import scipy.fftpack
# from scipy.fft import set_global_backend
from scipy.signal import fftconvolve

from Palette import Palette
from Linked_List import Linked_List
from Filler import Color_Fill, Line_Fill

# import matplotlib.pyplot as plt

# scipy.fftpack = pyfftw.interfaces.scipy_fftpack
# set_global_backend(pyfftw.interfaces.scipy_fft)
# pyfftw.config.NUM_THREADS = os.cpu_count()
# pyfftw.interfaces.cache.enable()


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
    def __init__(self, screen, radius, palette):
        pygame.font.init()

        self.screen = screen

        self.radius = radius
        self.new_radius = radius
        self.palette = palette

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

    @classmethod
    def Numba_Warmup(cls, radius):
        Compute_Distance(radius)
        Compute_Circular_Mask(radius)
        cls.Compute_Constant_Kernel(radius - 1)
        cls.Compute_Linear_Kernel(radius)
        cls.Compute_Quadratic_Kernel(radius - 1)
        cls.Compute_Cos_Kernel(radius) - 1

    @staticmethod
    # @njit(cache=True, nogil=True, fastmath=True)
    def Compute_Linear_Kernel(quadrant_length):
        distance = 1 - (Compute_Distance(quadrant_length) / quadrant_length)
        distance *= Compute_Circular_Mask(quadrant_length)
        return distance

    @staticmethod
    #     @njit(cache=True, nogil=True, fastmath=True)
    def Compute_Constant_Kernel(quadrant_length):
        return Compute_Circular_Mask(quadrant_length)

    @staticmethod
    # @njit(cache=True, nogil=True, fastmath=True)
    def Compute_Quadratic_Kernel(quadrant_length):
        distance_mask = Compute_Distance(quadrant_length) / quadrant_length

        kernel = ((-distance_mask ** 2) + 1)
        kernel *= Compute_Circular_Mask(quadrant_length)
        return kernel

    @staticmethod
    #     @njit(cache=True, fastmath=True)
    def Compute_Cos_Kernel(quadrant_length):
        distance_mask = Compute_Distance(quadrant_length) / quadrant_length
        # print
        kernel = (0.5 * np.cos(np.pi * distance_mask) + 0.5)
        kernel /= kernel[quadrant_length][quadrant_length]
        kernel *= Compute_Circular_Mask(quadrant_length)  # Mask out the necessary pixels to 0

        return kernel

    def Compute_Kernel(self, quadrant_length):
        # print(self.palette.Get_Kernel_Type(), quadrant_length)

        if "cos" == self.palette.Get_Kernel_Type():
            return self.Compute_Cos_Kernel(quadrant_length)

        if "linear" == self.palette.Get_Kernel_Type():
            return self.Compute_Linear_Kernel(quadrant_length)

        if "quadratic" == self.palette.Get_Kernel_Type():
            return self.Compute_Quadratic_Kernel(quadrant_length)

        if "constant" == self.palette.Get_Kernel_Type():
            return self.Compute_Constant_Kernel(int(quadrant_length * 0.8))

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
                self.new_radius = max(1, min(self.new_radius, 350))

                if confirm:
                    self.radius = self.new_radius

                    self.cursor.update(self.mouse_pos[0] - self.radius, self.mouse_pos[1] - self.radius,
                                       2 * self.radius, 2 * self.radius)

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
                self.new_strength = max(0.0, min(self.new_strength, 1.0))
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
    def __init__(self, screen,
                 start_pos, shape,
                 tile_size=5,
                 brush_radius=10,
                 saved_folder_path="",
                 pipe_connection=None):
        self.screen = screen
        self.shape = shape
        self.tile_size = tile_size
        self.pos = np.array(
            [start_pos[0], start_pos[1], shape[1] * self.tile_size, shape[0] * self.tile_size],
            dtype=np.uint32)
        self.pos[:2] += np.array([0, 180], dtype=np.uint32)

        self.drawn_curve = None
        self.image = pygame.Surface(self.pos[2:])
        self.image.fill((255, 255, 255))

        # start of the undo redo function
        self.saved_folder_path = saved_folder_path
        if os.path.isfile(f"{self.saved_folder_path}/Buffer.npz"):
            snapshots = np.load(f"{self.saved_folder_path}/Buffer.npz", allow_pickle=True)["buffer"]
            if snapshots[0].shape[:2][::-1] == self.shape:
                self.buffer = Linked_List(snapshots)
            else:
                # If the shapes are not the same, then we just reload the canvas as blank
                self.buffer = Linked_List()
                self.canvas_pixels = np.ones((*self.shape[::-1], 3), dtype=np.float32) * 255
                self.buffer.Add(np.array(self.canvas_pixels))
        else:
            self.buffer = Linked_List()
            self.canvas_pixels = np.ones((*self.shape[::-1], 3), dtype=np.float32) * 255
            self.buffer.Add(np.array(self.canvas_pixels))
        self.canvas_pixels = np.array(self.buffer.pointer.snapshot, dtype=np.float32)
        # print(self.canvas_pixels.shape)
        # print(self.Scale_Image(self.canvas_pixels).shape)
        pygame.surfarray.blit_array(self.image,
                                    self.Scale_Image(self.canvas_pixels))

        self.mouse_curve = deque()

        self.palette = Palette(self.screen, self.pos[:2] + np.array([0, -180], dtype=np.int32), self.saved_folder_path)

        self.brush = Brush(screen=self.screen, radius=brush_radius, palette=self.palette)
        self.transform = np.array([self.tile_size, self.tile_size], dtype=np.uint16)
        self.mask = None

        self.is_drawing = False
        self.clamp = 0

        self.fill_threshold = 0

        self.comment_font = pygame.font.Font(None, 40)
        self.comment_start_time = 0
        self.show_comment = False
        self.comment_surfs = []

        self.pipe_connection = pipe_connection
        self.generating_text = False

    def Set_Screen(self, screen):
        self.screen = screen
        self.brush.screen = screen

    def Save_Train(self, save_path, evaluation, comment):
        np.savez(f"{save_path}",
                 inputs=self.canvas_pixels,
                 outputs=evaluation,
                 comment=comment)

    def Save(self, save_path):
        np.savez(f"{save_path}", inputs=self.canvas_pixels)

    def Save_State(self):
        arr_buffer = self.buffer.To_Array()
        print(f"{self.saved_folder_path}/Buffer.npz")
        np.savez(f"{self.saved_folder_path}/Buffer.npz", buffer=arr_buffer)
        self.palette.Save_Palette()

    def Load(self, load_path):
        data = np.load(load_path, allow_pickle=True)
        self.canvas_pixels = data["inputs"]
        print(f"Loaded: {load_path}")
        if len(data['outputs']) != 3:

            print(f"Evaluation for this piece is {data['outputs']}")
            print(f"Since this is not using the new evaluation system please re-save this")
            print("If you want to over write please jud save it as the same name")
        else:
            print(f"Color, Form, Acceptability: {data['outputs']}")
        try:
            print(f"Comment is: {data['comment']}")
        except:
            print("There is no comment on this piece, consider re-saving this and assigning the comment")
            print("If you want to overwrite this, make sure to specify the same name as the loaded one")
        self.buffer.Add(self.canvas_pixels)
        pygame.surfarray.blit_array(self.image,
                                    self.Scale_Image(self.canvas_pixels))

    def Load_Internal(self, image_path):
        data = np.load(image_path, allow_pickle=True)
        self.canvas_pixels = data["inputs"]
        self.buffer.Add(self.canvas_pixels)
        pygame.surfarray.blit_array(self.image,
                                    self.Scale_Image(self.canvas_pixels))

    def Save_Image(self, file_path):
        data = self.Scale_Image(np.transpose(self.canvas_pixels, (1, 0, 2)))
        img = Image.fromarray(data.astype(np.uint8))
        img.save(file_path)

    def Check_Brush_Collision(self):
        return self.brush.cursor.colliderect(self.pos)

    def Scale_Image(self, arr):
        return np.repeat(np.repeat(arr, self.tile_size, axis=0), self.tile_size, axis=1)

    def Linear_Blend(self, color: np.array, mask: np.array):
        mask = np.ones((*mask.shape, 3), dtype=np.float32) * mask[:, :, np.newaxis]
        return np.around(self.canvas_pixels * (1 - mask) + (color * mask), decimals=5)

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

    def Set_Comment(self, input_text, max_words_in_line=5):
        if not self.show_comment:
            split_text = input_text.split(" ")

            combined_text = ""
            for i, text in enumerate(split_text):
                combined_text += text + " "
                if (i + 1) % max_words_in_line == 0 and i != 0:
                    self.comment_surfs.append(self.comment_font.render(combined_text, 1, (255, 255, 255)))
                    combined_text = ""
            else:  # first time in this program using for and else
                self.comment_surfs.append(self.comment_font.render(combined_text, 1, (255, 255, 255)))
            self.show_comment = True
            self.comment_start_time = time.time()

    def Render_Comment(self, pos):
        if self.show_comment:
            for i, comment_surf in enumerate(self.comment_surfs):
                self.screen.blit(comment_surf, (pos[0], pos[1] + i * 20))

    def Submit(self) -> (bool, str):
        return
        file_name = len(glob("Images/*.png"))
        self.Save_Image(f"Images/{file_name}.png")
        self.Save(f"Paintings/{file_name}.npz")
        time.sleep(0.1)
        self.pipe_connection.send(f"Paintings/{file_name}.npz".encode())
        self.generating_text = True

    def Get_Comment(self):
        if os.path.exists("Data/tmp/signal.txt"):
            os.remove("Data/tmp/signal.txt")
            print("Recieved comment")
            return self.pipe_connection.recv().decode()

    def Update(self, mouse_x, mouse_y, pygame_events):
        keys = pygame.key.get_pressed()
        self.pygame_events = pygame_events
        # if self.generating_text or self.show_comment:
        #     self.brush.mouse_pos = np.array(pygame.mouse.get_pos())
        #     model_output = self.Get_Comment()
        #     if model_output is not None:
        #         model_output = model_output.decode()
        #         print(model_output)
        #         comment, evaluation = model_output.split("|")
        #         color, form, _ = evaluation.split(",")
        #         color, form = float(color), float(form)
        #         comment += f"I give you a {color * 0.75 + form * 0.25}!"
        #         self.Set_Comment(comment)
        #         self.generating_text = False
        #
        # for event in pygame_events:
        #     if event == pygame.KEYDOWN and keys[pygame.K_RETURN]:
        #         self.show_comment = False
        #         self.comment_surfs = []
        #         self.comment_start_time = None
        #
        #         self.buffer = Linked_List()
        #         self.canvas_pixels = np.ones((*self.shape[::-1], 3), dtype=np.float32) * 255
        #         self.buffer.Add(np.array(self.canvas_pixels))
        #

            # return "Canvas"  # prevent any bug from occuring if you contiue to draw
        is_brush_collided = self.Check_Brush_Collision()

        mouse_down = False
        mouse_pressed = pygame.mouse.get_pressed()

        for event in self.pygame_events:
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
                # Redo
                if keys[pygame.K_LCTRL] and keys[pygame.K_LSHIFT] and keys[pygame.K_z]:
                    self.buffer.Move_Pointer(1)
                    self.canvas_pixels = self.buffer.pointer.snapshot
                    pygame.surfarray.blit_array(self.image,
                                                self.Scale_Image(self.canvas_pixels))
                # Undo
                elif keys[pygame.K_LCTRL] and keys[pygame.K_z]:
                    self.buffer.Move_Pointer(-1)
                    self.canvas_pixels = self.buffer.pointer.snapshot
                    pygame.surfarray.blit_array(self.image,
                                                self.Scale_Image(self.canvas_pixels))

                # Flip the canvas left or right
                if keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]:
                    # up down because pygame inverts the x and y axis, thus fliplr is flipud
                    self.canvas_pixels = np.flipud(self.canvas_pixels)
                    if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                        self.buffer.Add(np.array(self.canvas_pixels))

                if keys[pygame.K_UP] or keys[pygame.K_DOWN]:
                    self.canvas_pixels = np.fliplr(self.canvas_pixels)
                    if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                        self.buffer.Add(np.array(self.canvas_pixels))

                if keys[pygame.K_RETURN] or keys[pygame.K_SPACE]:
                    if not np.allclose(self.canvas_pixels, np.ones((210, 120, 3)) * 255):
                        file_name = len(glob("Images/*.png"))
                        self.Save(f"Paintings/{file_name}.npz")
                        self.Save_Image(f"Images/{file_name}.png")

                        self.buffer = Linked_List()
                        self.canvas_pixels = np.ones((*self.shape[::-1], 3), dtype=np.float32) * 255
                        self.buffer.Add(np.array(self.canvas_pixels))

            if event.type == pygame.KEYUP:
                pygame.surfarray.blit_array(self.image,
                                            self.Scale_Image(self.canvas_pixels))

        # must be before brush update
        self.palette.Update(mouse_x, mouse_y, mouse_pressed, pygame_events)
        self.brush.Update(keys, self.pygame_events, self.tile_size)
        if self.palette.is_selected_color_picker or self.palette.is_moving:
            return "Canvas"

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

                if mouse_pressed[0] and mouse_down:  # if we confirm the fill
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
                if mouse_pressed[0] and mouse_down:  # if we confirm the fill
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

            if self.palette.Get_Kernel_Type() != "constant":
                self.mask = self.Compute_Convolved_Mask(self.brush.Get_Kernel())
            else:
                self.mask = self.Compute_Constant_Mask(self.brush.Get_Kernel(), self.brush.strength)

            pygame.surfarray.blit_array(self.image,
                                        self.Scale_Image(self.Linear_Blend(self.palette.Get_Color(), self.mask)))
        if not self.mouse_curve or ((mouse_x, mouse_y) not in self.mouse_curve and self.is_drawing):
            if not keys[pygame.K_LSHIFT]:
                self.clamp = 0
            if keys[pygame.K_LSHIFT] and self.clamp == 0:

                if not self.mouse_curve:
                    self.mouse_curve.append((mouse_x, mouse_y))
                else:
                    delta_x, delta_y = np.array((mouse_x, mouse_y)) - np.array(self.mouse_curve[-1])
                    if delta_y == 0:
                        self.mouse_curve.append((mouse_x, self.mouse_curve[-1][1]))
                        self.clamp = -1
                    elif delta_x == 0:
                        self.mouse_curve.append((self.mouse_curve[-1][0], mouse_y))
                        self.clamp = 1
            elif self.clamp == -1:  # this means clamping x
                if (mouse_x, self.mouse_curve[-1][1]) not in self.mouse_curve:
                    self.mouse_curve.append((mouse_x, self.mouse_curve[-1][1]))
            elif self.clamp == 1:
                if (self.mouse_curve[-1][0], mouse_y) not in self.mouse_curve:
                    self.mouse_curve.append((self.mouse_curve[-1][0], mouse_y))
            else:
                self.mouse_curve.append((mouse_x, mouse_y))

        if (not mouse_pressed[0]) and not (
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
            self.clamp = 0

        # return the current state, also implenent the return back button here
        return "Canvas"

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

            self.screen.blit(self.palette.color_picker.info, (mouse_x, mouse_y - 30))

        # comment_screen = self.comment_font.render(self.comment, 1, (0, 0, 0))
        # self.screen.blit(comment_screen, (500, 500))
        self.Render_Comment((1500, 200))


if __name__ == "__main__":
    os.environ["SDL_VIDEO_CENTERED"] = "1"
    import ctypes

    pygame.init()
    ctypes.windll.user32.SetProcessDPIAware()
    true_res = (ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1) - 50)
    screen = pygame.display.set_mode(true_res,
                                     pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.HWACCEL | pygame.RESIZABLE,
                                     vsync=1)

    # pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEBUTTONDOWN])

    pygame.display.set_caption("Fine Arts Drawing Simulator")
    pygame_icon = pygame.transform.scale(pygame.image.load("Data/Icons/FADS_icon.png").convert(), (32, 32))
    pygame.display.set_icon(pygame_icon)
    clock = pygame.time.Clock()
    pygame.mouse.set_visible(False)

    saved_folder_path = "Save"
    os.makedirs(saved_folder_path, exist_ok=True)

    into = "/"
    save_path = f"Paintings{into}"
    os.makedirs(save_path, exist_ok=True)

    image_folder_path = "Images"
    os.makedirs(image_folder_path, exist_ok=True)

    canvas = Canvas(screen=screen,
                    start_pos=(50, 300),
                    shape=(120, 210),
                    # shape=(100, 300),
                    tile_size=5,
                    brush_radius=20,
                    saved_folder_path=saved_folder_path)
    running = True
    while running:
        clock.tick(60)
        pygame_events = pygame.event.get()
        screen.fill((153, 207, 224))
        # screen.fill((0, 0, 0))
        mouse_x, mouse_y = pygame.mouse.get_pos()
        canvas.Update(mouse_x, mouse_y, pygame_events)
        canvas.Set_Comment("Thisfsgfaegsfg fdgfa esdgf  fs f eads wefs fgfdhrg")

        canvas.Draw()

        keys = pygame.key.get_pressed()
        for event in pygame_events:
            if event.type == pygame.KEYDOWN:
                # if we are trying to save
                if keys[pygame.K_LCTRL] and keys[pygame.K_s]:
                    while True:
                        name = input("Name of file:")
                        if "/" in name or "\\" in name:
                            print("BRU, I literally said not to put slashes in the file name on the doc")
                        else:
                            break

                    # code for getting the eval
                    while True:
                        try:
                            color_evaluation = float(input("Evaluation of color (-1 to 1):"))
                            if -1 <= color_evaluation <= 1:
                                break
                            else:
                                print("Evaluations can only be from -1 to 1")
                        except:
                            print("Bru U gave some letters ")

                    while True:
                        try:
                            form_evaluation = float(input("Evaluation of form (-1 to 1):"))
                            if -1 <= form_evaluation <= 1:
                                break
                            else:
                                print("Evaluations can only be from -1 to 1")
                        except:
                            print("Bru U gave some letters ")
                    while True:
                        try:
                            acceptability = float(input(
                                "Is art acceptable (0 or 1), keep this at 1, this is for filtering inappropriate stuff:"))
                            if 0 == acceptability or 1 == acceptability:
                                break
                            else:
                                print("Acceptability can only be 0 or 1, try again")
                        except:
                            print("BRu U gave some letters ")
                    # code for getting the comment on the art
                    while True:
                        comment = input("Comment for this art:")
                        if comment != "":
                            break
                        else:
                            print("Please do not give '' as the comment")
                    canvas.Save_Train(f"{save_path}/{name}",
                                      np.array([color_evaluation, form_evaluation, acceptability]),
                                      comment=np.array(comment, dtype=object))
                # if trying to load
                elif keys[pygame.K_LCTRL] and keys[pygame.K_l]:
                    while True:
                        name = input("Load file name:")
                        if os.path.exists(f"Paintings{into}{name}.npz"):
                            break
                        else:
                            print("Name not found in paintings")
                    canvas.Load(f"Paintings/{into}{name}.npz")
                    print()

                elif keys[pygame.K_LCTRL] and keys[pygame.K_p]:
                    while True:
                        file_name = input("Name of file:")
                        if "/" in file_name or "\\" in file_name:
                            print("BRU, I literally said not to put slashes in the file name on the doc")
                        else:
                            break
                    canvas.Save_Image(f"{image_folder_path}/{file_name}.png")
                    print(f"Image saved at {image_folder_path}/{file_name}.png")

            if event.type == pygame.QUIT:
                canvas.Save_State()
                running = False
        pygame.display.flip()
    # sys.exit()
    pygame.quit()
