import os
import time
import pygame
import ctypes
from Canvas import Brush
import numpy as np
from PIL import Image

# import multiprocessing as mp
#
# from transformers import AutoProcessor, LlavaForConditionalGeneration
# from torch import load
# from accelerate import init_empty_weights
# from quanto import qint4, quantize, freeze
# from huggingface_hub import hf_hub_download

os.environ['TRANSFORMERS_CACHE'] = 'Data/Models/'


def Numba_Init():
    for radius in range(2, 51):
        Brush.Numba_Warmup(radius)


#
# def Load_Transformer(connection, is_new):
#     if is_new:
#         print("stated download")
#         # while True:
#         #     try:
#         hf_hub_download(repo_id="10ths/FADS", filename="state.pth", local_dir="Data/Models/")
#             #     break
#             # except:
#             #     time.sleep(2)
#         with open("Data/tmp/downloaded_signal.txt", "w") as f:
#             f.writelines("True")
#
#     processor = AutoProcessor.from_pretrained("Data/Models/empty")
#     state_dict = load("Data/Models/state.pth")
#
#     with init_empty_weights():
#         model = LlavaForConditionalGeneration.from_pretrained("Data/Models/empty")
#
#     print("start")
#     quantize(model, weights=qint4)
#
#     model.load_state_dict(state_dict, assign=True)
#     freeze(model)
#     print("passed")
#
#     def add_roles(text):
#         return "<|user|>\n<image>\n" + text + "<|end|>\n<|assistant|>\n"
#
#     acceptability_eval_prompt = "Is this art or image inappropriate, swear words, nsfw art? If it is inappropriate then give 0. If it is appropriate give a 1"
#     color_eval_prompt = "Rate this art piece's COLOR from 0 to 10. Consider its vibrancy, choice of colors and blending. Please give 1 number"
#     form_eval_prompt = "Rate this art piece's FORM from 0 to 10. Consider its shape, shadows and contrast. Please give 1 number"
#     comment_prompt = "You are an artist, what do you think of this art? Keep your comment as short as possible."
#     while True:
#         path = connection.recv().decode()
#         print(f"Recieved: {path}")
#         if path == "none":
#             break
#         time.sleep(0.05)
#         arr = np.transpose(np.load(path, allow_pickle=True)["inputs"], (1, 0, 2)).astype(np.uint8)
#         img = Image.fromarray(arr)
#
#         inputs_comment = processor(add_roles(comment_prompt), img, return_tensors="pt")
#         inputs_color = processor(add_roles(color_eval_prompt), img, return_tensors="pt")
#         inputs_form = processor(add_roles(form_eval_prompt), img, return_tensors="pt")
#
#         input_acceptability = processor(add_roles(acceptability_eval_prompt), img, return_tensors="pt")
#
#         acceptability = processor.decode(model.generate(**input_acceptability,
#                                          max_new_tokens=3)[0][2:], skip_special_tokens=True).split(" ")[-1]
#         print(acceptability)
#         if acceptability == 0:
#             connection.send("None|0,0,0").encode()
#             continue
#         else:
#             comment = processor.decode(model.generate(**inputs_comment,
#                                                       max_new_tokens=10)[0][2:], skip_special_tokens=True)
#             comment = comment.split(" ")
#
#             comment = str(comment[:len(comment_prompt)])
#             print("comment")
#             print(comment)
#             color_eval = processor.decode(model.generate(**inputs_color, max_new_tokens=3)[0][2:], skip_special_tokens=True).split(" ")[-1]
#             print("color eval")
#             print(color_eval)
#             try:
#                 color_eval = int(color_eval)
#             except:
#                 print("Model didn't produce a number")
#                 color_eval = 11
#             form_eval = processor.decode(model.generate(**inputs_form, max_new_tokens=3)[0][2:], skip_special_tokens=True).split(" ")[-1]
#             try:
#                 form_eval = int(form_eval)
#             except:
#                 print("Model didn't produce a number")
#                 form_eval = 11
#
#         with open("Data/tmp/signal.txt", "w") as f:
#             f.writelines("True")
#
#         connection.send(f"{comment}|{color_eval},{form_eval},{acceptability}".encode())
#

def Run_Loading_Screen(connection):
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pygame.init()
    ctypes.windll.user32.SetProcessDPIAware()

    loading_screen = pygame.display.set_mode((1000, 600),
                                             pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.HWACCEL,
                                             vsync=1)
    loading_image = pygame.image.load("Data/Backgrounds/loading_screen.png").convert_alpha()
    loading_image = pygame.transform.scale(loading_image, (1000, 600))

    new = not os.path.exists("Data/Models/state.pth")

    # if new:
    #     text_font = pygame.font.Font(None, 30)
    #     font_surf = text_font.render(
    #         "Downloading model, if you are in school (use VPN, hotspot shield), this might take a while!", 1, (255, 255, 255))
    #
    # disclaimer_font = pygame.font.Font(None, 50)
    # disclaimer_surf = disclaimer_font.render("Model generation speed is very slow on CPU", 1, (255, 255, 255))
    # warmup_process = mp.Process(target=Numba_Init)
    # probably also start a process for AI inference here

    # transformer_infer = mp.Process(target=Load_Transformer, args=(connection, new))
    # transformer_infer.start()

    # warmup_process.start()

    warming_up = True

    for _ in range(1200):
        pygame_events = pygame.event.get()
        loading_screen.blit(loading_image, (0, 0))
        # if new:
        #     loading_screen.blit(font_surf, (45, 530))
        # loading_screen.blit(disclaimer_surf, (120, 450))

        for event in pygame_events:
            if event.type == pygame.QUIT:
                # warmup_process.terminate()
                pygame.quit()
                return False
        pygame.display.update()
        # if not os.path.exists("Data/tmp/downloaded_signal.txt") and new:
        #     continue

        # if not warmup_process.is_alive():
        #     warming_up = False
        #     warmup_process.terminate()
        #     warmup_process.join()
    pygame.quit()
    return True


if __name__ == "__main__":
    import multiprocessing as mp

    connection_1, connection_2 = mp.Pipe(duplex=True)
    Run_Loading_Screen(connection_2)
    connection_1.send("Paintings_b/hi.npz".encode())
    print(connection_1.recv().decode())
    connection_1.send("none".encode())
