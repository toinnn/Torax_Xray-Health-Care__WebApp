from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

import json

import io

from ultralytics import YOLO #, settings
# settings
# settings.update({'runs_dir': './runs'})

# from servicer.my_models import my_model , my_model_medium , my_model_small
# from servicer.Transformer_Decoder import decoder


class image_processing_service() :

    def __init__(self):
        __path_yolo_m = "yolov8_models/weights_train-3/best.pt"
        __path_yolo_s = "yolov8_models/weights_train-7/best.pt"
        self.model = YOLO(__path_yolo_m).to("cuda")

    def predict(self , input):
        out = self.model.predict(source=input , imgsz= 640 )
        label_cls = out[-1].names
        print(label_cls)

        # cls = out[0].probs.top1
        out = [ label_cls[ out[i].probs.top1]  for i in range(len(out))]
        return out #label_cls[cls]



# class image_processing_service() :
#     path  = "I.A._Models//xRay_model_0_loss_0.03098666666666666.model"
#     path1 = "I.A._Models//xRay_model_1_loss_0.03098666666666666.model"
#     path_1 = "I.A._Models//state__xRay_model_1_loss_0.03098666666666666.model"
#     path_2 = "I.A._Models//state__xRay_model_2_loss_0.029229629629629626.model"
#     path_3 = "I.A._Models//state__xRay_model_3_loss_0.029611111111111116.model"
#     path_4 = "I.A._Models//state__xRay_model_4_loss_0.06855873015873015.model"
#     path_5 = "I.A._Models//state__xRay_model_5_loss_0.03429714285714285.model"
#     path_6 = "I.A._Models//state__xRay_model_6_loss_0.03428571428571429.model"
#     path_7 = "I.A._Models//state__xRay_model_7_loss_0.03432952380952381.model"
#     path_8 = "I.A._Models//state__xRay_model_8_loss_0.034127619047619046.model"
#     path_9 = "I.A._Models//state__xRay_model_9_loss_0.034420952380952385.model"

#     def __init__(self , model_class = my_model_small , model_args : dict = {} ,model_path = path_9 , dict = None) -> None :
#         self.model = None
#         # self.load_model(model_path)
#         self.load_model(model_path , model_class , model_args )
#         self.dictionary = dict

#     # def load_model(self, path) -> None :
#     #     with open(path, 'rb') as file:
#     #         # model = pickle.load(file)
#     #         self.model = pickle.load(file)
#     #     # self.model = model
#     #     # return model

#     def load_model(self, path , model_class , model_args : dict ) :
#         instance = model_class(**model_args)
#         instance.load_state_dict(torch.load(path))
#         self.model =  instance
    
#     def use_model(self , input):
#         out = self.model(input)
#         return [ self.dictionary[i] for i in out]
    
image_model = image_processing_service()

# if torch.cuda.is_available():
#     image_model.model.setDevice(torch.device("cuda"))
# def pega_argmax(tensor_input) :
#     out = torch.argmax(tensor_input , dim = 2)
#     return out
def show_resuts(contents):
    
    print("Entrou aqui no show resuts")
    # Lê os bytes da imagem
    # contents = await file.read()

    # Converte os bytes para um objeto de imagem usando o Pillow
    image = Image.open(io.BytesIO(contents))
    

    out = image_model.predict(image)
    # print(f"\n\n\nResposta do modelo = { out }\n\n")
    # print(io.BytesIO(contents))
    # print(np.array(image).shape)

    # # Converte a imagem para um array NumPy
    # image_array = np.array(image)

    # # Crie uma representação visual da imagem usando matplotlib
    # plt.imshow(image_array)
    # plt.axis('off')  # Desliga os eixos
    # plt.tight_layout(pad=0)

    # plt.show()
    # plt.close()

    return out

    
         


