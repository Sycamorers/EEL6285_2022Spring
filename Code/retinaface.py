import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from nets.facenet import Facenet
from nets_retinaface.retinaface import RetinaFace
from utils.anchors import Anchors
from utils.config import cfg_mnet, cfg_re50
from utils.utils import (Alignment_1, compare_faces, letterbox_image,
                         preprocess_input)
from utils.utils_bbox import (decode, decode_landm, non_max_suppression,
                              retinaface_correct_boxes)



class Retinaface(object):
    _defaults = {
        
        #  Retinaface's trained weight path
        "retinaface_model_path" : 'model_data/Retinaface_mobilenet0.25.pth',
        #  The backbone networks used by retinaface are mobilenet and resnet50
        "retinaface_backbone"   : "mobilenet",
        #  Only prediction frames with scores greater than the confidence level in retinaface will be retained
        "confidence"            : 0.5,
        #  The size of nms_iou used for non-extreme suppression in retinaface
        "nms_iou"               : 0.3,
        #  The input_shape can be adjusted according to the size of the input image.
        "retinaface_input_shape": [640, 640, 3],
        #  Whether the image size limit is required.
        "letterbox_image"       : True,
        #  Facenet's trained weight path
        "facenet_model_path"    : 'model_data/facenet_mobilenet.pth',
        #  The backbone network used by facenet, mobilenet and inception_resnetv1
        "facenet_backbone"      : "mobilenet",
        #  The size of the input image used by facenet
        "facenet_input_shape"   : [160, 160, 3],
        #  Face distance thresholds used by facenet
        "facenet_threhold"      : 0.9,
        # No GPU can be set to False
        "cuda"                  : False
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # Initialize Retinaface
    def __init__(self, encoding=0, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # config information for different backbone networks
        if self.retinaface_backbone == "mobilenet":
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50

        # A priori box generation
        self.anchors = Anchors(self.cfg, image_size=(self.retinaface_input_shape[0], self.retinaface_input_shape[1])).get_anchors()
        self.generate()

        try:
            self.known_face_encodings = np.load("model_data/{backbone}_face_encoding.npy".format(backbone=self.facenet_backbone))
            self.known_face_names     = np.load("model_data/{backbone}_names.npy".format(backbone=self.facenet_backbone))
        except:
            if not encoding:
                print("If the loading of existing face features fails, please check whether the relevant face feature file is generated under model_data.")
            pass
    # Get all categories
    def generate(self):
        # Load model with weights
        self.net        = RetinaFace(cfg=self.cfg, phase='eval', pre_train=False).eval()
        self.facenet    = Facenet(backbone=self.facenet_backbone, mode="predict").eval()

        print('Loading weights into state dict...')
        state_dict = torch.load(self.retinaface_model_path,map_location=torch.device('cpu'))
        self.net.load_state_dict(state_dict)

        state_dict = torch.load(self.facenet_model_path,map_location=torch.device('cpu'))
        self.facenet.load_state_dict(state_dict, strict=False)

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

            self.facenet = nn.DataParallel(self.facenet)
            self.facenet = self.facenet.cuda()
        print('Finished!')

    def encode_face_dataset(self, image_paths, names):
        face_encodings = []
        for index, path in enumerate(tqdm(image_paths)):
            # Open face image
            image       = np.array(Image.open(path), np.float32)
            # Make a backup of the input image
            old_image   = image.copy()
            # Calculate the height and width of the input image
            im_height, im_width, _ = np.shape(image)
            # Calculate the scale, which is used to convert the obtained prediction frame to the height and width of the original image
            scale = [
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
            ]
            scale_for_landmarks = [
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                np.shape(image)[1], np.shape(image)[0]
            ]
            if self.letterbox_image:
                image = letterbox_image(image, [self.retinaface_input_shape[1], self.retinaface_input_shape[0]])
                anchors = self.anchors
            else:
                anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

            # Pass the processed images into the Retinaface network for prediction
            
            with torch.no_grad():
                # Image pre-processing, normalization.
                image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)

                if self.cuda:
                    image               = image.cuda()
                    anchors             = anchors.cuda()

                loc, conf, landms = self.net(image)
                # Decode the prediction frame
                boxes   = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])
                # Gain confidence in the prediction results
                conf    = conf.data.squeeze(0)[:, 1:2]
                # Decoding of key points of the face
                landms  = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])

                # Stacking of face detection results
                boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
                boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

                if len(boxes_conf_landms) <= 0:
                    print(names[index], ": No face detected")
                    continue

                # If letterbox_image is used, remove the gray bar.
                if self.letterbox_image:
                    boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                        np.array([self.retinaface_input_shape[0], self.retinaface_input_shape[1]]), np.array([im_height, im_width]))

            boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
            boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

           # Select the largest face frame.
            best_face_location  = None
            biggest_area        = 0
            for result in boxes_conf_landms:
                left, top, right, bottom = result[0:4]

                w = right - left
                h = bottom - top
                if w * h > biggest_area:
                    biggest_area = w * h
                    best_face_location = result

            # Captured images encoding
            crop_img = old_image[int(best_face_location[1]):int(best_face_location[3]), int(best_face_location[0]):int(best_face_location[2])]
            landmark = np.reshape(best_face_location[5:],(5,2)) - np.array([int(best_face_location[0]),int(best_face_location[1])])
            crop_img,_ = Alignment_1(crop_img,landmark)

            crop_img = np.array(letterbox_image(np.uint8(crop_img),(self.facenet_input_shape[1],self.facenet_input_shape[0])))/255
            crop_img = crop_img.transpose(2, 0, 1)
            crop_img = np.expand_dims(crop_img,0)
            # Use the image to calculate the feature vector of length 128
            with torch.no_grad():
                crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
                if self.cuda:
                    crop_img = crop_img.cuda()

                # Using facenet_model to calculate the eigenvector of length 128
                face_encoding = self.facenet(crop_img)[0].cpu().numpy()
                face_encodings.append(face_encoding)

        np.save("model_data/{backbone}_face_encoding.npy".format(backbone=self.facenet_backbone),face_encodings)
        np.save("model_data/{backbone}_names.npy".format(backbone=self.facenet_backbone),names)

 
    def detect_image(self, image):

        # Make a backup of the input image, which is used later for drawing
        old_image   = image.copy()
        
         # Convert images to numpy form
        image       = np.array(image, np.float32)

        # Retinaface detection section - start
        # Calculate the height and width of the input image
        im_height, im_width, _ = np.shape(image)
        # Calculate the scale, which is used to convert the obtained prediction frame to the height and width of the original image
        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]


         # letterbox_image can add gray bars to the image to achieve undistorted resize
        if self.letterbox_image:
            image = letterbox_image(image, [self.retinaface_input_shape[1], self.retinaface_input_shape[0]])
            anchors = self.anchors
        else:
            anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()




        # Pass the processed images into the Retinaface network for prediction
        with torch.no_grad():
            # Image pre-processing, normalization.
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)



            if self.cuda:
                anchors = anchors.cuda()
                image   = image.cuda()



            # Incoming network for prediction
            loc, conf, landms = self.net(image)
            # Decoding of the Retinaface network, we end up with prediction frames
            # Decoding and non-extreme suppression of the prediction results


            boxes   = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])

            conf    = conf.data.squeeze(0)[:, 1:2]
            
            landms  = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])

            # Stacking of face detection results
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)

            # the faces with the highest scores in a certain region can be filtered out 
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

        
            # If there is no prediction box then return to original image
            if len(boxes_conf_landms) <= 0:
                return old_image



            # If letterbox_image is used, remove the gray bar.
            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                    np.array([self.retinaface_input_shape[0], self.retinaface_input_shape[1]]), np.array([im_height, im_width]))

            boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
            boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks
       # Retinaface detection section - end









        # Facenet coding section - start
        face_encodings = []
        for boxes_conf_landm in boxes_conf_landms:



            # Image capture, face correction
            boxes_conf_landm    = np.maximum(boxes_conf_landm, 0)
            crop_img            = np.array(old_image)[int(boxes_conf_landm[1]):int(boxes_conf_landm[3]), int(boxes_conf_landm[0]):int(boxes_conf_landm[2])]
            landmark            = np.reshape(boxes_conf_landm[5:],(5,2)) - np.array([int(boxes_conf_landm[0]),int(boxes_conf_landm[1])])
            crop_img, _         = Alignment_1(crop_img, landmark)




            # Face encoding
            crop_img = np.array(letterbox_image(np.uint8(crop_img),(self.facenet_input_shape[1],self.facenet_input_shape[0])))/255
            crop_img = np.expand_dims(crop_img.transpose(2, 0, 1),0)
            with torch.no_grad():
                crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
                if self.cuda:
                    crop_img = crop_img.cuda()
                # Compute feature vectors of length 128 using facenet_model
                face_encoding = self.facenet(crop_img)[0].cpu().numpy()
                face_encodings.append(face_encoding)
        # Facenet coding section - end




        # Face Feature Comparison - Start
        face_names = []
        for face_encoding in face_encodings:
            # Take a face and compare it with all the faces in the database and calculate the score
            matches, face_distances = compare_faces(self.known_face_encodings, face_encoding, tolerance = self.facenet_threhold)
            # Default settings
            name = "Unknown"
            # Fetch the rating of this most recent face
            # Fetch the serial number of the closest known face for the current incoming face
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]: 
                name = self.known_face_names[best_match_index]
            face_names.append(name)
        # Face Feature Comparison - End

        
        mask_path = r'/Users/huangzijing/Desktop/facenet-retinaface-pytorch-main/img/mask.jpg'
        mask = cv2.imread(mask_path)
        for i, b in enumerate(boxes_conf_landms):
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            # b[0]-b[3] are the coordinates of the face frame, b[4] is the score
            cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(old_image, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]: 
                name = self.known_face_names[best_match_index]

                # Add facemask at this position
                mask = cv2.resize(mask, (abs(b[2]-b[0]), abs(b[3]-b[1])))
                old_image[b[1]:b[3],b[0]:b[2]] = mask

            # b[5]-b[14] are the coordinates of the key points of the face
            cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)
            
            name = face_names[i]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(old_image, name, (b[0] , b[3] - 15), font, 0.75, (255, 255, 255), 2) 

        return old_image


    def get_FPS(self, image, test_interval):
        # Make a backup of the input image to be used later for plotting
        old_image   = image.copy()
        # Convert the image to numpy form
        image       = np.array(image, np.float32)

        # Retinaface detection section - start
        # Calculate the height and width of the input image
        im_height, im_width, _ = np.shape(image)
        # Calculate scale, used to convert the obtained prediction frame to the height and width of the original image
        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]

        # letterbox_image can add gray bars to the image to achieve undistorted resize
        if self.letterbox_image:
            image = letterbox_image(image, [self.retinaface_input_shape[1], self.retinaface_input_shape[0]])
            anchors = self.anchors
        else:
            anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

        # Predict the processed images by passing them into the Retinaface network
        with torch.no_grad():
            # Image preprocessing, normalization.
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)

            if self.cuda:
                anchors = anchors.cuda()
                image   = image.cuda()

              # incoming network for prediction
            loc, conf, landms = self.net(image)
            # Decoding of the Retinaface network, we end up with prediction frames
            # Decode and non-extreme suppress the prediction results
            boxes   = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])

            conf    = conf.data.squeeze(0)[:, 1:2]
            
            landms  = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])
            
            # Stacking of face detection results
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)
            
        if len(boxes_conf_landms)>0:
            # If letterbox_image is used, remove the gray bar.
            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                    np.array([self.retinaface_input_shape[0], self.retinaface_input_shape[1]]), np.array([im_height, im_width]))

            boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
            boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks
            # Retinaface detection section - end
            
            # Facenet encoding section-begin
            face_encodings = []
            for boxes_conf_landm in boxes_conf_landms:
                # Image capture, face correction
                boxes_conf_landm    = np.maximum(boxes_conf_landm, 0)
                crop_img            = np.array(old_image)[int(boxes_conf_landm[1]):int(boxes_conf_landm[3]), int(boxes_conf_landm[0]):int(boxes_conf_landm[2])]
                landmark            = np.reshape(boxes_conf_landm[5:],(5,2)) - np.array([int(boxes_conf_landm[0]),int(boxes_conf_landm[1])])
                crop_img, _         = Alignment_1(crop_img, landmark)

                # Face encoding
                crop_img = np.array(letterbox_image(np.uint8(crop_img),(self.facenet_input_shape[1],self.facenet_input_shape[0])))/255
                crop_img = np.expand_dims(crop_img.transpose(2, 0, 1),0)
                with torch.no_grad():
                    crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
                    if self.cuda:
                        crop_img = crop_img.cuda()

                    # Compute a feature vector of length 128 using facenet_model
                    face_encoding = self.facenet(crop_img)[0].cpu().numpy()
                    face_encodings.append(face_encoding)
            # Facenet encoding section - end

            # Facenet feature matching-start
            face_names = []
            for face_encoding in face_encodings:
                # Take a face and compare it with all the faces in the database and calculate the score
                matches, face_distances = compare_faces(self.known_face_encodings, face_encoding, tolerance = self.facenet_threhold)
                name = "Unknown"
                # Fetch the rating of this most recent face
                # Fetch the serial number of the closest known face for the current incoming face
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]: 
                    name = self.known_face_names[best_match_index]
                face_names.append(name)
            # Face Feature Matching - End
        
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                # incoming network for prediction
                loc, conf, landms = self.net(image)
                # Decoding of the Retinaface network, we end up with prediction frames
                # Decode and non-extreme suppress the prediction results
                boxes   = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])

                conf    = conf.data.squeeze(0)[:, 1:2]
                
                landms  = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])
                
                # Stacking of face detection results
                boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
                boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)
            
            if len(boxes_conf_landms) > 0:
                # If letterbox_image is used, remove the gray bar.
                if self.letterbox_image:
                    boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                        np.array([self.retinaface_input_shape[0], self.retinaface_input_shape[1]]), np.array([im_height, im_width]))

                boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
                boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks
                # Retinaface detection section - end
                
                # Facenet encoding section - start
                face_encodings = []
                for boxes_conf_landm in boxes_conf_landms:
                    # Image capture, face correction
                    boxes_conf_landm    = np.maximum(boxes_conf_landm, 0)
                    crop_img            = np.array(old_image)[int(boxes_conf_landm[1]):int(boxes_conf_landm[3]), int(boxes_conf_landm[0]):int(boxes_conf_landm[2])]
                    landmark            = np.reshape(boxes_conf_landm[5:],(5,2)) - np.array([int(boxes_conf_landm[0]),int(boxes_conf_landm[1])])
                    crop_img, _         = Alignment_1(crop_img, landmark)

                    # Face encoding
                    crop_img = np.array(letterbox_image(np.uint8(crop_img),(self.facenet_input_shape[1],self.facenet_input_shape[0])))/255
                    crop_img = np.expand_dims(crop_img.transpose(2, 0, 1),0)
                    with torch.no_grad():
                        crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
                        if self.cuda:
                            crop_img = crop_img.cuda()

                        # Compute feature vectors of length 128 using facenet_model
                        face_encoding = self.facenet(crop_img)[0].cpu().numpy()
                        face_encodings.append(face_encoding)
                # Facenet coding section - end

                # Face Feature Matching - Start
                face_names = []
                for face_encoding in face_encodings:
                    # Take a face and compare it with all the faces in the database and calculate the score
                    matches, face_distances = compare_faces(self.known_face_encodings, face_encoding, tolerance = self.facenet_threhold)
                    name = "Unknown"
                    # Fetch the rating of this recent face
                    # Fetch the serial number of the closest known face for the current incoming face
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]: 
                        name = self.known_face_names[best_match_index]
                    face_names.append(name)
                # Face Feature Matching - End
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
