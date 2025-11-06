
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import argparse
import torchvision.io as io
from typing import Any, Dict, List, Optional, Tuple, Union

import av

import cv2
import torch
import numpy as np
from typing import Any, Dict, Optional

def write_video(
    filename: str,
    video_array: torch.Tensor,
    fps: float,
    video_codec: str = "mp4v",
    # The following arguments are kept for compatibility but are not used by this OpenCV-based function.
    bit_rate: Optional[int] = None,
    options: Optional[Dict[str, Any]] = None,
    audio_array: Optional[torch.Tensor] = None,
    audio_fps: Optional[float] = None,
    audio_codec: Optional[str] = None,
    audio_options: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Writes a 4d tensor in [T, H, W, C] format into a video file using OpenCV.

    NOTE: This implementation does not handle audio. Audio arguments are ignored.

    Args:
        filename (str): Path where the video will be saved.
        video_array (Tensor[T, H, W, C]): A tensor of frames, assumed to be in uint8 format and RGB channel order.
        fps (float): Video frames per second.
        video_codec (str): The FourCC code for the video codec (e.g., "mp4v", "XVID", "H264").
                           'mp4v' is a good default for .mp4 files.
    """
    # Ensure the tensor is on the CPU and in numpy format, with uint8 data type.
    video_array_numpy = video_array.cpu().to(torch.uint8).numpy()

    # Get video dimensions.
    num_frames, height, width, _ = video_array_numpy.shape

    # Define the codec and create VideoWriter object.
    # The FourCC code is a 4-byte code used to specify the video codec.
    fourcc = cv2.VideoWriter_fourcc(*video_codec)
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not open video writer for path {filename}")
        return

    # Loop through all the frames and write them to the file.
    for frame_rgb in video_array_numpy:
        # OpenCV expects images in BGR format, so we need to convert from RGB.
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    # Release the VideoWriter object.
    out.release()


def load_image(img_path):
    
    image = Image.open(img_path).convert('RGB')
    try:
        dpi = image.info['dpi'][0]
    except:
        dpi = 72
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),   # ImageNet 
                        ])   
    # change image's size to (b, 3, h, w)
    image = transform(image)[:3, :, :].unsqueeze(0)
    return image, dpi


def im_convert(tensor):

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze(0) 
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))   # unnormalized image
    image = image.clip(0, 1)    

    return image

# Instant Style Transfer
class IPST(nn.Module):
    def __init__(self, VGG, content, style):
        super(IPST, self).__init__()

        self.VGG = VGG
        self.content = content
        self.style = transforms.functional.resize(style, self.content.shape[2:], antialias=True)
        self.resolution = 480

        self.content_features = self.get_features(transforms.functional.resize(content, self.resolution, antialias=True), self.VGG)
        self.style_features = self.get_features(transforms.functional.resize(style, self.resolution, antialias=True), self.VGG)
        self.style_gram_matrixs = {layer: self.get_grim_matrix(self.style_features[layer]) for layer in self.style_features}
        
        self.style_net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, kernel_size=1)
        )
        
    def forward(self, x):

        downsample_content = transforms.functional.resize(x, self.resolution, antialias=True)
        # print(downsample_content.shape)

        style = self.style_net(downsample_content)

        style = transforms.functional.resize(style, self.content.shape[2:], antialias=True)

        return style + x
    
    def get_grim_matrix(self, tensor):
        b, c, h, w = tensor.size()
        tensor = tensor.view(b * c, h * w)
        gram_matrix = torch.mm(tensor, tensor.t())
        return gram_matrix
    
    def get_features(self, image, model):
        layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '28': 'conv5_1'}
        features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)    
            if name in layers:
                features[layers[name]] = x
        
        return features
    
    def get_loss(self, target):
        target_features = self.get_features(transforms.functional.resize(target, self.resolution, antialias=True), self.VGG)
        content_loss = torch.mean((target_features['conv4_1'] - self.content_features['conv4_1']) ** 2)
        style_loss = 0
        style_weights = {'conv1_1': 1, 'conv2_1': 1, 'conv3_1': 1, 'conv4_1': 1, 'conv5_1': 1}
        for layer in style_weights:
            target_feature = target_features[layer]  
            target_gram_matrix = self.get_grim_matrix(target_feature)
            style_gram_matrix = self.style_gram_matrixs[layer]

            layer_style_loss = style_weights[layer] * torch.mean((target_gram_matrix - style_gram_matrix) ** 2)
            b, c, h, w = target_feature.shape
            style_loss = style_loss + layer_style_loss / (c * h * w)
        return content_loss, style_loss
    
    def transfer(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        best_loss = float('inf')  
        patience = 10 
        early_stop_counter = 0  
        for epoch in tqdm(range(0, 150)):
            target = self.forward(self.content)
            content_loss, style_loss = self.get_loss(target)
            if epoch==0:
                inital_content_loss = content_loss.item()
                loss = content_loss + style_loss
                initial_loss = loss.item()
            loss = torch.exp(content_loss/inital_content_loss - 1) * content_loss + style_loss # maybe leave the selection of content to style ratio here·····
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            normalized_loss = loss.item()/initial_loss
            # plt.imsave(f'outputs/{epoch}.png', im_convert(target))

            if normalized_loss < best_loss - 0.01:
                best_loss = normalized_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    break
        return target
        

def main():
    seed = 0
    torch.manual_seed(seed)
    parser = argparse.ArgumentParser(description='Image Style Transfer')

    parser.add_argument('--content-image', type=str, help='Path to the content image')
    parser.add_argument('--content-video', type=str, help='Path to the content video')
    parser.add_argument('--style-image', type=str, help='Path to the style image')
    parser.add_argument('--output-folder',type=str, help='Path to output folder', default='./outputs/')
    parser.add_argument('--frame-by-frame',type=bool, help='Transfer videos frame by frame', default=False)

    args = parser.parse_args()

    # Check if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    np.random.seed(seed)
    print(torch.__version__, device)    
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    # Transfer image
    if args.content_image and args.style_image:
        print('Loading VGG model')
        VGG = models.vgg19(weights='DEFAULT').features
        VGG.to(device)
        for parameter in VGG.parameters():
            parameter.requires_grad_(False)
        print('Input:', args.content_image, args.style_image)
        print('Loading input data')
        content_image, dpi = load_image(args.content_image)
        style_image, _ = load_image(args.style_image)
        content_image = content_image.to(device)
        style_image = style_image.to(device)

        ipst = IPST(VGG, content_image, style_image)
        ipst.to(device)
        print('Transfering')
        result = im_convert(ipst.transfer())

        print('Saving')
        plt.imsave(os.path.join(output_folder, os.path.basename(args.content_image)), result, dpi=dpi)

    
    elif args.content_video and args.style_image:
        print('Loading VGG model')
        VGG = models.vgg19(weights='DEFAULT').features
        VGG.to(device)
        for parameter in VGG.parameters():
            parameter.requires_grad_(False)
        print('Input:', args.content_video, args.style_image)
        
        reader = io.VideoReader(args.content_video)
        fps = float(reader.get_metadata()['video']['fps'][0])
        bit_rate = av.open(args.content_video).streams.video[0].bit_rate

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        if args.frame_by_frame:
            total_frame = av.open(args.content_video).streams.video[0].frames
            frames = []
            print('Transfer video frame by frame')
            for i in tqdm(range((total_frame))):
                frame = next(reader)['data'].unsqueeze(0)
                frame = (frame/255. - mean[None, :, None, None]) / std[None, :, None, None]
                frame = frame.to(device)
                if i == 0:
                    content_image = frame
                    style_image,_ = load_image(args.style_image)
                    style_image = style_image.to(device)

                    ipst = IPST(VGG, content_image, style_image)
                    ipst.to(device)
                    frame = ipst.transfer()
                    
                else:
                    with torch.no_grad():
                        frame = ipst.forward(frame)
                frame = frame.cpu().detach()
                frame = (frame * std[None, :, None, None]) + mean[None, :, None, None]
                frame = frame.permute(0, 2, 3, 1)
                frame = frame.clip(0, 1)        
                frame = frame*255
                frame = frame.squeeze(0)
                frames.append(frame.to(torch.uint8))
            frames = torch.stack(frames, 0)
        else:
            print('Loading input data')
            frames = []
            for frame in tqdm(reader):            
                frames.append(frame['data'])
            
            frames = torch.stack(frames, 0).float() / 255
            del frame
            frames = (frames - mean[None, :, None, None]) / std[None, :, None, None]
            frames = frames.to(device)
            print('Transfer the whole video')
            for i in tqdm(range(len(frames))):
                frame = frames[i].unsqueeze(0)
                if i == 0:
                    content_image = frame
                    style_image,_ = load_image(args.style_image)
                    style_image = style_image.to(device)

                    ipst = IPST(VGG, content_image, style_image)
                    ipst.to(device)
                    frame = ipst.transfer()
                    
                else:
                    with torch.no_grad():
                        frame = ipst.forward(frame)
                frames[i] = frame
            frames = frames.cpu().detach()
            frames = (frames * std[None, :, None, None]) + mean[None, :, None, None]
            frames = frames.permute(0, 2, 3, 1)
            frames = frames.clip(0, 1)        
            frames = frames*255
        print('Saving')
        write_video(os.path.join(output_folder, os.path.basename(args.content_video)), frames, fps, bit_rate=bit_rate)

    else:
        print('Please provide --content-image/--content-video and --style-image paths')

if __name__ == '__main__':
    main()
