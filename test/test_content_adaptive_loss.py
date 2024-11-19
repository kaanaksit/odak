import sys
sys.path.append("./")

import argparse
from os.path import join, isfile
import odak
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from odak.learn.wave.loss import content_adaptive_loss 
from odak.learn.perception.util import slice_rgbd_targets
from tqdm import tqdm
from os import listdir, makedirs
from os.path import join
from odak.learn.perception import linear_rgb_to_rgb, rgb_to_linear_rgb
from temp_util import *

__title__ = 'Content Adaptive Targeting'


def main():
    number_of_planes = 5
    output_path = "test_output"
    makedirs(output_path, exist_ok=True)
    settings_file_path = "test/holoeye_DragonBunny.txt"
    settings = odak.tools.load_dictionary(settings_file_path)
    scene_name = "Dragon_Bunny"
    settings['general']['output directory'] = join(output_path, scene_name,str(number_of_planes))
    settings['target']['number of depth layers'] = number_of_planes
    settings['target']['target scheme'] = "contrast" 
    print(f"Start {scene_name}, number of planes: {number_of_planes}, target scheme: {settings['target']['target scheme']}")
    process(settings = settings)
                   
def process(settings):
    device = torch.device(settings['general']['device'])
    resolution = settings['spatial light modulator']['resolution']
    output_dir = settings['general']['output directory']
    
    target_image = odak.learn.tools.load_image(
                                               settings["target"]["image filename"],
                                               normalizeby = 2 ** settings["target"]["color depth"],
                                               torch_style = True
                                              ).to(device)[0:3, 0:resolution[0], 0:resolution[1]]
    target_depth = odak.learn.tools.load_image(
                                               settings["target"]["depth filename"],
                                               normalizeby = 2 ** settings["target"]["color depth"],
                                               torch_style = True
                                              ).to(device)
    
    saliency_map = odak.learn.tools.load_image(
                                               "test/data/DragonBunny_saliency.png",
                                               normalizeby = 2 ** settings["target"]["color depth"],
                                               torch_style = True
                                              ).to(device)
    if len(target_depth.shape) > 2:
        target_depth = torch.mean(target_depth, dim = 0)
    if settings["general"]["reverse depth"]:
        target_depth = torch.abs(1-target_depth)
    target_image = rgb_to_linear_rgb(target_image).squeeze()
    
    mt = MultiplaneTargeting(
                         target_image = target_image, 
                         target_depth = target_depth,
                         output_dir = output_dir,
                         number_of_planes = settings['target']['number of depth layers'], 
                         target_scheme = settings['target']['target scheme'],
                         targeting_params = settings["targeting prameters"],
                         pixel_pitch = settings["spatial light modulator"]["pixel pitch"],
                         wavelengths = settings["beam"]["wavelengths"],
                         eyepiece = settings["incoherent propagation"]["eyepiece"],
                         propagation_distance = settings["incoherent propagation"]["propagation distance"],
                         saliency_map = saliency_map,
                         depthmap_dists_range = settings["target"]["volume depth"],
                         device = settings["general"]["device"]
                        )
    targets, masks, overlay_depth = mt.get_targets()
    mt.save()

class MultiplaneTargeting():
    def __init__(
                 self,
                 target_image,
                 target_depth,
                 number_of_planes,
                 output_dir,
                 saliency_map,
                 target_scheme = "contrast",
                 depth_plane_positions = None,
                 targeting_params = None,
                 pixel_pitch = None,
                 wavelengths = None,
                 eyepiece = 0.04, 
                 propagation_distance = 0.12, 
                 depthmap_dists_range = 0.01107, 
                 epsilon = 1e-6,
                 device = "cpu"
                ):
        self.target_image = target_image
        self.target_depth = target_depth
        self.output_dir = output_dir
        self.device = torch.device(device)
        self.number_of_planes = number_of_planes
        self.pixel_pitch = pixel_pitch
        self.wavelengths = wavelengths
        self.targeting_params = targeting_params
        self.eyepiece = eyepiece
        self.propagation_distance = propagation_distance
        self.depthmap_dists_range = depthmap_dists_range
        self.epsilon = epsilon
        self.saliency = saliency_map.to(self.device)
        self.loss_function = content_adaptive_loss(saliency=self.saliency, number_of_planes = number_of_planes, device = self.device)
        self.init_depth_planes(target_scheme, depth_plane_positions)
        
    def init_depth_planes(self, target_scheme, depth_plane_positions):
        if target_scheme == "contrast":
            
            self.divisions = self.depth_plane_optimization()
            self.divisions = get_z_vec(self.divisions, self.number_of_planes)
        elif target_scheme == "naive":
            self.divisions = torch.linspace(0, 1, steps=self.number_of_planes + 1, device=self.device)
            self.saliency = None
        elif target_scheme == "custom" and not isinstance(depth_plane_positions, type(None)):
            self.divisions = depth_plane_positions
            self.saliency = None
        else:
           pass # TODO error handling here
        self.propagation_distances = self.get_distances()
        print(f"The depth plane targets: {self.divisions.tolist()}")
        print(f"The propagation_distances: {self.propagation_distances}")
        
    def depth_plane_optimization(self):
        with torch.no_grad():
            peak_depths = self.depth_plane_initialization()
            
        # Initialize depth planes with CPU tensors initially
        if len(peak_depths) < (self.number_of_planes - 1):
            num_missing = (self.number_of_planes - 1) - len(peak_depths)
            random_depths = torch.rand(num_missing)  # Created on CPU
            cur_depths = torch.sort(torch.cat([peak_depths.cpu(), random_depths]))[0]
        else:
            cur_depths = torch.sort(peak_depths[:(self.number_of_planes - 1)].cpu())[0]
        
        # Move to GPU only when needed
        cur_depths = cur_depths.to(self.device)
        depth_scores = torch.zeros(self.number_of_planes - 1, device=self.device)
        
        # Compute unique values on CPU to save VRAM
        combined = torch.cat((peak_depths.cpu(), cur_depths.cpu()))
        uniques, counts = combined.unique(return_counts=True)
        available_peaks = uniques[counts == 1].to(self.device)
        
        opt_depths = cur_depths.clone().requires_grad_(True)
        optimizer = optim.AdamW([opt_depths], lr=self.targeting_params["learning rate"])

        lowest_loss = float('inf')
        lowest_loss_depths = None
        no_improvement_count = 0
        
        # Pre-compute propagation distances
        self.propagation_distances = self.get_distances()
        
        pbar = tqdm(range(self.targeting_params["epochs"]), ncols=0)
        for epoch in pbar:
            optimizer.zero_grad()
            
            self.divisions = get_z_vec(opt_depths=opt_depths, number_of_planes=self.number_of_planes)
            self.set_targets()
            self.targets = incoherent_focal_stack_rgbd_diff(
                targets=self.targets,
                masks=self.masks,
                distances=self.propagation_distances,
                dx=self.pixel_pitch,
                wavelengths=self.wavelengths,
            )
            
            contrast_sum = 0
            # Process targets in smaller batches if needed
            for k in range(len(self.targets)):
                # Move single target to GPU, process, then move back if needed
                I_zk = self.targets[k].unsqueeze(0)
                
                # Initialize pyramid with proper device
                lpyr = lpyr_dec_2(W=I_zk.shape[-1], H=I_zk.shape[-2], ppd=30, device=self.device, keep_gaussian=True)
                lpyr.decompose(I_zk)

                # Move frequency calculations to CPU if possible
                rho_band = torch.tensor(lpyr.get_freqs(), device=self.device)
                rho_band[lpyr.get_band_count()-1] = 0.1
                sensitivity = contrast_sensitivity_function_Barten1999(rho_band)
                sensitivity = sensitivity / sensitivity.max()

                depth_score = 0
                for b in range(lpyr.get_band_count()-1):
                    # Process bands one at a time
                    upsampled_gband = lpyr.gausspyr_expand(
                        lpyr.get_gband(b+1),
                        [lpyr.get_gband(b).shape[-2], lpyr.get_gband(b).shape[-1]],
                        kernel_a=0.4
                    )
                    upsampled_gband = upsampled_gband.nan_to_num(0.0)
                    lband = lpyr.get_lband(b)
                    
                    # Calculate contrast
                    contrast = torch.abs(lband / (upsampled_gband + self.epsilon))
                    
                    # Interpolate saliency map
                    interpolated_saliency = F.interpolate(
                        self.saliency.unsqueeze(0).unsqueeze(0),
                        contrast.shape[2:4]
                    )
                    interpolated_saliency = max_min_normalization(interpolated_saliency)
                    
                    # Apply saliency and sensitivity
                    contrast = contrast * interpolated_saliency
                    contrast = sensitivity[b] * contrast
                    
                    # Calculate mean and accumulate
                    contrast = torch.mean(contrast)
                    contrast_sum += -torch.log(contrast)
                    depth_score += -torch.log(contrast)
                    
                    # Clear some intermediate tensors
                    del upsampled_gband, lband, interpolated_saliency
                    
                if k < len(opt_depths):
                    depth_scores[k] = depth_score.item()
                
                # Clear individual target processing
                del I_zk, lpyr
            
            # Compute final loss
            loss, range_order_check = self.loss_function(self.masks, contrast_sum=contrast_sum, opt_depths=opt_depths)
            
            loss.backward()
            optimizer.step()
            
            # Update best result if needed
            if (loss.item()) < lowest_loss and range_order_check:
                lowest_loss_depths = opt_depths.detach().clone()
                lowest_loss = loss.item()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            pbar.set_postfix({
                'Positions': f"{opt_depths.tolist()}",
                'Lowest loss': f"{lowest_loss}",
                'Loss': f"{loss.item()}",
            })
            
            # Handle plateau with memory-efficient replacement strategy
            if (self.targeting_params["patience"] > 0 and 
                no_improvement_count >= self.targeting_params["patience"]):
                worst_depth_idx = torch.argmax(depth_scores)
                
                if len(available_peaks) > 0:
                    new_depth = available_peaks[-1].item()
                    available_peaks = available_peaks[:-1]
                else:
                    new_depth = torch.rand(1, device=self.device).item()
                    
                cur_depths = opt_depths.detach().clone()
                cur_depths[worst_depth_idx] = new_depth
                cur_depths, _ = torch.sort(cur_depths)
                
                opt_depths = cur_depths
                opt_depths.requires_grad = True
                optimizer = optim.AdamW([opt_depths], lr=self.targeting_params["learning rate"])
                
                depth_scores = torch.zeros(self.number_of_planes-1, device=self.device)
                no_improvement_count = 0
                
        return lowest_loss_depths
    
    def depth_plane_initialization(self, window_size = 3): 
        with torch.no_grad():
            contrasts = get_contrast(self.target_image, self.saliency) # Get the over all contrasts for each pyramid layer
            
        depth_to_contrast = torch.zeros((len(contrasts), 256), dtype=torch.float32).to(self.device)
        for idx, contrast in enumerate(contrasts):
            contrast = contrast.squeeze(0)
            contrast = max_min_normalization(contrast)
            
            depth_image_ = F.interpolate(self.target_depth.unsqueeze(0).unsqueeze(0), contrast.shape[1:3]).clone()
            depth_image_ = (depth_image_*255).to(torch.uint8)

            inner_list = torch.zeros(256, dtype=torch.float32).to(self.device)
            for depth in range(256):
                mask = torch.where(depth_image_ == depth, torch.tensor(255, dtype=torch.uint8), torch.tensor(0, dtype=torch.uint8))
                mask = mask.to(torch.uint8).squeeze(0)
                slice_contrast = torch.sum(contrast * mask).item()
                slice_contrast = slice_contrast/torch.sum(mask).item() if torch.sum(mask).item() !=0 else 0
                inner_list[depth] = slice_contrast if not torch.isnan(torch.tensor(slice_contrast).to(self.device)) else 0

            depth_to_contrast[idx] = inner_list

        depth_to_contrast_sum = torch.sum(depth_to_contrast, dim=0).unsqueeze(0).unsqueeze(0).to(self.device)
        depth_to_contrast_sum = max_min_normalization(depth_to_contrast_sum, type="torch")
        depth_to_contrast_sum_reverse = max_min_normalization(-depth_to_contrast_sum, type="torch")
        depth_to_contrast_sum_reverse = depth_to_contrast_sum_reverse.to(self.device).squeeze(0).squeeze(0)
        
        boundaries = torch.empty((1)).to(self.device)
        boundaries = find_local_maxima(depth_to_contrast_sum_reverse, 25)
        
        contrast_sum_per_region = torch.zeros((len(boundaries)-1)).to(self.device)   
        pairs = torch.zeros((len(boundaries)-1, 2)).to(self.device)
        for idx, boundary in enumerate(boundaries):
            if idx < len(boundaries)-1:
                contrast_sum_per_region[idx] = torch.sum(depth_to_contrast_sum[boundary: boundaries[idx+1]])
                pairs[idx]= torch.tensor([boundary, boundaries[idx+1]], dtype=torch.float32).to(self.device)
                
        boundary_indices = torch.argsort(contrast_sum_per_region, descending=True)
        # Reorder pairs based on sorted indices
        boundaries = pairs[boundary_indices]
        # Flatten and get unique boundaries
        boundaries = torch.unique(boundaries.flatten())
        
        # Normalize to [0, 1]
        boundaries = boundaries.float() / 255.0
        return torch.clip(boundaries, 0, 1)

    def get_distances(self):
        return compute_focal_stack_distances(self.eyepiece, self.number_of_planes, self.propagation_distance, self.depthmap_dists_range, self.device)
    
    def set_targets(self, differentiable = True):
        if differentiable:
            self.targets, self.masks = slice_rgbd_targets_diff(self.target_image, self.target_depth, self.divisions, temperature=0.001)
        else:
            self.targets, self.masks = slice_rgbd_targets(self.target_image, self.target_depth, self.divisions)
            
    def get_targets(self):
        self.set_targets(differentiable=False)
        self.targets = incoherent_focal_stack_rgbd(
                                                   targets = self.targets, 
                                                   masks =  self.masks, 
                                                   distances = self.propagation_distances, 
                                                   dx =  self.pixel_pitch, 
                                                   wavelengths = self.wavelengths,
                                                  )
        overlay_depth = torch.zeros_like(self.masks[0], dtype = torch.float32).to(self.device)
        for idx, mask in enumerate(self.masks):
            overlay_depth += idx*(mask / (1e-12 + mask.max()))
        return (self.targets.detach().clone(), self.masks.detach().clone(), overlay_depth.detach().clone())
    
    def save(self):
        odak.tools.check_directory(self.output_dir)
        targets, masks, overlay_depth = self.get_targets()
        if not isinstance(self.saliency, type(None)):
            odak.learn.tools.save_image(
                                        join(self.output_dir, f"00_saliency.png"), 
                                        self.saliency,
                                        cmin = self.saliency.min(),
                                        cmax = self.saliency.max()
                                    )
        odak.learn.tools.save_image(
                                    join(self.output_dir, f"01_overlaydepth.png"), 
                                    overlay_depth,
                                    cmin = overlay_depth.min(),
                                    cmax = overlay_depth.max()
                                   )
        for i in range(len(targets)):
            mask = masks[i]
            focal_stack_frame = targets[i]
            focal_stack_frame = linear_rgb_to_rgb(focal_stack_frame)
            odak.learn.tools.save_image(
                                        join(self.output_dir, f"02_focalstack_{i}.png"), 
                                        focal_stack_frame,
                                        cmin = focal_stack_frame.min(),
                                        cmax = focal_stack_frame.max()
                                       )
            odak.learn.tools.save_image(
                                        join(self.output_dir, f"03_mask_{i}.png"), 
                                        mask.to(torch.float32),
                                        cmin = 0,
                                        cmax = 1
                                       )
            
if __name__ == "__main__":
    sys.exit(main())