import re

import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms

from motif import weights
from motif.tddfa_v3.face_box.retinaface.network import RetinaFace
from motif.tddfa_v3.face_box.facelandmark.nets.large_base_lmks_net import LargeBaseLmksNet
from motif.tddfa_v3.face_box.retinaface.prior_box import priorbox
from motif.tddfa_v3.util.preprocess import load_lm3d
from motif.tddfa_v3.model import networks

# Alignment currently resizes the full frame before taking a 224 px crop.
# Bound that intermediate so bad landmarks cannot request unbounded GPU memory.
MAX_ALIGNMENT_DIMENSION = 4096


def _load_retinaface_state_dict(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    return {re.sub(r"^model\.", "", key): value for key, value in state_dict.items()}

def batch_resize_normalize_pad(images, max_size):
    """
    Pure PyTorch implementation of A.LongestMaxSize + A.Normalize + Pad

    Args:
        images: Tensor of shape (B, C, H, W) in range [0, 255]
        max_size: Maximum size for the longest dimension and target size for padding

    Returns:
        - images: Transformed tensor of shape (B, C, max_size, max_size)
        - pads: Tensor of shape (4,) with (left_pad, right_pad, top_pad, bottom_pad)
    """
    batch_size, channels, height, width = images.shape
    assert channels == 3, "Input images must have 3 channels (RGB)"
    assert batch_size > 0, "Batch size must be greater than 0"

    # Convert to float and normalize to [0, 1]
    images = images.float() / 255.0

    # Calculate scale factor (equivalent to LongestMaxSize)
    scale_factor = max_size / max(height, width)
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    # Resize using bilinear interpolation
    images = F.interpolate(
        images, size=(new_height, new_width), mode="bilinear", align_corners=False
    )

    # Calculate padding
    target_height, target_width = max_size, max_size
    pad_height = target_height - new_height
    pad_width = target_width - new_width

    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad
    if pad_height == 0 and pad_width == 0:
        pads = torch.tensor([0, 0, 0, 0], dtype=torch.int32, device=images.device)
    else:
        pads = (left_pad, right_pad, top_pad, bottom_pad)  # (left, right, top, bottom)
        images = F.pad(images, pads, value=0.0)  # Pad with zeros (black)
        pads = torch.tensor(
            pads, dtype=torch.int32, device=images.device
        )  # Convert pads to tensor

    # ImageNet normalization (equivalent to A.Normalize())
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    images = (images - mean) / std

    return images, pads


class CoeffExtractor:
    """CUPID's FastUVExtractor with the UV-rendering head removed: same RetinaFace
    detection, 106-landmark refinement and face alignment, but the aligned faces go
    to the 3DMM regressor (net_recon) and we keep the coefficients instead of
    rendering a texture. No nvdiffrast / face_model.npy needed."""

    def __init__(
        self,
        retina_model_path: str | None = None,
        landmark_model_path: str | None = None,
        rec_model_path: str | None = None,
        input_size: int = 224,
        max_size: int = 512,
        device: str = "cuda",
        weights_dir: str | None = None,
    ):
        retina_model_path = retina_model_path or weights.get_tddfa_asset(
            weights.RETINAFACE_FILE, weights_dir
        )
        landmark_model_path = landmark_model_path or weights.get_tddfa_asset(
            weights.LANDMARK_FILE, weights_dir
        )
        rec_model_path = rec_model_path or weights.get_tddfa_asset(
            weights.NET_RECON_FILE, weights_dir
        )

        self.device = device
        self.input_size = input_size
        self.max_size = max_size

        ### Retinaface parameters
        self.prior_box = priorbox(
            min_sizes=[[16, 32], [64, 128], [256, 512]],
            steps=[8, 16, 32],
            clip=False,
            image_size=(max_size, max_size),
        ).to(device)
        self.variance = [0.1, 0.2]
        self.confidence_threshold = 0.7
        self.nms_threshold = 0.4
        self.top_k = 10

        # Landmark parameters
        self.enlarge_ratio = 1.35
        self.selected_landmarks = [74, 83, 54, 84, 90]

        ### Load retinaface detection model
        self.retinaface = RetinaFace(
            name="Resnet50",
            pretrained=False,
            return_layers={"layer2": 1, "layer3": 2, "layer4": 3},
            in_channels=256,
            out_channels=256,
        ).to(device)
        self.retinaface.load_state_dict(
            _load_retinaface_state_dict(retina_model_path)
        )
        self.retinaface.eval()

        ### Load landmark model
        self.landmark_model = LargeBaseLmksNet(infer=False).to(device)
        checkpoint = torch.load(
            landmark_model_path, weights_only=False, map_location=device
        )
        self.landmark_model.load_state_dict(
            {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()},
            strict=False,
        )
        self.landmark_model.eval()
        self.landmarks_3d = torch.tensor(load_lm3d()).to(device)  # Load 3D landmarks

        ### Load 3DMM coefficient regressor (same weights CUPID uses for UV)
        self.net_recon = networks.ReconNetWrapper()
        self.net_recon.load_state_dict(
            torch.load(rec_model_path, map_location="cpu")["net_recon"]
        )
        self.net_recon = self.net_recon.to(device)
        self.net_recon.eval()

    def process_retina_annotations(
        self,
        locs,
        lands,
        scores_batch,
        pads,
        original_height,
        original_width,
    ):
        """
        Complete batch face detection pipeline from raw predictions to final coordinates

        Args:
            locs: Raw location predictions from RetinaFace
            lands: Raw landmark predictions from RetinaFace
            scores_batch: Confidence scores [batch_size, num_priors]
            prior_box: Prior boxes for decoding
            variance: Variance values for decoding
            max_size: Size used for padding/resizing (512)
            pads: Padding values [left, right, top, bottom]
            original_height: Original image height
            original_width: Original image width
            confidence_threshold: Minimum confidence for keeping detections
            nms_threshold: NMS threshold
            device: PyTorch device
            topk_limit: Maximum detections per image to consider for NMS

        Returns:
            frame_boxes: [batch_size, 4] - final boxes in original coordinates
            frame_scores: [batch_size] - confidence scores
            frame_landmarks: [batch_size, 5, 2] - landmarks in original coordinates
            frame_valid: [batch_size] - boolean mask for valid detections
        """
        actual_batch_size = scores_batch.shape[0]

        # Decode coordinates: reverse training transformation + scale back to resized image size
        boxes_batch = (
            self.decode_batch(locs, self.prior_box, self.variance) * self.max_size
        )  # From [0,1] to [0,512]
        landmarks_batch = (
            self.decode_landm_batch(lands, self.prior_box, self.variance)
            * self.max_size
        )  # From [0,1] to [0,512]

        # Get top-k detections per image to limit NMS computation
        # topk is faster than thresholding first then sorting, plus homogeneous tensor shapes
        topk_scores, topk_indices = torch.topk(
            scores_batch, min(self.top_k, scores_batch.shape[1]), dim=1
        )

        # Gather corresponding boxes and landmarks
        # For each position [i, j]:
        batch_indices = (
            torch.arange(actual_batch_size, device=self.device)
            .unsqueeze(1)
            .expand(-1, topk_indices.shape[1])
        )
        topk_boxes = boxes_batch[batch_indices, topk_indices]
        topk_landmarks = landmarks_batch[batch_indices, topk_indices]

        # Flatten for batched_nms
        flat_boxes = topk_boxes.reshape(-1, 4)
        flat_scores = topk_scores.reshape(-1)
        flat_landmarks = topk_landmarks.reshape(-1, 10)

        # Create batch IDs
        batch_ids = torch.arange(
            actual_batch_size, device=self.device
        ).repeat_interleave(topk_scores.shape[1])

        # Apply batched NMS - it will naturally filter low scores
        keep = batched_nms(flat_boxes, flat_scores, batch_ids, self.nms_threshold)

        # Filter by confidence threshold after NMS (more efficient)
        confidence_keep = flat_scores[keep] > self.confidence_threshold
        final_keep = keep[confidence_keep]

        final_boxes = flat_boxes[final_keep]
        final_scores = flat_scores[final_keep]
        final_landmarks = flat_landmarks[final_keep]
        final_batch_ids = batch_ids[final_keep]

        # No guarantee of one detection per frame: keep the highest scoring detection per frame
        # final_batch_ids is sorted by highest score due to NMS
        frame_boxes = torch.zeros(actual_batch_size, 4, device=self.device)
        frame_scores = torch.zeros(actual_batch_size, device=self.device)
        frame_landmarks = torch.zeros(actual_batch_size, 10, device=self.device)
        frame_valid = torch.zeros(
            actual_batch_size, dtype=torch.bool, device=self.device
        )

        if len(final_batch_ids) > 0:
            # Get unique batch IDs and their first occurrence indices
            unique_batch_ids = torch.unique(final_batch_ids, sorted=True)

            # For each unique batch ID, find its first occurrence
            for i, batch_id in enumerate(unique_batch_ids):
                first_occurrence = (final_batch_ids == batch_id).nonzero(as_tuple=True)[
                    0
                ][0]
                frame_boxes[batch_id] = final_boxes[first_occurrence]
                frame_scores[batch_id] = final_scores[first_occurrence]
                frame_landmarks[batch_id] = final_landmarks[first_occurrence]
                frame_valid[batch_id] = True

        # Reshape landmarks to [batch_size, 5, 2]
        frame_landmarks = frame_landmarks.reshape(-1, 5, 2)

        # Calculate scale factor
        left_pad, right_pad, top_pad, bottom_pad = pads
        scale_factor = max(original_height, original_width) / self.max_size

        # Adjust boxes and landmarks back to original image coordinates
        if pads.sum() > 0:
            # Adjust boxes
            frame_boxes[:, 0] -= left_pad  # x1
            frame_boxes[:, 1] -= top_pad  # y1
            frame_boxes[:, 2] -= left_pad  # x2
            frame_boxes[:, 3] -= top_pad  # y2

            # Adjust landmarks
            frame_landmarks[:, :, 0] -= left_pad  # x coordinates
            frame_landmarks[:, :, 1] -= top_pad  # y coordinates

        frame_landmarks *= scale_factor
        frame_boxes *= scale_factor

        # Clamp boxes and landmarks to original image boundaries
        frame_boxes[:, [0, 2]] = frame_boxes[:, [0, 2]].clamp(0, original_width - 1)
        frame_boxes[:, [1, 3]] = frame_boxes[:, [1, 3]].clamp(0, original_height - 1)
        frame_landmarks[:, :, 0] = frame_landmarks[:, :, 0].clamp(0, original_width - 1)
        frame_landmarks[:, :, 1] = frame_landmarks[:, :, 1].clamp(
            0, original_height - 1
        )

        return frame_boxes, frame_scores, frame_landmarks, frame_valid

    def decode_batch(self, loc, priors, variances):
        """Decode locations from predictions using priors to undo the encoding we did for offset regression at train time.
        Args:
            loc: location predictions for loc layers,
                Shape: [batch_size, num_priors, 4]
            priors: Prior boxes in center-offset form.
                Shape: [num_priors, 4] - will be broadcasted across batch
            variances: Variances of priorboxes
        Return:
            decoded bounding box predictions, Shape: [batch_size, num_priors, 4]
        """
        # Handle broadcasting: priors [num_priors, 4] -> [1, num_priors, 4]
        if priors.dim() == 2:
            priors = priors.unsqueeze(0)  # [10752, 4] -> [1, 10752, 4]

        # PyTorch will automatically broadcast [1, 10752, 4] to [batch_size, 10752, 4]
        boxes = torch.cat(
            (
                priors[:, :, :2] + loc[:, :, :2] * variances[0] * priors[:, :, 2:],
                priors[:, :, 2:] * torch.exp(loc[:, :, 2:] * variances[1]),
            ),
            dim=2,
        )
        boxes[:, :, :2] -= boxes[:, :, 2:] / 2  # Convert center to top-left
        boxes[:, :, 2:] += boxes[:, :, :2]  # Convert width/height to bottom-right
        return boxes

    def decode_landm_batch(self, pre, priors, variances):
        """Decode landmarks from predictions using priors to undo the encoding we did for offset regression at train time.
        Args:
            pre: landmark predictions for loc layers,
                Shape: [batch_size, num_priors, 10]
            priors: Prior boxes in center-offset form.
                Shape: [num_priors, 4] or [batch_size, num_priors, 4] (broadcasted)
            variances: Variances of priorboxes
        Return:
            decoded landmark predictions, Shape: [batch_size, num_priors, 10]
        """
        # Handle broadcasting: priors [num_priors, 4] -> [1, num_priors, 4]
        if priors.dim() == 2:
            priors = priors.unsqueeze(0)  # [num_priors, 4] -> [1, num_priors, 4]

        return torch.cat(
            (
                priors[:, :, :2] + pre[:, :, :2] * variances[0] * priors[:, :, 2:],
                priors[:, :, :2] + pre[:, :, 2:4] * variances[0] * priors[:, :, 2:],
                priors[:, :, :2] + pre[:, :, 4:6] * variances[0] * priors[:, :, 2:],
                priors[:, :, :2] + pre[:, :, 6:8] * variances[0] * priors[:, :, 2:],
                priors[:, :, :2] + pre[:, :, 8:10] * variances[0] * priors[:, :, 2:],
            ),
            dim=2,  # Concatenate along the landmark coordinate dimension
        )

    def extract_landmarks(
        self,
        video,
        frame_boxes,
        frame_valid,
        num_iterations=2,
    ):
        """
        Extract facial landmarks using two-stage refinement process.

        Args:
            video: Video tensor [batch_size, C, H, W] in range [0, 255]
            frame_boxes: Face bounding boxes [batch_size, 4] (x1, y1, x2, y2)
            frame_valid: Boolean mask [batch_size] indicating valid face detections
            batch_facebox: BatchFaceBox model for landmark detection
            device: PyTorch device
            enlarge_ratio: Factor to enlarge face crop region
            input_size: Target size for face crops (224x224)
            num_iterations: Number of refinement iterations (default: 2)

        Returns:
            transformed_landmarks: Final landmarks [batch_size, 106, 2] in original image coordinates
        """
        actual_batch_size = frame_boxes.shape[0]
        image_height, image_width = video.shape[-2:]

        # Initialize bounding box coordinates
        face_left = frame_boxes[:, 0]  # x1 coordinates
        face_top = frame_boxes[:, 1]  # y1 coordinates
        face_right = frame_boxes[:, 2]  # x2 coordinates
        face_bottom = frame_boxes[:, 3]  # y2 coordinates

        # Two-stage landmark detection (like original infer function)
        for iteration in range(num_iterations):
            # Calculate face dimensions and center points
            face_width = face_right - face_left + 1
            face_height = face_bottom - face_top + 1
            face_center_x = (face_right + face_left) / 2
            face_center_y = (face_bottom + face_top) / 2

            # Create enlarged square regions around faces
            enlarged_size = torch.max(face_height, face_width) * self.enlarge_ratio

            # Calculate new square crop coordinates (centered on face)
            crop_left = face_center_x - enlarged_size / 2
            crop_top = face_center_y - enlarged_size / 2
            crop_right = crop_left + enlarged_size
            crop_bottom = crop_top + enlarged_size

            # Store transformation offsets for later coordinate mapping
            transform_offset_x = crop_left.clone()
            transform_offset_y = crop_top.clone()

            # Calculate padding needed for boundary cases
            left_padding = torch.clamp(-crop_left, min=0)
            top_padding = torch.clamp(-crop_top, min=0)
            right_padding = torch.clamp(crop_right - image_width, min=0)
            bottom_padding = torch.clamp(crop_bottom - image_height, min=0)

            # Clamp crop coordinates to image boundaries
            crop_left = torch.clamp(crop_left, min=0)
            crop_top = torch.clamp(crop_top, min=0)
            crop_right = torch.clamp(crop_right, max=image_width)
            crop_bottom = torch.clamp(crop_bottom, max=image_height)

            # Convert to integers for cropping operations
            crop_coordinates = torch.stack(
                [crop_left, crop_top, crop_right, crop_bottom], dim=1
            ).int()
            padding_amounts = torch.stack(
                [left_padding, right_padding, top_padding, bottom_padding], dim=1
            ).int()  # F.pad format

            # Extract and process face crops from video frames
            num_frames = actual_batch_size
            processed_face_crops = []

            for frame_idx in range(num_frames):
                if not frame_valid[frame_idx]:
                    # Create a dummy crop for invalid face detections
                    processed_face_crops.append(
                        torch.zeros(
                            3, self.input_size, self.input_size, device=self.device
                        )
                    )
                    continue

                # Get crop coordinates and padding for this frame
                x1, y1, x2, y2 = crop_coordinates[frame_idx]
                left_pad, right_pad, top_pad, bottom_pad = padding_amounts[frame_idx]

                # Extract face region from the video frame
                face_region = video[frame_idx, :, y1:y2, x1:x2]  # [C, H, W]

                # Apply padding if crop extends beyond image boundaries
                if left_pad > 0 or top_pad > 0 or right_pad > 0 or bottom_pad > 0:
                    # F.pad format: (left, right, top, bottom)
                    face_region = F.pad(
                        face_region,
                        (left_pad, right_pad, top_pad, bottom_pad),
                        mode="constant",
                        value=0.0,
                    )

                # Resize to standard input size using bilinear interpolation
                face_region = F.interpolate(
                    face_region.unsqueeze(0).float(),
                    size=(self.input_size, self.input_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

                processed_face_crops.append(face_region)

            # Stack all face crops into a batch tensor for efficient processing
            batched_face_crops = torch.stack(
                processed_face_crops, dim=0
            )  # [num_frames, C, input_size, input_size]

            # Normalize correctly: subtract mean values then divide by 255
            # Mean values are already in [0,255] range, so subtract first, then normalize
            mean_values = torch.tensor([103.94, 116.78, 123.68], device=self.device).view(
                1, 3, 1, 1
            )
            batched_face_crops_norm = (batched_face_crops - mean_values) / 255.0

            # Get landmark predictions
            base_lmks = (
                self.landmark_model(batched_face_crops_norm) * self.input_size
            )

            # Transform landmarks back to original image coordinates
            # Calculate inverse scale for each face crop
            inv_scale = enlarged_size / self.input_size  # [batch_size]

            # Reshape landmarks from [batch_size, 212] to [batch_size, 106, 2]
            base_lmks_reshaped = base_lmks.view(-1, 106, 2)  # [batch_size, 106, 2]

            # Apply transformation: scale back and translate to original coordinates
            # inv_scale needs to be broadcast: [batch_size] -> [batch_size, 106, 2]
            inv_scale_expanded = inv_scale.view(-1, 1, 1).expand(-1, 106, 2)
            transform_offset = (
                torch.stack([transform_offset_x, transform_offset_y], dim=1)
                .view(-1, 1, 2)
                .expand(-1, 106, 2)
            )

            # Transform landmarks: scale * landmarks + offset
            transformed_landmarks = (
                base_lmks_reshaped * inv_scale_expanded + transform_offset
            )

            # For next iteration: use landmark-based bounding boxes
            if iteration < num_iterations - 1:
                face_left = torch.min(transformed_landmarks[:, :, 0], dim=1)[
                    0
                ]  # min x across all landmarks
                face_top = torch.min(transformed_landmarks[:, :, 1], dim=1)[
                    0
                ]  # min y across all landmarks
                face_right = torch.max(transformed_landmarks[:, :, 0], dim=1)[
                    0
                ]  # max x across all landmarks
                face_bottom = torch.max(transformed_landmarks[:, :, 1], dim=1)[
                    0
                ]  # max y across all landmarks

        return transformed_landmarks

    def batch_POS_gpu(self, landmarks_2d):
        """
        Batch GPU version of Procrustes analysis for 2D-3D landmark alignment.
        
        Args:
            landmarks_2d: torch.Tensor [batch_size, num_landmarks, 2] - 2D facial landmarks from images
            landmarks_3d: torch.Tensor [num_landmarks, 3] - 3D facial landmarks from standard model
            
        Returns:
            translation: torch.Tensor [batch_size, 2] - translation parameters (tx, ty)
            scale: torch.Tensor [batch_size] - scale factors
        """
        batch_size = landmarks_2d.shape[0]
        num_landmarks = landmarks_2d.shape[1]  # Should be 5
        device = landmarks_2d.device
        
        # Create coefficient matrix for least squares problem Ax = b
        # Each landmark contributes 2 equations (x and y), so 10 equations total
        # 8 unknowns: [r1_x, r1_y, r1_z, scale_tx, r2_x, r2_y, r2_z, scale_ty]
        coefficient_matrix = torch.zeros(batch_size, 2*num_landmarks, 8, device=device)  # [B, 10, 8]
        
        # Expand 3D landmarks to match batch size
        landmarks_3d_batch = self.landmarks_3d.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 5, 3]
        
        # Fill coefficient matrix:
        # Even rows (0,2,4,6,8): equations for x-coordinates
        coefficient_matrix[:, 0::2, 0:3] = landmarks_3d_batch  # 3D coords for rotation vector 1
        coefficient_matrix[:, 0::2, 3] = 1                     # constant term for translation
        
        # Odd rows (1,3,5,7,9): equations for y-coordinates  
        coefficient_matrix[:, 1::2, 4:7] = landmarks_3d_batch  # 3D coords for rotation vector 2
        coefficient_matrix[:, 1::2, 7] = 1                     # constant term for translation

        # Create target vector from 2D landmarks: reshape [B,5,2] -> [B,10,1]
        target_vector = landmarks_2d.reshape(batch_size, 2*num_landmarks, 1)  # [B, 10, 1]
        
        # Solve least squares problem: coefficient_matrix * solution = target_vector
        solution = torch.linalg.lstsq(coefficient_matrix, target_vector).solution  # [B, 8, 1]
        solution = solution.squeeze(-1)  # [B, 8]
        
        # Extract transformation parameters from solution
        rotation_vector_1 = solution[:, 0:3]   # First rotation vector [B, 3]
        rotation_vector_2 = solution[:, 4:7]   # Second rotation vector [B, 3] 
        scaled_translation_x = solution[:, 3]  # Scale * translation_x [B]
        scaled_translation_y = solution[:, 7]  # Scale * translation_y [B]
        
        # Calculate scale factor as average of rotation vector magnitudes
        scale = (torch.norm(rotation_vector_1, dim=1) + torch.norm(rotation_vector_2, dim=1)) / 2  # [B]
        
        # Combine translation components
        translation = torch.stack([scaled_translation_x, scaled_translation_y], dim=1)  # [B, 2]
        
        return translation, scale


    def batch_resize_n_crop_gpu(
        self,
        images,
        translation_params,
        scale_factors,
        frame_valid=None,
        output_size=224,
    ):
        """
        Batch GPU implementation of face alignment: resize and crop faces to standard size.
        
        Args:
            images: torch.Tensor [batch_size, C, H, W] - input images in range [0, 255]
            translation_params: torch.Tensor [batch_size, 2] - translation (tx, ty) from POS
            scale_factors: torch.Tensor [batch_size] - scale factors from POS (already computed as 102./scale)
            output_size: int - target crop size (default: 224x224)
            device: torch.device - computation device
            
        Returns:
            aligned_faces: torch.Tensor [batch_size, C, output_size, output_size]
            geometry_valid: torch.BoolTensor [batch_size]
        """
        batch_size, num_channels, image_height, image_width = images.shape
        finite = torch.isfinite(scale_factors) & torch.isfinite(
            translation_params
        ).all(dim=1)
        positive = scale_factors > 0
        safe_geometry = finite & positive
        safe_scales = torch.where(
            safe_geometry, scale_factors, torch.ones_like(scale_factors)
        )
        safe_translations = torch.where(
            safe_geometry[:, None],
            translation_params,
            translation_params.new_tensor([image_width / 2, image_height / 2]),
        )

        scaled_widths = (image_width * safe_scales).type(torch.int32)
        scaled_heights = (image_height * safe_scales).type(torch.int32)

        # Calculate crop coordinates
        # Center crop adjusted by translation parameters
        left_crop = (
            (scaled_widths - output_size) / 2
            + (safe_translations[:, 0] - image_width / 2) * safe_scales
        ).type(torch.int32)
        top_crop = (
            (scaled_heights - output_size) / 2
            + (image_height / 2 - safe_translations[:, 1]) * safe_scales
        ).type(torch.int32)
        right_crop = left_crop + output_size
        bottom_crop = top_crop + output_size

        geometry_valid = (
            safe_geometry
            & (scaled_widths > 0)
            & (scaled_heights > 0)
            & (scaled_widths <= MAX_ALIGNMENT_DIMENSION)
            & (scaled_heights <= MAX_ALIGNMENT_DIMENSION)
            & (left_crop < scaled_widths)
            & (right_crop > 0)
            & (top_crop < scaled_heights)
            & (bottom_crop > 0)
        )
        if frame_valid is not None:
            geometry_valid &= frame_valid.to(
                device=geometry_valid.device, dtype=torch.bool
            )

        # Calculate padding for all images in parallel
        pad_left = torch.clamp(-left_crop, min=0)
        pad_right = torch.clamp(right_crop - scaled_widths, min=0)
        pad_top = torch.clamp(-top_crop, min=0)
        pad_bottom = torch.clamp(bottom_crop - scaled_heights, min=0)
        
        # Clamp crop coordinates for all images in parallel
        cropped_left = torch.clamp(left_crop, min=0)
        cropped_right = torch.minimum(scaled_widths, right_crop)
        cropped_top = torch.clamp(top_crop, min=0)
        cropped_bottom = torch.minimum(scaled_heights, bottom_crop)

        # Convert images to float for processing
        images_float = images.float()

        # Process each image in the batch
        aligned_face_crops = []
        
        for batch_idx in range(batch_size):
            if not geometry_valid[batch_idx].item():
                aligned_face_crops.append(
                    images_float.new_zeros(num_channels, output_size, output_size)
                )
                continue

            # Resize the image 
            face_image = images_float[batch_idx]  # [C, H, W]
            face_image_resized = F.interpolate(
                face_image.unsqueeze(0),  # Add batch dimension: [1, C, H, W]
                size=(
                    scaled_heights[batch_idx].item(),
                    scaled_widths[batch_idx].item(),
                ),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)  # Remove batch dimension: [C, H, W]

            # Crop the face region using precomputed coordinates
            face_region = face_image_resized[:, 
                                        cropped_top[batch_idx]:cropped_bottom[batch_idx], 
                                        cropped_left[batch_idx]:cropped_right[batch_idx]]
            
            # Apply padding using precomputed padding values
            if pad_left[batch_idx] > 0 or pad_right[batch_idx] > 0 or pad_top[batch_idx] > 0 or pad_bottom[batch_idx] > 0:
                face_region = F.pad(face_region, 
                                (pad_left[batch_idx].item(), pad_right[batch_idx].item(), 
                                pad_top[batch_idx].item(), pad_bottom[batch_idx].item()), 
                                mode='constant', value=0.0)

            aligned_face_crops.append(face_region)

        
        # Stack all aligned faces into a batch tensor
        aligned_faces = torch.stack(aligned_face_crops, dim=0)  # [batch_size, C, output_size, output_size]
        
        return aligned_faces, geometry_valid

    def extract_aligned_faces(self, original_batch, transformed_batch, pads, original_height, original_width):
        """Extract aligned faces from batch of images"""

        locs, confs, lands = self.retinaface(transformed_batch.to(self.device))

        # Scores
        conf_batch = F.softmax(confs, dim=-1)
        scores_batch = conf_batch[:, :, 1]

        # Extract boxes and landmarks with retinaface
        frame_boxes, frame_scores, frame_landmarks, frame_valid = (
            self.process_retina_annotations(
                locs=locs,
                lands=lands,
                scores_batch=scores_batch,
                pads=pads,
                original_height=original_height,
                original_width=original_width,
            )
        )


        # import numpy as np







        # fig.suptitle("Initial face detection with RetinaFace", fontsize=16)




        final_landmarks = self.extract_landmarks(
            video=original_batch,
            frame_boxes=frame_boxes,
            frame_valid=frame_valid,
        )









        selected_landmarks_coords = final_landmarks[:, self.selected_landmarks, :]
        selected_landmarks_coords[:, :, -1] = original_height - 1 - selected_landmarks_coords[:, :, -1]
        translation_params, scale_factors_raw = self.batch_POS_gpu(selected_landmarks_coords)

        scale_factors = 102. / scale_factors_raw  # [B] - This is your final scale

        # Now align the faces
        aligned_faces, alignment_valid = self.batch_resize_n_crop_gpu(
            images=original_batch,  # Original video frames
            translation_params=translation_params,
            scale_factors=scale_factors,
            frame_valid=frame_valid,
        )

        return aligned_faces, final_landmarks, frame_valid & alignment_valid

    ALPHA_SLICES = {
        "id": (0, 80), "exp": (80, 144), "alb": (144, 224),
        "angle": (224, 227), "sh": (227, 254), "trans": (254, 257),
    }

    @torch.no_grad()
    def extract_coeffs(self, aligned_faces):
        """3DMM coefficients from aligned 224x224 faces (same net_recon input
        convention as CUPID's uv_face_model.forward: pixels / 255)."""
        alpha = self.net_recon(aligned_faces / 255.0)  # [B, 257]
        return {k: alpha[:, a:b] for k, (a, b) in self.ALPHA_SLICES.items()}

    @torch.no_grad()
    def process_frames(self, frames):
        """Full pipeline for a [B, 3, H, W] uint8 RGB frame batch.

        Returns dict of CPU tensors: exp [B,64], id [B,80], angle [B,3],
        trans [B,3], landmarks [B,106,2] (original image coords), valid [B].
        """
        frames = frames.to(self.device)
        h, w = frames.shape[-2:]
        transformed, pads = batch_resize_normalize_pad(frames, self.max_size)
        aligned, landmarks, valid = self.extract_aligned_faces(
            frames, transformed, pads, h, w
        )
        coeffs = self.extract_coeffs(aligned)
        return {
            "exp": coeffs["exp"].cpu(),
            "id": coeffs["id"].cpu(),
            "angle": coeffs["angle"].cpu(),
            "trans": coeffs["trans"].cpu(),
            "landmarks": landmarks.cpu(),
            "valid": valid.cpu(),
        }

