import cv2
import math
import numpy as np

from matplotlib import pyplot as plt
from torchreid.data.masks_transforms import CocoToEightBodyMasks
from torchreid.utils.visualization.visualize_query_gallery_rankings import draw_keypoints, colored_body_parts_overlay


def display_kpr_reid_samples_grid(samples, display_mode, save_path=None):
    kp_grouping_eight_bp = CocoToEightBodyMasks()

    images_rgb = []
    images_with_parts_masks = []
    has_parts_masks = "parts_masks" in samples[0] and samples[0]["parts_masks"] is not None

    for sample in samples:
        img = sample["image"]
        keypoints_xyc = sample.get("keypoints_xyc")
        negative_kps = sample.get("negative_kps")

        img_with_keypoints = img.copy()  # Start with the original image

        # Add keypoints if available
        if keypoints_xyc is not None:
            # Add a new dimension 'k' indicating keypoint group
            keypoints_xyck = kp_grouping_eight_bp.apply_to_keypoints_xyc(keypoints_xyc)
            img_with_keypoints = draw_keypoints(img_with_keypoints, keypoints_xyck)

        # Draw negative keypoints if available
        if negative_kps is not None:
            for i in range(negative_kps.shape[0]):
                negative_kps_grouped = kp_grouping_eight_bp.apply_to_keypoints_xyc(negative_kps[i])
                img_with_keypoints = draw_keypoints(img_with_keypoints, negative_kps_grouped, color=(0, 0, 255))

        # Convert the image from BGR to RGB
        img_rgb = cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB)
        images_rgb.append(img_rgb)

        # If parts_masks exist, overlay them on a copy of the image
        if has_parts_masks:
            img_with_parts_masks = colored_body_parts_overlay(img.copy(), sample["parts_masks"])
            img_with_parts_masks_rgb = cv2.cvtColor(img_with_parts_masks, cv2.COLOR_BGR2RGB)
            images_with_parts_masks.append(img_with_parts_masks_rgb)

    # Determine the number of images
    num_samples = len(samples)
    total_images = num_samples * 2 if has_parts_masks else num_samples

    # Compute grid size (rows and cols)
    cols = math.ceil(math.sqrt(total_images))
    rows = math.ceil(total_images / cols)

    # Set the figure size
    plt.figure(figsize=(cols * 2, rows * 2.5))

    # Display each image in the grid
    for i in range(num_samples):
        position = i * 2 + 1 if has_parts_masks else i + 1
        plt.subplot(rows, cols, position)
        plt.imshow(images_rgb[i])
        plt.axis('off')

        if has_parts_masks:
            plt.subplot(rows, cols, i * 2 + 2)
            plt.imshow(images_with_parts_masks[i])
            plt.axis('off')

    plt.tight_layout()
    if display_mode == "plot":
        plt.show()
    elif display_mode == "save":
        if save_path is not None:
            print(f"Saving samples grid to {save_path}")
            plt.savefig(str(save_path))
            plt.close()
    else:
        raise ValueError(f"Invalid display mode: {display_mode} (choose 'plot' or 'save')")


def display_distance_matrix(distance_matrix, samples_grp_1, samples_grp_2, display_mode, save_path=None):
    kp_grouping_eight_bp = CocoToEightBodyMasks()

    num_group1 = len(samples_grp_1)
    num_group2 = len(samples_grp_2)

    # Normalize the distance matrix to a range of [0, 1]
    min_val = np.min(distance_matrix)
    max_val = np.max(distance_matrix)
    normalized_dist_matrix = (distance_matrix - min_val) / (max_val - min_val)

    # Create a figure with (N+1)x(M+1) subplots and adjust the overall size
    fig, axes = plt.subplots(num_group1 + 1, num_group2 + 1, figsize=(num_group2 * 4, num_group1 * 4))

    # Set the colormap for the distance matrix
    cmap = plt.get_cmap('jet')

    # Helper function to create keypoint and parts mask overlays
    def create_overlays(sample):
        img = sample["image"]
        keypoints_xyc = sample.get("keypoints_xyc")
        negative_kps = sample.get("negative_kps")

        img_with_keypoints = img.copy()  # Start with the original image

        # Add keypoints if available
        if keypoints_xyc is not None:
            # Add a new dimension 'k' indicating keypoint group
            keypoints_xyck = kp_grouping_eight_bp.apply_to_keypoints_xyc(keypoints_xyc)
            img_with_keypoints = draw_keypoints(img_with_keypoints, keypoints_xyck)

        # Draw negative keypoints if available
        if negative_kps is not None:
            for i in range(negative_kps.shape[0]):
                negative_kps_grouped = kp_grouping_eight_bp.apply_to_keypoints_xyc(negative_kps[i])
                img_with_keypoints = draw_keypoints(img_with_keypoints, negative_kps_grouped, color=(0, 0, 255))

        img_rgb = cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB)

        # Overlay parts masks if they exist
        if "parts_masks" in sample and sample["parts_masks"] is not None:
            img_with_parts_masks = colored_body_parts_overlay(img.copy(), sample["parts_masks"])
            img_with_parts_masks_rgb = cv2.cvtColor(img_with_parts_masks, cv2.COLOR_BGR2RGB)
        else:
            img_with_parts_masks_rgb = img_rgb

        return img_rgb, img_with_parts_masks_rgb

    # Display images from group 2 at the top row (first row, second column onwards)
    for j in range(num_group2):
        img_rgb, img_with_parts_masks_rgb = create_overlays(samples_grp_2[j])
        axes[0, j + 1].imshow(np.hstack([img_rgb, img_with_parts_masks_rgb]))
        axes[0, j + 1].axis('off')

    # Display images from group 1 in the first column (second row onwards, first column)
    for i in range(num_group1):
        img_rgb, img_with_parts_masks_rgb = create_overlays(samples_grp_1[i])
        axes[i + 1, 0].imshow(np.hstack([img_rgb, img_with_parts_masks_rgb]))
        axes[i + 1, 0].axis('off')

    # Display the normalized distance matrix with color coding
    for i in range(num_group1):
        for j in range(num_group2):
            norm_dist_value = normalized_dist_matrix[i, j]
            color = cmap(norm_dist_value)  # Get color based on the normalized distance value

            # Fill the cell with the color
            axes[i + 1, j + 1].imshow([[color]])
            # Center the text and make sure it's visible on any background color
            axes[i + 1, j + 1].text(0., 0., f'{distance_matrix[i, j]:.3f}', ha='center', va='center',
                                    fontsize=30, color='white', weight='bold')
            axes[i + 1, j + 1].axis('off')

    # Hide the top-left corner (axes[0, 0])
    axes[0, 0].axis('off')

    # Adjust layout to make everything fit, then add color bar
    plt.subplots_adjust(left=0.05, right=0.78, top=0.95, bottom=0.05)
    cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])  # Position of colorbar
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=min_val, vmax=max_val), cmap=cmap),
                        cax=cbar_ax)
    cbar.set_label('Distance (0 = similar, 1 = dissimilar)', fontsize=26)
    cbar.ax.tick_params(labelsize=22)  # Adjust tick size

    if display_mode == "plot":
        plt.show()
    elif display_mode == "save":
        print(f"Saving samples grid to {save_path}")
        plt.savefig(str(save_path))
        plt.close()
    else:
        raise ValueError(f"Invalid display mode: {display_mode} (choose 'plot' or 'save')")
