import ntpath
import random

import cv2
import math
import matplotlib
import numpy as np
import matplotlib.cm as cm
import colorsys

from torchreid.data.datasets.keypoints_to_masks import rescale_keypoints
from torchreid.utils import Logger, perc
from torchreid.utils.engine_state import EngineState
from scipy import stats

GRID_SPACING_V = 50
GRID_SPACING_H = 30
QUERY_EXTRA_SPACING = 10
TOP_MARGIN = 150
LEFT_MARGIN = 100
RIGHT_MARGIN = 450
BOTTOM_MARGIN = 200
ROW_BACKGROUND_LEFT_MARGIN = 75
ROW_BACKGROUND_RIGHT_MARGIN = 75
LEFT_TEXT_OFFSET = 10
BW = 12  # border width
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (255, 255, 0)
PURPLE = (191, 64, 191)
BLACK = (0, 0, 0)
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (0, 0, 0)
TEXT_LINE_TYPE = cv2.LINE_AA
WIDTH = 128
HEIGHT = 256
SMALL_FONSCALE = 0.7
SMALL_THICK = 1
MEDIUM_FONSCALE = 1.5
MEDIUM_THICK = 2
cmap = matplotlib.cm.get_cmap('hsv')
BP_COLORS = [(153, 153, 255), (204, 153, 255), (255, 255, 153), (153, 255, 204), (153, 255, 153), (153, 255, 204), (153, 255, 255), (153, 204, 255), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]


# TODO document and make code easier to read and adapt, i.e. less intricate
def visualize_ranking_grid(distmat, body_parts_distmat, test_loader, dataset_name, qf_parts_visibility, gf_parts_visibility, q_parts_masks, g_parts_masks, q_pids, g_pids, q_camids, g_camids, q_anns, g_anns, eval_metrics, save_dir, topk, visrank_q_idx_list, visrank_count, config=None, bp_idx=None):
    num_q, num_g = distmat.shape
    query_dataset = test_loader['query'].dataset
    gallery_dataset = test_loader['gallery'].dataset
    assert num_q == len(query_dataset)
    assert num_g == len(gallery_dataset)
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    vis_thresh = config.model.kpr.keypoints.vis_thresh
    display_mode = config.test.visrank_display_mode
    all_AP = eval_metrics['all_AP']
    mAP_index_sort = np.argsort(all_AP)

    # Prepare images for query visualization
    mask_filtering_flag = qf_parts_visibility is not None or gf_parts_visibility is not None
    if qf_parts_visibility is None:
        qf_parts_visibility = np.ones((num_q, body_parts_distmat.shape[0]), dtype=bool)

    if gf_parts_visibility is None:
        gf_parts_visibility = np.ones((num_g, body_parts_distmat.shape[0]), dtype=bool)

    if display_mode == 'display_worst':
        q_idx_list = mAP_index_sort[:visrank_count]
    elif display_mode == 'display_worst_rand':
        if visrank_count >= len(mAP_index_sort):
            q_idx_list = mAP_index_sort
        else:
            frozen_lognorm = stats.lognorm(s=1., scale=math.exp(0.))
            proba_per_sorted_index = frozen_lognorm.pdf(np.linspace(0, 4, len(mAP_index_sort)))
            q_idx_list = np.random.choice(mAP_index_sort, replace=False, size=visrank_count, p=proba_per_sorted_index/proba_per_sorted_index.sum())
    else:
        n_missing = visrank_count - len(visrank_q_idx_list)
        if n_missing > 0:
            q_idx_list = visrank_q_idx_list
            remaining_idx = np.arange(0, num_q)
            q_idx_list = np.append(q_idx_list, np.random.choice(remaining_idx, replace=False, size=n_missing))
        elif n_missing < 0:
            q_idx_list = np.array(visrank_q_idx_list[:visrank_count])
        else:
            q_idx_list = np.array(visrank_q_idx_list)

    q_idx_list = q_idx_list.astype(int)
    print("visualize_ranking_grid for dataset {}, bp {} and ids {}".format(dataset_name, bp_idx, q_idx_list))
    for q_idx in q_idx_list:
        if q_idx >= len(query_dataset):
            # FIXME this happen when using multiple target dataset with 'visrank_q_idx_list' provided for another dataset
            new_q_idx = random.randint(0, len(query_dataset)-1)
            print("Invalid query index {}, using random index {} instead".format(q_idx, new_q_idx))
            q_idx = new_q_idx
        query = query_dataset[q_idx]
        qpid, qcamid, qimg_path, q_target_m, q_prompt_m = query['pid'], query['camid'], query['img_path'], query.get('target_masks', None), query.get('prompt_masks', None)
        display_prompt = q_prompt_m is not None
        display_target = q_target_m is not None
        qmasks = q_parts_masks[q_idx]
        if "keypoints_xyc" in q_anns:
            qkp = q_anns["keypoints_xyc"][q_idx]
        else:
            qkp = None
        if "negative_keypoints_xyc" in q_anns:
            qnegkp = q_anns["negative_keypoints_xyc"][q_idx]
        else:
            qnegkp = None
        if bp_idx is not None:
            qmasks = qmasks[bp_idx:bp_idx+1]
        query_sample = (q_idx, qpid, qcamid, qimg_path, q_target_m, q_prompt_m, qmasks, qkp, qnegkp, qf_parts_visibility[q_idx, :])
        gallery_topk_samples = []
        rank_idx = 1

        order = indices[q_idx]
        remove = test_loader['query'].dataset.gallery_filter(qpid, qcamid, None, g_pids[order], g_camids[order], None)
        keep = np.invert(remove) & (distmat[q_idx, order] >= 0)
        valid_g_indices = indices[q_idx][keep]
        q_matches = matches[q_idx][keep]
        last_match = np.where(q_matches == 1)[0][-1]
        ranking_info = {
            "total_positive": np.sum(q_matches),
            "total_candidates": len(q_matches),
            "last_match_idx": last_match,
        }
        to_display_q = np.append(valid_g_indices[:topk], valid_g_indices[last_match])

        assert gallery_dataset[valid_g_indices[last_match]]['pid'] == qpid
        for g_idx in to_display_q:
            gallery = gallery_dataset[g_idx]
            gpid, gcamid, gimg_path, g_target_m, g_prompt_m = gallery['pid'], gallery['camid'], gallery['img_path'], gallery.get('target_masks', None), gallery.get('prompt_masks', None)
            gmasks = g_parts_masks[g_idx]
            if "keypoints_xyc" in g_anns:
                gkp = g_anns["keypoints_xyc"][g_idx]
            else:
                gkp = None
            if "negative_keypoints_xyc" in q_anns:
                gnegkp = g_anns["negative_keypoints_xyc"][g_idx]
            else:
                gnegkp = None
            if bp_idx is not None:
                gmasks = gmasks[bp_idx:bp_idx+1]
            gallery_sample = (g_idx, gpid, gcamid, gimg_path, g_target_m, g_prompt_m, gmasks, gkp, gnegkp, gf_parts_visibility[g_idx, :], qpid == gpid,
                              distmat[q_idx, g_idx],
                              body_parts_distmat[:, q_idx, g_idx])
            gallery_topk_samples.append(gallery_sample)
            rank_idx += 1
        if len(gallery_topk_samples) > 0:
            show_ranking_grid(query_sample, gallery_topk_samples, eval_metrics, dataset_name, config, mask_filtering_flag, bp_idx, display_prompt, display_target, ranking_info, vis_thresh)
        else:
            print("Skip ranking plot of query id {} ({}), no valid gallery available".format(q_idx, qimg_path))


def show_ranking_grid(query_sample, gallery_topk_samples, eval_metrics, dataset_name, config, mask_filtering_flag, bp_idx, display_prompt, display_target, ranking_info, vis_thresh):
    qidx, qpid, qcamid, qimg_path, q_target_m, q_prompt_m, qmasks, qkp, qnegkp, qf_parts_visibility = query_sample
    mAP = eval_metrics['mAP']
    AP = eval_metrics['all_AP'][qidx]
    rank1 = eval_metrics['cmc'][0]
    use_negative_keypoints = config.model.kpr.keypoints.use_negative_keypoints

    add_cols = 0
    if display_prompt:
        add_cols += 1
        if use_negative_keypoints:
            add_cols += 1
    if display_target:
        add_cols += 1

    topk = len(gallery_topk_samples)
    bp_num = len(qf_parts_visibility)

    num_cols = bp_num + 1 + add_cols
    num_rows = topk + 1
    grid_img = 255 * np.ones(
        (
            num_rows * HEIGHT + (num_rows + 1) * GRID_SPACING_V + QUERY_EXTRA_SPACING + TOP_MARGIN + BOTTOM_MARGIN,
            num_cols * WIDTH + (num_cols + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN + RIGHT_MARGIN,
            3
        ),
        dtype=np.uint8
    )

    samples = [query_sample] + gallery_topk_samples

    insert_background_line(grid_img, BLUE, 0, HEIGHT, 120, 0)
    insert_background_line(grid_img, BLUE, len(samples), HEIGHT, 0, -75)

    pos = (int(grid_img.shape[1]/2), 0)
    filtering_str = "body part filtering with threshold {}".format(config.model.kpr.masks.mask_filtering_threshold) if config.model.kpr.mask_filtering_testing else "no body part filtering"
    align_top_text(grid_img, "Ranking for dataset {}, job {}, pid {}, AP {:.2f}%, mAP {:.2f}%, rank1 {:.2f}%, loss {}, {}".format(dataset_name, config.project.job_id, qpid, AP * 100, mAP * 100, rank1 * 100, config.loss.part_based.name, filtering_str), pos, SMALL_FONSCALE, SMALL_THICK, 15)

    for row, sample in enumerate(samples):
        display_sample_on_row(grid_img, sample, row, (WIDTH, HEIGHT), mask_filtering_flag, qf_parts_visibility, [config.data.width, config.data.height], display_prompt, display_target, use_negative_keypoints, ranking_info, len(samples), vis_thresh)

    for col in range(1+add_cols, num_cols):
        parts_visibility_count = 0
        row = topk+1
        bp_idx = col - 1 - add_cols
        distances = []
        for i, sample in enumerate(samples):
            if i == 0:
                idx, pid, camid, img_path, target_m, prompt_m, masks, kp, negkp, parts_visibility = sample
            else:
                idx, pid, camid, img_path, target_m, prompt_m, masks, kp, negkp, parts_visibility, matched, dist_to_query, body_parts_dist_to_query = sample
                distances.append(body_parts_dist_to_query[bp_idx])
            parts_visibility_count += parts_visibility[bp_idx]
        distances = np.asarray(distances)
        min = distances.min()
        max = distances.max()
        mean = distances.mean()
        pos = (col * WIDTH + int(WIDTH / 2) + (col + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN,
               (row) * HEIGHT + int(HEIGHT / 2) + (row + 1) * GRID_SPACING_V + QUERY_EXTRA_SPACING + TOP_MARGIN)

        align_top_multi_text(grid_img, "Bp={:.1f}/{}\nMin={:.1f}\nMean={:.1f}\nMax={:.1f}".format(
                parts_visibility_count, topk + 1, min, mean, max), pos, SMALL_FONSCALE, SMALL_THICK, 10)

    if bp_idx is not None:
        filename = "_{}_{}_qidx_{}_qpid_{}_{}_part_{}.jpg".format(config.project.job_id, dataset_name, qidx, qpid, ntpath.basename(qimg_path), bp_idx)
    else:
        filename = "_{}_{}_qidx_{}_qpid_{}_{}.jpg".format(config.project.job_id, dataset_name, qidx, qpid, ntpath.basename(qimg_path))
    # path = os.path.join(save_dir, filename)
    # Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    # cv2.imwrite(path, grid_img)
    Logger.current_logger().add_image("Ranking grid", filename, cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB), EngineState.current_engine_state().epoch)


def insert_background_line(grid_img, match_color, row, height, padding_top=0, padding_bottom=0):
    alpha = 0.1
    color = (255 * (1-alpha) + match_color[0] * alpha,
             255 * (1-alpha) + match_color[1] * alpha,
             255 * (1-alpha) + match_color[2] * alpha)
    hs = row * height + (row + 1) * GRID_SPACING_V + QUERY_EXTRA_SPACING + TOP_MARGIN - int(GRID_SPACING_V/2) + 15 - padding_top
    he = (row + 1) * height + (row + 1) * GRID_SPACING_V + QUERY_EXTRA_SPACING + TOP_MARGIN + int(GRID_SPACING_V/2) + 15 + padding_bottom
    ws = ROW_BACKGROUND_LEFT_MARGIN
    we = grid_img.shape[1] - ROW_BACKGROUND_RIGHT_MARGIN
    grid_img[hs:he, ws:we, :] = color


def display_sample_on_row(grid_img, sample, row, img_shape, mask_filtering_flag, q_parts_visibility, model_img_size, display_prompt, display_target, use_negative_keypoints, ranking_info, total_rows, vis_thresh):
    if row == 0:
        idx, pid, camid, img_path, target_m, prompt_m, masks, kp, negkp, parts_visibility = sample
        matched, dist_to_query, body_parts_dist_to_query = None, None, None
    else:
        idx, pid, camid, img_path, target_m, prompt_m, masks, kp, negkp, parts_visibility, matched, dist_to_query, body_parts_dist_to_query = sample

    masks = masks.numpy()
    width, height = img_shape
    bp_num = masks.shape[0]
    img = cv2.imread(img_path)
    img = cv2.resize(img, (width, height))

    add_cols = 0
    if display_prompt:
        add_cols += 1
        if use_negative_keypoints:
            add_cols += 1
    if display_target:
        add_cols += 1

    num_cols = bp_num + 1 + add_cols + 1
    for col in range(0, num_cols):
        bp_idx = col - 1 - add_cols - 1
        if row == 0 and col == 0:
            img_to_insert = img.copy()
            if kp is not None:
                img_to_insert = draw_keypoints(img_to_insert, kp, model_img_size, vis_thresh=vis_thresh)
            if negkp is not None:
                for i in range(negkp.shape[0]):
                    img_to_insert = draw_keypoints(img_to_insert, negkp[i], model_img_size, vis_thresh=vis_thresh, color=RED)

            img_to_insert = make_border(img_to_insert, BLUE, BW)
            pos = ((num_cols) * width + (num_cols + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN,
                   row * height + int(height / 2) + (row + 1) * GRID_SPACING_V + TOP_MARGIN)
            align_left_multitext(grid_img, "*Id = {}*\n"
                                      "Visible = {}/{}\n"
                                      "# positives = {}\n"
                                      "# gallery candidates = {}\n".format(
                pid, parts_visibility.sum(), bp_num, ranking_info["total_positive"], ranking_info["total_candidates"]), pos, SMALL_FONSCALE, SMALL_THICK, 10)
        elif col == 0:
            match_color = GREEN if matched else RED
            insert_background_line(grid_img, match_color, row, height)
            img_to_insert = img.copy()
            if kp is not None:
                img_to_insert = draw_keypoints(img_to_insert, kp, model_img_size, vis_thresh=vis_thresh)
            if negkp is not None:
                for i in range(negkp.shape[0]):
                    img_to_insert = draw_keypoints(img_to_insert, negkp[i], model_img_size, vis_thresh=vis_thresh, color=RED)

            img_to_insert = make_border(img_to_insert, match_color, BW)
            pos = (LEFT_MARGIN + GRID_SPACING_H,
                   row * height + int(height / 2) + (row + 1) * GRID_SPACING_V + QUERY_EXTRA_SPACING + TOP_MARGIN)
            ranking_position = row if row<total_rows-1 else ranking_info["last_match_idx"]+1
            align_right_text(grid_img, str(ranking_position), pos, MEDIUM_FONSCALE, MEDIUM_THICK, 30)
            pos = (LEFT_MARGIN + GRID_SPACING_H + int(width / 2),
                   (row + 1) * height + (row + 1) * GRID_SPACING_V + QUERY_EXTRA_SPACING + TOP_MARGIN)
            g_to_q_vis_score = np.sqrt(q_parts_visibility * parts_visibility).sum() / bp_num
            align_top_text(grid_img, "{}% | {:.2f}".format(int(perc(g_to_q_vis_score, 0)), dist_to_query), pos, SMALL_FONSCALE, SMALL_THICK, 10)

            pos = ((num_cols) * width + (num_cols + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN,
                   row * height + int(height / 2) + (row + 1) * GRID_SPACING_V + QUERY_EXTRA_SPACING + TOP_MARGIN)
            if len(parts_visibility) == 1 or parts_visibility.sum() == 0:
                valid_body_parts_dist = body_parts_dist_to_query
            else:
                valid_body_parts_dist = body_parts_dist_to_query[parts_visibility > 0]

            align_left_multitext(grid_img, "*Id = {}*\n"
                                      "Idx = {}\n"
                                      "Cam id = {}\n"
                                      "Name = {}\n"
                                      "Bp Visibles = {}/{}\n"
                                      "[{:.2f}; {:.2f}; {:.2f}]\n"
                                      "[{:.2f}; {:.2f}; {:.2f}]".format(
                pid, idx, camid, ntpath.basename(img_path), (parts_visibility > 0).sum(), bp_num,
                body_parts_dist_to_query.min(), body_parts_dist_to_query.mean(), body_parts_dist_to_query.max(),
                valid_body_parts_dist.min(), valid_body_parts_dist.mean(), valid_body_parts_dist.max()), pos, 0.6, 1, 10, match_color)
        elif col == 1 and display_prompt:
            if row == 0:
                pos = (col * width + int(width / 2) + (col + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN,
                       (row + 1) * height + (row + 1) * GRID_SPACING_V + TOP_MARGIN)
            else:
                pos = (col * width + int(width / 2) + (col + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN,
                       (row + 1) * height + (row + 1) * GRID_SPACING_V + QUERY_EXTRA_SPACING + TOP_MARGIN)
            align_top_text(grid_img, "pos. prompt", pos, SMALL_FONSCALE, SMALL_THICK, 10)
            offset = 1
            if use_negative_keypoints:
                offset += 1
            mask = prompt_m[offset:].max(dim=0)[0].numpy()
            img_with_mask_overlay = mask_overlay(img, mask)
            img_to_insert = make_border(img_with_mask_overlay, GREEN, BW)
        elif col == 2 and display_prompt and use_negative_keypoints:
            if row == 0:
                pos = (col * width + int(width / 2) + (col + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN,
                       (row + 1) * height + (row + 1) * GRID_SPACING_V + TOP_MARGIN)
            else:
                pos = (col * width + int(width / 2) + (col + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN,
                       (row + 1) * height + (row + 1) * GRID_SPACING_V + QUERY_EXTRA_SPACING + TOP_MARGIN)
            align_top_text(grid_img, "neg. prompt", pos, SMALL_FONSCALE, SMALL_THICK, 10)
            bckg_mask = prompt_m[1].numpy()
            img_with_mask_overlay = mask_overlay(img, bckg_mask)
            img_to_insert = make_border(img_with_mask_overlay, GREEN, BW)
        elif col == add_cols and display_target:
            if row == 0:
                pos = (col * width + int(width / 2) + (col + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN,
                       (row + 1) * height + (row + 1) * GRID_SPACING_V + TOP_MARGIN)
            else:
                pos = (col * width + int(width / 2) + (col + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN,
                       (row + 1) * height + (row + 1) * GRID_SPACING_V + QUERY_EXTRA_SPACING + TOP_MARGIN)
            align_top_text(grid_img, "hum. pars. l.", pos, SMALL_FONSCALE, SMALL_THICK, 10)
            img_with_mask_overlay = colored_body_parts_overlay(img, target_m[1:].numpy())
            # mask = target_m[1:].max(dim=0)[0].numpy()
            # img_with_mask_overlay = mask_overlay(img, mask)
            img_to_insert = make_border(img_with_mask_overlay, GREEN, BW)
        elif col == add_cols+1:
            if row == 0:
                pos = (col * width + int(width / 2) + (col + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN,
                       (row + 1) * height + (row + 1) * GRID_SPACING_V + TOP_MARGIN)
            else:
                pos = (col * width + int(width / 2) + (col + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN,
                       (row + 1) * height + (row + 1) * GRID_SPACING_V + QUERY_EXTRA_SPACING + TOP_MARGIN)
            align_top_text(grid_img, "parts att.", pos, SMALL_FONSCALE, SMALL_THICK, 10)
            img_with_mask_overlay = colored_body_parts_overlay(img, masks[1:])
            img_to_insert = make_border(img_with_mask_overlay, GREEN, BW)
        # elif col == add_cols+2:
        #     if row == 0:
        #         pos = (col * width + int(width / 2) + (col + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN,
        #                (row + 1) * height + (row + 1) * GRID_SPACING_V + TOP_MARGIN)
        #     else:
        #         pos = (col * width + int(width / 2) + (col + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN,
        #                (row + 1) * height + (row + 1) * GRID_SPACING_V + QUERY_EXTRA_SPACING + TOP_MARGIN)
        #     align_top_text(grid_img, "parts", pos, SMALL_FONSCALE, SMALL_THICK, 10)
        #     # display colored body parts with white background
        #     img_with_mask_overlay = colored_body_parts_overlay(img, masks[1:], alpha=0.)
        #     img_to_insert = make_border(img_with_mask_overlay, GREEN, BW)
        else:
            if row == 0:
                pos = (col * width + int(width / 2) + (col + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN,
                       TOP_MARGIN + GRID_SPACING_V)
                align_bottom_text(grid_img, str(bp_idx), pos, MEDIUM_FONSCALE, MEDIUM_THICK, 35)
                pos = (col * width + int(width / 2) + (col + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN,
                       (row + 1) * height + (row + 1) * GRID_SPACING_V + TOP_MARGIN)
                align_top_text(grid_img, "{}%".format(int(perc(parts_visibility[bp_idx], 0))), pos, SMALL_FONSCALE, SMALL_THICK, 10)
            if row != 0:
                pos = (col * width + int(width / 2) + (col + 1) * GRID_SPACING_H + QUERY_EXTRA_SPACING + LEFT_MARGIN,
                       (row + 1) * height + (row + 1) * GRID_SPACING_V + QUERY_EXTRA_SPACING + TOP_MARGIN)
                thickness = 2 if body_parts_dist_to_query.argmax() == bp_idx or body_parts_dist_to_query.argmin() == bp_idx else 1
                align_top_text(grid_img, "{}% | {:.2f}".format(int(perc(parts_visibility[bp_idx], 0)), body_parts_dist_to_query[bp_idx]), pos, SMALL_FONSCALE, thickness, 10)
            mask = masks[bp_idx, :, :]
            img_with_mask_overlay = mask_overlay(img, mask)
            if mask_filtering_flag:
                # match_color = GREEN if parts_visibility[bp_idx] else RED
                match_color = cmap(parts_visibility[bp_idx].item()/3, bytes=True)[0:-1]  # divided by three because hsv colormap goes from red to green inside [0, 0.333]
                img_to_insert = make_border(img_with_mask_overlay, (int(match_color[2]), int(match_color[1]), int(match_color[0])), BW)
            else:
                img_to_insert = img_with_mask_overlay

        insert_img_into_grid(grid_img, img_to_insert, row, col)


def mask_overlay(img, mask, clip=True, interpolation=cv2.INTER_CUBIC):
    width, height = img.shape[1], img.shape[0]
    mask = cv2.resize(mask, dsize=(width, height), interpolation=interpolation)
    if clip:
        mask = np.clip(mask, 0, 1)
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = np.interp(mask, (mask.min(), mask.max()), (0, 255)).astype(np.uint8)
    mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    masked_img = cv2.addWeighted(img, 0.5, mask_color.astype(img.dtype), 0.5, 0)
    return masked_img


def scale_lightness(rgb, scale_l=1.4):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)


def colored_body_parts_overlay(img, masks, clip=True, interpolation=cv2.INTER_CUBIC, alpha=0.28, mask_threshold=0, weight_scale=1):
    width, height = img.shape[1], img.shape[0]
    white_bckg = np.ones_like(img) * 255
    for i in range(masks.shape[0]):
        mask = cv2.resize(masks[i], dsize=(width, height), interpolation=interpolation)
        if clip:
            mask = np.clip(mask, 0, 1)
        else:
            mask = np.interp(mask, (mask.min(), mask.max()), (0, 255)).astype(np.uint8)
        weight = mask
        mask_alpha = np.ones_like(weight)
        mask_alpha[mask < mask_threshold] = 0
        mask_alpha = np.expand_dims(mask_alpha, 2)
        weight = np.expand_dims(weight, 2) / weight_scale
        color_img = np.zeros_like(img)
        # print(f"(len(masks)-1) {(len(masks)-1)} - i {i}")
        color = scale_lightness(cm.gist_rainbow(i / (len(masks)-1))[0:-1])
        # print(f"color {color}")
        # print(f"color_flip {np.flip(np.array(color)*255).astype(np.uint8)}")
        color_img[:] = np.flip(np.array(color)*255).astype(np.uint8)
        white_bckg = white_bckg * (1 - mask_alpha * weight) + color_img * mask_alpha * weight
    masked_img = cv2.addWeighted(img, alpha, white_bckg.astype(img.dtype), 1-alpha, 0)
    return masked_img


def align_top_text(img, text, pos, fontScale=1.0, thickness=1, padding=4):
    textsize = cv2.getTextSize(text, TEXT_FONT, fontScale, thickness)[0]
    textX = int(pos[0] - (textsize[0] / 2))
    textY = pos[1] + textsize[1] + padding
    cv2.putText(img, text, (textX, textY), TEXT_FONT, fontScale=fontScale, color=TEXT_COLOR, thickness=thickness,
                lineType=TEXT_LINE_TYPE)


def align_top_multi_text(img, text, pos, fontScale=1.0, thickness=1, padding=4, text_color=(0, 0, 0)):
    v_padding = 20
    text_lines = text.split('\n')
    text_line_height = cv2.getTextSize(text_lines[0], TEXT_FONT, fontScale, thickness)[0][1]
    text_height = len(text_lines) * text_line_height + (len(text_lines)-1) * v_padding
    textY = int(pos[1] - text_height + text_line_height) + padding

    for i, text_line in enumerate(text_lines):
        bold_marker = "*"
        bold = text_line.startswith(bold_marker) and text_line.endswith(bold_marker)
        line_thickness = thickness+1 if bold else thickness
        if bold:
            text_line = text_line[len(bold_marker):len(text_line)-len(bold_marker)]
        textsize = cv2.getTextSize(text_line, TEXT_FONT, fontScale, thickness)[0]
        text_line_pos = (int(pos[0] - (textsize[0] / 2)), textY + (text_line_height + v_padding) * i)
        text_color = text_color if i == 0 else TEXT_COLOR
        cv2.putText(img, text_line, text_line_pos, TEXT_FONT, fontScale=fontScale, color=TEXT_COLOR, thickness=line_thickness,
                    lineType=TEXT_LINE_TYPE)


def align_bottom_text(img, text, pos, fontScale=1.0, thickness=1, padding=4):
    textsize = cv2.getTextSize(text, TEXT_FONT, fontScale, thickness)[0]
    textX = int(pos[0] - (textsize[0] / 2))
    textY = pos[1] - padding
    cv2.putText(img, text, (textX, textY), TEXT_FONT, fontScale=fontScale, color=TEXT_COLOR, thickness=thickness,
                lineType=TEXT_LINE_TYPE)


def align_right_text(img, text, pos, fontScale=1.0, thickness=1, padding=4):
    textsize = cv2.getTextSize(text, TEXT_FONT, fontScale, thickness)[0]
    textX = pos[0] - textsize[0] - padding
    textY = int(pos[1] + (textsize[1] / 2))
    cv2.putText(img, text, (textX, textY), TEXT_FONT, fontScale=fontScale, color=TEXT_COLOR, thickness=thickness,
                lineType=TEXT_LINE_TYPE)


def align_left_multitext(img, text, pos, fontScale=1.0, thickness=1, padding=4, text_color=(0, 0, 0)):
    v_padding = 20
    text_lines = text.split('\n')
    text_line_height = cv2.getTextSize(text_lines[0], TEXT_FONT, fontScale, thickness)[0][1]
    text_height = len(text_lines) * text_line_height + (len(text_lines)-1) * v_padding
    textX = pos[0] + padding
    textY = int(pos[1] - (text_height / 2) + text_line_height)

    for i, text_line in enumerate(text_lines):
        bold_marker = "*"
        bold = text_line.startswith(bold_marker) and text_line.endswith(bold_marker)
        line_thickness = thickness+1 if bold else thickness
        if bold:
            text_line = text_line[len(bold_marker):len(text_line)-len(bold_marker)]
        pos = (textX, textY + (text_line_height + v_padding) * i)
        text_color = text_color if i == 0 else TEXT_COLOR
        cv2.putText(img, text_line, pos, TEXT_FONT, fontScale=fontScale, color=text_color, thickness=line_thickness,
                    lineType=TEXT_LINE_TYPE)


def centered_text(img, text, pos, fontScale=1, thickness=1):
    textsize = cv2.getTextSize(text, TEXT_FONT, fontScale, thickness)[0]
    textX = int(pos[0] - (textsize[0] / 2))
    textY = int(pos[1] + (textsize[1] / 2))
    cv2.putText(img, text, (textX, textY), TEXT_FONT, fontScale=fontScale, color=TEXT_COLOR, thickness=thickness,
                lineType=TEXT_LINE_TYPE)


def insert_img_into_grid(grid_img, img, row, col):
    extra_spacing_h = QUERY_EXTRA_SPACING if row > 0 else 0
    extra_spacing_w = QUERY_EXTRA_SPACING if col > 0 else 0
    width, height = img.shape[1], img.shape[0]
    hs = row * height + (row + 1) * GRID_SPACING_V + extra_spacing_h + TOP_MARGIN
    he = (row + 1) * height + (row + 1) * GRID_SPACING_V + extra_spacing_h + TOP_MARGIN
    ws = col * width + (col + 1) * GRID_SPACING_H + extra_spacing_w + LEFT_MARGIN
    we = (col + 1) * width + (col + 1) * GRID_SPACING_H + extra_spacing_w + LEFT_MARGIN
    grid_img[hs:he, ws:we, :] = img


def make_border(img, border_color, bw):
    img_b = cv2.copyMakeBorder(
        img,
        bw, bw, bw, bw,
        cv2.BORDER_CONSTANT,
        value=border_color
    )
    img_b = cv2.resize(img_b, (img.shape[1], img.shape[0]))
    return img_b


def draw_keypoints(img_to_insert, kp, model_img_size=None, radius=2, thickness=2, vis_thresh=0, color=None, use_confidence_color=False):
    if model_img_size is not None:
        kp = rescale_keypoints(kp, model_img_size, (img_to_insert.shape[1], img_to_insert.shape[0]))
    for xyck in kp:
        x, y, c, k = xyck
        if c > 0:
            if color is not None:
                if c > vis_thresh:
                    match_color = color
                    kp_thickness = 2
                    kp_radius = 1
                else:
                    match_color = BLACK
                    kp_thickness = thickness-1
                    kp_radius = radius-1
            elif use_confidence_color:
                if c > vis_thresh:
                    match_color = cmap(c/3, bytes=True)[0:-1]  # divided by three because hsv colormap goes from red to green inside [0, 0.333]
                    match_color = (int(match_color[2]), int(match_color[1]), int(match_color[0]))
                    kp_thickness = thickness
                    kp_radius = radius
                else:
                    match_color = RED
                    kp_thickness = thickness-1
                    kp_radius = radius-1
            else:
                if c > vis_thresh:
                    # print(f"kp[:, -1].max(): {kp[:, -1].max()} - kp[:, -1].min(): {kp[:, -1].min()} - k: {k}")
                    match_color = scale_lightness(cm.gist_rainbow(k / kp[:, -1].max())[0:-1])
                    # print(f"match_color 1: {match_color}")
                    match_color = (np.array(match_color) * 255).astype(np.uint8)
                    match_color = (int(match_color[2]), int(match_color[1]), int(match_color[0]))
                    # print(f"match_color 2: {match_color}")
                    kp_thickness = thickness
                    kp_radius = radius
                else:
                    match_color = RED
                    kp_thickness = thickness-1
                    kp_radius = radius-1
            cv2.circle(
                img_to_insert,
                (int(x), int(y)),
                color=match_color,
                thickness=kp_thickness,
                radius=kp_radius,
                # lineType=cv2.LINE_AA,
            )
    return img_to_insert
