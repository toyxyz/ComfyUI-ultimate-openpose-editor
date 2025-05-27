import math
import json
import numpy as np
import matplotlib
import cv2
from comfy.utils import ProgressBar

eps = 0.01

#Face_landmark : Left Eye(42, 43, 44, 45, 46, 47, 69), Right eye(36, 37, 38, 39, 40, 41, 68), lefteyebrow(22,23,24,25,26), righteyebrow(17,18,19,20,21),  mouth(48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67), nose(27, 28, 29, 30, 31, 32, 33, 34, 35), face_shape(0~16)


def scale(point, scale_factor, pivot):
    if not isinstance(point, np.ndarray): point = np.array(point)
    if not isinstance(pivot, np.ndarray): pivot = np.array(pivot)
    return pivot + (point - pivot) * scale_factor

def draw_pose_json(pose_json_str, resolution_x, use_ground_plane, show_body, show_face, show_hands,
                   pose_marker_size, face_marker_size, hand_marker_size,
                   pelvis_scale, torso_scale, neck_scale, head_scale, eye_distance_scale, eye_height, eyebrow_height,
                   left_eye_scale, right_eye_scale, left_eyebrow_scale, right_eyebrow_scale,
                   mouth_scale, nose_scale_face, face_shape_scale,
                   shoulder_scale, arm_scale, leg_scale, hands_scale, overall_scale):
    pose_imgs = []
    all_frames_keypoints_output = []

    if pose_json_str:
        images_data_list = json.loads(pose_json_str)
        if not isinstance(images_data_list, list): images_data_list = [images_data_list]

        pbar = ProgressBar(len(images_data_list))
        
        KP = {
            "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
            "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
            "LEye": 15, "REar": 16, "LEar": 17
        }
        
        FACE_KP_GROUPS_INDICES = {
            "Left_Eye": [42, 43, 44, 45, 46, 47, 69],
            "Right_Eye": [36, 37, 38, 39, 40, 41, 68],
            "Left_Eyebrow": [22, 23, 24, 25, 26],
            "Right_Eyebrow": [17, 18, 19, 20, 21],
            "Mouth": list(range(48, 68)),
            "Nose_Face": list(range(27, 36)),
            "Face_Shape": list(range(0, 17))
        }
        
        INDIVIDUAL_FACE_SCALES = {
            "Left_Eye": left_eye_scale, "Right_Eye": right_eye_scale,
            "Left_Eyebrow": left_eyebrow_scale, "Right_Eyebrow": right_eyebrow_scale,
            "Mouth": mouth_scale, "Nose_Face": nose_scale_face,
            "Face_Shape": face_shape_scale
        }

        BODY_HEAD_PARTS = {KP["REye"], KP["LEye"], KP["REar"], KP["LEar"]}
        R_LEG_INDICES = {KP["RKnee"], KP["RAnkle"]}
        L_LEG_INDICES = {KP["LKnee"], KP["LAnkle"]}
        FEET_INDICES = {KP["RAnkle"], KP["LAnkle"]}

        for image_data in images_data_list:
            if 'people' not in image_data or not image_data['people']:
                pbar.update(1); continue
            
            figures = image_data['people']
            H = image_data['canvas_height']
            W = image_data['canvas_width']
            
            current_image_people_data_for_output = []
            all_scaled_candidates_for_drawing, all_scaled_faces_for_drawing, all_scaled_hands_for_drawing = [], [], []
            final_subset_for_drawing = [[]] 
            
            for fig_idx, figure in enumerate(figures):
                body_raw, face_raw, lhand_raw, rhand_raw = [figure.get(k, []) for k in ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']]

                if not body_raw or len(body_raw) < (KP["LEar"] + 1) * 3: continue
                
                initial_candidate = np.array([body_raw[i:i+2] for i in range(0, len(body_raw), 3)])
                confidence_scores_body = [body_raw[i*3+2] for i in range(len(initial_candidate))]
                scaled_candidate_np = initial_candidate.copy()

                r_hip_orig, l_hip_orig = initial_candidate[KP["RHip"]], initial_candidate[KP["LHip"]]
                hip_center_orig = (r_hip_orig + l_hip_orig) / 2 
                neck_orig, r_shoulder_orig, l_shoulder_orig, nose_orig = [initial_candidate[KP[k]] for k in ["Neck", "RShoulder", "LShoulder", "Nose"]]
                lwrist_orig, rwrist_orig = initial_candidate[KP["LWrist"]], initial_candidate[KP["RWrist"]]

                r_hip_final = scale(r_hip_orig, pelvis_scale, hip_center_orig)
                l_hip_final = scale(l_hip_orig, pelvis_scale, hip_center_orig)
                scaled_candidate_np[KP["RHip"]], scaled_candidate_np[KP["LHip"]] = r_hip_final, l_hip_final
                
                hip_center_final = (r_hip_final + l_hip_final) / 2
                neck_final = scale(neck_orig, torso_scale, hip_center_final)
                scaled_candidate_np[KP["Neck"]] = neck_final

                scaled_candidate_np[KP["RShoulder"]] = neck_final + (r_shoulder_orig - neck_orig) * shoulder_scale
                scaled_candidate_np[KP["LShoulder"]] = neck_final + (l_shoulder_orig - neck_orig) * shoulder_scale
                
                r_shoulder_final, l_shoulder_final = scaled_candidate_np[KP["RShoulder"]], scaled_candidate_np[KP["LShoulder"]]
                for i in [KP["RElbow"], KP["RWrist"]]: scaled_candidate_np[i] = r_shoulder_final + (initial_candidate[i] - r_shoulder_orig) * arm_scale
                for i in [KP["LElbow"], KP["LWrist"]]: scaled_candidate_np[i] = l_shoulder_final + (initial_candidate[i] - l_shoulder_orig) * arm_scale
                    
                for i in R_LEG_INDICES: scaled_candidate_np[i] = r_hip_final + (initial_candidate[i] - r_hip_orig) * leg_scale
                for i in L_LEG_INDICES: scaled_candidate_np[i] = l_hip_final + (initial_candidate[i] - l_hip_orig) * leg_scale
                
                nose_body_final = neck_final + (nose_orig - neck_orig) * neck_scale
                scaled_candidate_np[KP["Nose"]] = nose_body_final
                
                effective_nose_translation = nose_body_final - nose_orig
                for i in BODY_HEAD_PARTS:
                    part_moved_with_nose = initial_candidate[i] + effective_nose_translation
                    scaled_candidate_np[i] = scale(part_moved_with_nose, head_scale, nose_body_final)

                face_points_scaled_current_fig = []
                if face_raw:
                    face_points_orig = [np.array(face_raw[i:i+2]) for i in range(0, len(face_raw), 3)]
                    num_face_points = len(face_points_orig)
                    
                    face_points_positioned = [p + effective_nose_translation for p in face_points_orig]
                    face_points_after_global_head_scale = [scale(p, head_scale, nose_body_final) for p in face_points_positioned]
                    face_points_scaled_current_fig = list(face_points_after_global_head_scale)

                    # --- [모듈형 이동 로직] ---

                    # 1. eye_distance_scale로 인한 수평 이동량 계산
                    reye_pos_after_head_scale = scale(initial_candidate[KP["REye"]] + effective_nose_translation, head_scale, nose_body_final)
                    leye_pos_after_head_scale = scale(initial_candidate[KP["LEye"]] + effective_nose_translation, head_scale, nose_body_final)
                    eye_center = (reye_pos_after_head_scale + leye_pos_after_head_scale) / 2
                    reye_pos_after_dist_scale = scale(reye_pos_after_head_scale, eye_distance_scale, eye_center)
                    leye_pos_after_dist_scale = scale(leye_pos_after_head_scale, eye_distance_scale, eye_center)
                    right_dist_translation = reye_pos_after_dist_scale - reye_pos_after_head_scale
                    left_dist_translation = leye_pos_after_dist_scale - leye_pos_after_head_scale

                    # 2. eye_height와 eyebrow_height로 인한 수직 이동량 각각 계산
                    eye_height_offset = np.array([0.0, 0.0])
                    eyebrow_height_offset = np.array([0.0, 0.0])
                    direction_vector = nose_body_final - neck_final
                    norm_direction = np.linalg.norm(direction_vector)
                    if norm_direction > eps:
                        unit_direction = direction_vector / norm_direction
                        if abs(eye_height) > eps: eye_height_offset = unit_direction * eye_height
                        if abs(eyebrow_height) > eps: eyebrow_height_offset = unit_direction * eyebrow_height
                    
                    # 3. 각 그룹에 적용될 최종 이동량 정의
                    group_translations = {
                        "Right_Eye": right_dist_translation + eye_height_offset,
                        "Left_Eye": left_dist_translation + eye_height_offset,
                        "Right_Eyebrow": right_dist_translation + eyebrow_height_offset,
                        "Left_Eyebrow": left_dist_translation + eyebrow_height_offset,
                    }

                    # 4. 몸통의 눈 키포인트 최종 위치 업데이트 (eye_height 오프셋만 적용)
                    scaled_candidate_np[KP["REye"]] = reye_pos_after_dist_scale + eye_height_offset
                    scaled_candidate_np[KP["LEye"]] = leye_pos_after_dist_scale + eye_height_offset
                    
                    # 5. 얼굴 랜드마크 그룹에 최종 이동량 및 개별 스케일 적용
                    for group_name, indices in FACE_KP_GROUPS_INDICES.items():
                        group_scale_modifier = INDIVIDUAL_FACE_SCALES.get(group_name, 1.0)
                        valid_indices = [idx for idx in indices if idx < num_face_points]
                        if not valid_indices: continue

                        points_after_head_scale = [face_points_after_global_head_scale[idx] for idx in valid_indices]

                        if group_name in group_translations:
                            points_after_translation = [p + group_translations[group_name] for p in points_after_head_scale]
                        else:
                            points_after_translation = points_after_head_scale

                        if abs(group_scale_modifier - 1.0) > eps:
                            if group_name == "Face_Shape":
                                pivot = nose_body_final
                                direction_vector = neck_final - nose_body_final
                                norm_direction = np.linalg.norm(direction_vector)
                                if norm_direction > eps:
                                    unit_direction = direction_vector / norm_direction
                                    final_points = []
                                    for p in points_after_translation:
                                        point_vector = p - pivot
                                        proj_length = np.dot(point_vector, unit_direction)
                                        parallel_component = proj_length * unit_direction
                                        perpendicular_component = point_vector - parallel_component
                                        scaled_parallel_component = parallel_component * group_scale_modifier
                                        new_point = pivot + scaled_parallel_component + perpendicular_component
                                        final_points.append(new_point)
                                else:
                                    final_points = points_after_translation
                            else:
                                pivot = np.mean(points_after_translation, axis=0)
                                final_points = [scale(p, group_scale_modifier, pivot) for p in points_after_translation]
                        else:
                            final_points = points_after_translation

                        for i, idx in enumerate(valid_indices):
                            face_points_scaled_current_fig[idx] = final_points[i]
                
                lwrist_final_calc, rwrist_final_calc = scaled_candidate_np[KP["LWrist"]], scaled_candidate_np[KP["RWrist"]]
                lhand_scaled_current_fig = [(scale(np.array(lhand_raw[i:i+2]), hands_scale, lwrist_orig) + (lwrist_final_calc - lwrist_orig)) for i in range(0, len(lhand_raw), 3)] if lhand_raw else []
                rhand_scaled_current_fig = [(scale(np.array(rhand_raw[i:i+2]), hands_scale, rwrist_orig) + (rwrist_final_calc - rwrist_orig)) for i in range(0, len(rhand_raw), 3)] if rhand_raw else []

                scales_to_check = [leg_scale, torso_scale, overall_scale, pelvis_scale, head_scale] 
                is_scaling_active = any(abs(s - 1.0) > 0.001 for s in scales_to_check)
                
                candidate_list_current_fig_np = scaled_candidate_np
                face_list_current_fig_np = np.array(face_points_scaled_current_fig) if face_points_scaled_current_fig else np.array([])
                lhand_list_current_fig_np = np.array(lhand_scaled_current_fig) if lhand_scaled_current_fig else np.array([])
                rhand_list_current_fig_np = np.array(rhand_scaled_current_fig) if rhand_scaled_current_fig else np.array([])
                                
                if use_ground_plane and is_scaling_active:
                    ground_y_coord = H
                    orig_feet_coords = [initial_candidate[i] for i in FEET_INDICES if i < len(initial_candidate)]
                    orig_lowest_y = max(p[1] for p in orig_feet_coords) if orig_feet_coords else H
                    orig_dist_to_ground = ground_y_coord - orig_lowest_y

                    feet_coords_for_overall_pivot = [candidate_list_current_fig_np[i] for i in FEET_INDICES if i < len(candidate_list_current_fig_np)]
                    
                    if feet_coords_for_overall_pivot:
                        feet_pos_pivot = np.mean(feet_coords_for_overall_pivot, axis=0)
                        candidate_list_current_fig_np = np.array([scale(p, overall_scale, feet_pos_pivot) for p in candidate_list_current_fig_np])
                        if face_list_current_fig_np.size > 0: face_list_current_fig_np = np.array([scale(p, overall_scale, feet_pos_pivot) for p in face_list_current_fig_np])
                        if lhand_list_current_fig_np.size > 0: lhand_list_current_fig_np = np.array([scale(p, overall_scale, feet_pos_pivot) for p in lhand_list_current_fig_np])
                        if rhand_list_current_fig_np.size > 0: rhand_list_current_fig_np = np.array([scale(p, overall_scale, feet_pos_pivot) for p in rhand_list_current_fig_np])

                        final_feet_coords = [candidate_list_current_fig_np[i] for i in FEET_INDICES if i < len(candidate_list_current_fig_np)]
                        if final_feet_coords:
                            final_lowest_y = max(p[1] for p in final_feet_coords)
                            desired_final_y = ground_y_coord - orig_dist_to_ground
                            vertical_translation = desired_final_y - final_lowest_y
                            
                            candidate_list_current_fig_np = candidate_list_current_fig_np + np.array([0, vertical_translation])
                            if face_list_current_fig_np.size > 0: face_list_current_fig_np = face_list_current_fig_np + np.array([0, vertical_translation])
                            if lhand_list_current_fig_np.size > 0: lhand_list_current_fig_np = lhand_list_current_fig_np + np.array([0, vertical_translation])
                            if rhand_list_current_fig_np.size > 0: rhand_list_current_fig_np = rhand_list_current_fig_np + np.array([0, vertical_translation])
                else: 
                    center_pivot = [W * 0.5, H * 0.5]
                    candidate_list_current_fig_np = np.array([scale(p, overall_scale, center_pivot) for p in candidate_list_current_fig_np])
                    if face_list_current_fig_np.size > 0: face_list_current_fig_np = np.array([scale(p, overall_scale, center_pivot) for p in face_list_current_fig_np])
                    if lhand_list_current_fig_np.size > 0: lhand_list_current_fig_np = np.array([scale(p, overall_scale, center_pivot) for p in lhand_list_current_fig_np])
                    if rhand_list_current_fig_np.size > 0: rhand_list_current_fig_np = np.array([scale(p, overall_scale, center_pivot) for p in rhand_list_current_fig_np])

                body_kps_out_current_fig = [item for i, p in enumerate(candidate_list_current_fig_np) for item in [p[0], p[1], confidence_scores_body[i]]]
                face_kps_out_current_fig = [item for p in face_list_current_fig_np for item in [p[0], p[1], 1.0]] if face_list_current_fig_np.size > 0 else []
                lhand_kps_out_current_fig = [item for p in lhand_list_current_fig_np for item in [p[0], p[1], 1.0]] if lhand_list_current_fig_np.size > 0 else []
                rhand_kps_out_current_fig = [item for p in rhand_list_current_fig_np for item in [p[0], p[1], 1.0]] if rhand_list_current_fig_np.size > 0 else []
                
                current_image_people_data_for_output.append({
                    "pose_keypoints_2d": body_kps_out_current_fig, "face_keypoints_2d": face_kps_out_current_fig,
                    "hand_left_keypoints_2d": lhand_kps_out_current_fig, "hand_right_keypoints_2d": rhand_kps_out_current_fig,
                })

                all_scaled_candidates_for_drawing.extend(candidate_list_current_fig_np.tolist())
                if face_list_current_fig_np.size > 0: all_scaled_faces_for_drawing.extend(face_list_current_fig_np.tolist())
                if lhand_list_current_fig_np.size > 0: all_scaled_hands_for_drawing.append(lhand_list_current_fig_np.tolist())
                if rhand_list_current_fig_np.size > 0: all_scaled_hands_for_drawing.append(rhand_list_current_fig_np.tolist())

                if fig_idx == 0 and not final_subset_for_drawing[0]:
                    final_subset_for_drawing[0].extend([i if body_raw[i*3+2]>0 else -1 for i in range(len(candidate_list_current_fig_np))])
                else:
                    prev_candidate_count = len(all_scaled_candidates_for_drawing) - len(candidate_list_current_fig_np)
                    final_subset_for_drawing.append([prev_candidate_count+i if body_raw[i*3+2]>0 else -1 for i in range(len(candidate_list_current_fig_np))])
            
            current_frame_keypoint_object = { "people": current_image_people_data_for_output, "canvas_width": W, "canvas_height": H }
            all_frames_keypoints_output.append(current_frame_keypoint_object)
            
            candidate_norm, faces_norm = all_scaled_candidates_for_drawing, all_scaled_faces_for_drawing
            hands_norm_for_drawing = all_scaled_hands_for_drawing 
            
            if candidate_norm:
                candidate_np_norm = np.array(candidate_norm).astype(float); candidate_np_norm[...,0] /= float(W); candidate_np_norm[...,1] /= float(H)
                candidate_norm = candidate_np_norm.tolist()
            if faces_norm: 
                faces_np_norm = np.array(faces_norm).astype(float); 
                if faces_np_norm.size > 0: faces_np_norm[...,0] /= float(W); faces_np_norm[...,1] /= float(H)
                faces_norm = faces_np_norm.tolist()
            
            hands_final_norm_for_drawing = []
            if hands_norm_for_drawing: 
                for hand_kps_list in hands_norm_for_drawing:
                    current_normalized_hand = []
                    for point_list in hand_kps_list: 
                        if not isinstance(point_list, (list, np.ndarray)) or len(point_list) != 2: continue 
                        norm_point = np.array(point_list).astype(float)
                        norm_point[0] /= float(W)
                        norm_point[1] /= float(H)
                        current_normalized_hand.append(norm_point.tolist())
                    if current_normalized_hand : hands_final_norm_for_drawing.append(current_normalized_hand)
            
            bodies = dict(candidate=candidate_norm, subset=final_subset_for_drawing)
            original_face_exists = any(fig.get('face_keypoints_2d') for fig in figures)
            original_lhand_exists = any(fig.get('hand_left_keypoints_2d') for fig in figures)
            original_rhand_exists = any(fig.get('hand_right_keypoints_2d') for fig in figures)

            pose = dict(
                bodies=bodies if show_body else {'candidate':[], 'subset':[]}, 
                faces=faces_norm if show_face and original_face_exists else [], 
                hands=hands_final_norm_for_drawing if show_hands and (original_lhand_exists or original_rhand_exists) else []
            )
            W_scaled = resolution_x if resolution_x >= 64 else W
            H_scaled = int(H*(W_scaled*1.0/W))
            pose_imgs.append(draw_pose(pose, H_scaled, W_scaled, pose_marker_size, face_marker_size, hand_marker_size))
            pbar.update(1)

    return pose_imgs, all_frames_keypoints_output

def draw_pose(pose, H, W, pose_marker_size, face_marker_size, hand_marker_size):
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    body_render_info = pose.get('bodies', {})
    candidate = body_render_info.get('candidate', [])
    subset = body_render_info.get('subset', [])
    faces_data = pose.get('faces', []) 
    hands_data = pose.get('hands', [])

    if candidate and subset and np.array(candidate).size > 0 : canvas = draw_bodypose(canvas, np.array(candidate), np.array(subset), pose_marker_size)
    if hands_data and np.array(hands_data).size > 0 : canvas = draw_handpose(canvas, hands_data, hand_marker_size)
    if faces_data and np.array(faces_data).size > 0 : canvas = draw_facepose(canvas, faces_data, face_marker_size)
    return canvas

def draw_bodypose(canvas, candidate, subset, pose_marker_size):
    H, W, C = canvas.shape
    limbSeq = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    if candidate.ndim != 2 or candidate.shape[1] != 2: return canvas 
    for i in range(len(limbSeq)):
        for n in range(len(subset)):
            limb = limbSeq[i]
            if max(limb) >= subset.shape[1]: continue
            index = subset[n][np.array(limb)].astype(int)
            if -1 in index or max(index) >= len(candidate): continue
            Y, X = candidate[index, 0] * float(W), candidate[index, 1] * float(H)
            mX, mY = np.mean(X), np.mean(Y)
            length = np.linalg.norm(np.array([X[0], Y[0]]) - np.array([X[1], Y[1]]))
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            if length < 1: continue
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), pose_marker_size), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i % len(colors)])
    for n in range(len(subset)):
        for i in range(subset.shape[1]): 
            index = int(subset[n][i])
            if index == -1 or index >= len(candidate): continue
            x, y = candidate[index][0:2]
            x, y = int(x * W), int(y * H)
            cv2.circle(canvas, (x, y), pose_marker_size, colors[i % len(colors)], thickness=-1)
    return canvas

def draw_handpose(canvas, all_hand_peaks, hand_marker_size):
    H, W, C = canvas.shape
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    for peaks_list_for_one_hand in all_hand_peaks:
        peaks_np = np.array(peaks_list_for_one_hand)
        if peaks_np.ndim != 2 or peaks_np.shape[1] != 2: continue
        for ie, e in enumerate(edges):
            if e[0] >= len(peaks_np) or e[1] >= len(peaks_np): continue
            x1_coord, y1_coord = peaks_np[e[0]] 
            x2_coord, y2_coord = peaks_np[e[1]]
            x1, y1 = int(x1_coord * W), int(y1_coord * H)
            x2, y2 = int(x2_coord * W), int(y2_coord * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=max(1, hand_marker_size))
        for i, keyponit in enumerate(peaks_np):
            x_coord, y_coord = keyponit
            x, y = int(x_coord * W), int(y_coord * H)
            if x > eps and y > eps: cv2.circle(canvas, (x, y), max(1, hand_marker_size) + 1, (0, 0, 255), thickness=-1)
    return canvas

def draw_facepose(canvas, all_lmks, face_marker_size):
    H, W, C = canvas.shape
    lmks_np = np.array(all_lmks) 
    if lmks_np.ndim != 2 or lmks_np.shape[1] != 2: return canvas
    for lmk in lmks_np:
        x_coord, y_coord = lmk
        x, y = int(x_coord * W), int(y_coord * H)
        if x > eps and y > eps: cv2.circle(canvas, (x, y), face_marker_size, (255, 255, 255), thickness=-1)
    return canvas
