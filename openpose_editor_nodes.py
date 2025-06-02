import json
import torch
import numpy as np
from .util import draw_pose_json, draw_pose

OpenposeJSON = dict

class OpenposeEditorNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "show_body": ("BOOLEAN", {"default": True}),
                "show_face": ("BOOLEAN", {"default": True}),
                "show_hands": ("BOOLEAN", {"default": True}),
                "resolution_x": ("INT", {"default": -1, "min": -1, "max": 12800}),
                "use_ground_plane": ("BOOLEAN", {"default": True}),
                "pose_marker_size": ("INT", { "default": 4, "min": 0, "max": 100 }),
                "face_marker_size": ("INT", { "default": 3, "min": 0, "max": 100 }),
                "hand_marker_size": ("INT", { "default": 2, "min": 0, "max": 100 }),
                "pelvis_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01 }),
                "torso_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01 }),
                "neck_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01 }),
                # --- 머리 및 눈 관련 스케일 ---
                "head_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01 }),
                "eye_distance_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01 }),
                "eye_height": ("FLOAT", { "default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1 }),
                "eyebrow_height": ("FLOAT", { "default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1 }), # 눈썹 높이 조절
                "left_eye_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01 }),
                "right_eye_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01 }),
                "left_eyebrow_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01 }),
                "right_eyebrow_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01 }),
                "mouth_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01 }),
                "nose_scale_face": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01 }),
                "face_shape_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01 }),
                # ---
                "shoulder_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01 }),
                "arm_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01 }),
                "leg_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01 }),
                "hands_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01 }),
                "overall_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01 }),
                "rotate_angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "translate_x": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.1}),
                "translate_y": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.1}),
                "POSE_JSON": ("STRING", {"multiline": True}),
                "POSE_KEYPOINT": ("POSE_KEYPOINT",{"default": None}),
                "Target_pose_keypoint": ("POSE_KEYPOINT", {"default": None}),
            },
        }

    RETURN_NAMES = ("POSE_IMAGE", "POSE_KEYPOINT", "POSE_JSON")
    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT", "STRING")
    OUTPUT_NODE = True
    FUNCTION = "load_pose"
    CATEGORY = "ultimate-openpose"

    def load_pose(self, show_body, show_face, show_hands, resolution_x, use_ground_plane,
                  pose_marker_size, face_marker_size, hand_marker_size,
                  pelvis_scale, torso_scale, neck_scale, head_scale, eye_distance_scale, eye_height, eyebrow_height,
                  left_eye_scale, right_eye_scale, left_eyebrow_scale, right_eyebrow_scale,
                  mouth_scale, nose_scale_face, face_shape_scale,
                  shoulder_scale, arm_scale, leg_scale, hands_scale, overall_scale,
                  rotate_angle, translate_x, translate_y,
                  POSE_JSON: str, POSE_KEYPOINT=None, Target_pose_keypoint=None) -> tuple[OpenposeJSON]:
        
        # 내부 함수인 process_pose에 Target_pose_keypoint를 전달하도록 수정
        def process_pose(pose_input_str_list, target_pose_obj=None, rot_angle=0, tr_x=0.0, tr_y=0.0):
            pose_imgs, final_keypoints_batch = draw_pose_json(
                pose_input_str_list, resolution_x, use_ground_plane, show_body, show_face, show_hands,
                pose_marker_size, face_marker_size, hand_marker_size,
                pelvis_scale, torso_scale, neck_scale, head_scale, eye_distance_scale, eye_height, eyebrow_height,
                left_eye_scale, right_eye_scale, left_eyebrow_scale, right_eyebrow_scale,
                mouth_scale, nose_scale_face, face_shape_scale,
                shoulder_scale, arm_scale, leg_scale, hands_scale, overall_scale,
                rot_angle, tr_x, tr_y,
                target_pose_keypoint_obj=target_pose_obj # util.py 함수로 Target_pose_keypoint 전달
            )
            
            if not pose_imgs: return None, None, None
            
            pose_imgs_np = np.array(pose_imgs).astype(np.float32) / 255
            final_json_str = json.dumps(final_keypoints_batch, indent=4)
            return torch.from_numpy(pose_imgs_np), final_keypoints_batch, final_json_str

        input_json_str = ""
        # 팔 길이 비교를 위해 POSE_KEYPOINT가 우선순위를 갖도록 순서 조정
        if POSE_KEYPOINT is not None:
            normalized_json_data = json.dumps(POSE_KEYPOINT, indent=4).replace("'",'"').replace('None','[]')
            if not isinstance(POSE_KEYPOINT, list):
                input_json_str = f'[{normalized_json_data}]'
            else:
                input_json_str = normalized_json_data
        elif POSE_JSON: 
            temp_json = POSE_JSON.replace("'",'"').replace('None','[]')
            try:
                parsed_json = json.loads(temp_json)
                input_json_str = f"[{temp_json}]" if not isinstance(parsed_json, list) else temp_json
            except json.JSONDecodeError: input_json_str = f"[{temp_json}]"
        
        if input_json_str:
            # process_pose 호출 시 Target_pose_keypoint 객체를 인자로 전달
            image_tensor, keypoint_obj_batch, json_str_batch = process_pose(input_json_str, Target_pose_keypoint, rotate_angle, translate_x, translate_y)
            if image_tensor is not None:
                return { "ui": {"POSE_JSON": [json_str_batch]}, "result": (image_tensor, keypoint_obj_batch, json_str_batch) }

        W, H = 512, 768
        blank_person = dict(pose_keypoints_2d=[], face_keypoints_2d=[], hand_left_keypoints_2d=[], hand_right_keypoints_2d=[])
        blank_output_keypoints = [{"people": [blank_person], "canvas_width": W, "canvas_height": H}]
        W_scaled = resolution_x if resolution_x >= 64 else W
        H_scaled = int(H*(W_scaled*1.0/W))
        blank_pose_for_draw = {"people": [blank_person]}
        pose_img = [draw_pose(blank_pose_for_draw, H_scaled, W_scaled, pose_marker_size, face_marker_size, hand_marker_size)]
        pose_img_np = np.array(pose_img).astype(np.float32) / 255
        return { "ui": {"POSE_JSON": [json.dumps(blank_output_keypoints)]}, "result": (torch.from_numpy(pose_img_np), blank_output_keypoints, json.dumps(blank_output_keypoints)) }
