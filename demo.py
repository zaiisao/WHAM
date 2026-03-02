import numpy as np
np.float = float

import os
import argparse
import os.path as osp
from glob import glob
from collections import defaultdict

import cv2
import torch
import joblib
from loguru import logger
from progress.bar import Bar

from configs.config import get_cfg_defaults
from lib.data.datasets import CustomDataset
from lib.utils.imutils import avg_preds
from lib.utils.transforms import matrix_to_axis_angle
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor
from lib.models.smplify import TemporalSMPLify

try: 
    from lib.models.preproc.slam import SLAMModel
    _run_global = True
except: 
    logger.info('DPVO is not properly installed. Only estimate in local coordinates !')
    _run_global = False

def run(cfg,
        video,
        output_pth,
        network,
        calib=None,
        run_global=True,
        save_pkl=False,
        visualize=False):
    
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f'Faild to load video file {video}'
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # Whether or not estimating motion in global coordinates
    run_global = run_global and _run_global
    
    # Preprocess
    with torch.no_grad():
        if not (osp.exists(osp.join(output_pth, 'tracking_results.pth')) and 
                osp.exists(osp.join(output_pth, 'slam_results.pth'))):
            
            detector = DetectionModel(cfg.DEVICE.lower())
            extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)
            
            if run_global: slam = SLAMModel(video, output_pth, width, height, calib)
            else: slam = None
            
            bar = Bar('Preprocess: 2D detection and SLAM', fill='#', max=length)
            while (cap.isOpened()):
                flag, img = cap.read()
                if not flag: break
                
                # 2D detection and tracking
                detector.track(img, fps, length)
                
                # SLAM
                if slam is not None: 
                    slam.track()
                
                bar.next()

            tracking_results = detector.process(fps)

            # =========================================================================
            # PATCH: THE FRAGMENT STITCHER (Safely glues broken timelines together)
            # =========================================================================
            print("\n" + "="*40)
            print(f"Original Tracking IDs (Fragmented): {len(tracking_results)}")
            
            fragments = []
            for tid, data in tracking_results.items():
                f_key = 'frame_id' if 'frame_id' in data else 'frames'
                if f_key not in data or len(data[f_key]) == 0: continue
                
                fragments.append({
                    'id': tid,
                    'start_f': data[f_key][0],
                    'end_f': data[f_key][-1],
                    'data': data,
                    'start_bbox': data['bbox'][0],
                    'end_bbox': data['bbox'][-1]
                })
            
            # Sort fragments chronologically by their start time
            fragments.sort(key=lambda x: x['start_f'])
            
            stitched_timelines = []
            MAX_GAP_FRAMES = 90  # Wait up to 3 seconds for occlusion to end
            MAX_SPATIAL_DIST = 1.0 # Bounding box distance threshold
            
            for frag in fragments:
                matched = False
                for timeline in stitched_timelines:
                    last_frag = timeline[-1]
                    gap = frag['start_f'] - last_frag['end_f']
                    
                    # If the temporal gap is reasonable (person was hidden briefly)
                    if 0 < gap <= MAX_GAP_FRAMES:
                        # Check spatial distance (did they reappear near where they vanished?)
                        dx = frag['start_bbox'][0] - last_frag['end_bbox'][0]
                        dy = frag['start_bbox'][1] - last_frag['end_bbox'][1]
                        dist = (dx**2 + dy**2)**0.5
                        avg_scale = (frag['start_bbox'][2] + last_frag['end_bbox'][2]) / 2.0
                        
                        if dist / (avg_scale + 1e-6) < MAX_SPATIAL_DIST:
                            timeline.append(frag)
                            matched = True
                            break
                            
                if not matched:
                    stitched_timelines.append([frag])
            
            # Rebuild tracking_results from the glued timelines
            merged_results = {}
            for i, timeline in enumerate(stitched_timelines):
                merged_data = {'frame_id': [], 'bbox': [], 'keypoints': []}
                for frag in timeline:
                    f_key = 'frame_id' if 'frame_id' in frag['data'] else 'frames'
                    merged_data['frame_id'].extend(frag['data'][f_key])
                    merged_data['bbox'].extend(frag['data']['bbox'])
                    merged_data['keypoints'].extend(frag['data']['keypoints'])
                
                merged_data['frame_id'] = np.array(merged_data['frame_id'])
                merged_data['bbox'] = np.array(merged_data['bbox'])
                merged_data['keypoints'] = np.array(merged_data['keypoints'])
                
                # Fix keys and init lists to prevent KeyError downstream
                merged_data['features'] = []
                if cfg.FLIP_EVAL:
                    merged_data['flipped_bbox'] = []
                    merged_data['flipped_keypoints'] = []
                    merged_data['flipped_features'] = []
                    
                # Quality gate: Only keep timelines longer than 2 seconds
                if len(merged_data['frame_id']) > 60:
                    merged_results[i] = merged_data
                    
            tracking_results = merged_results
            print(f"Successfully stitched into {len(tracking_results)} continuous subjects.")
            print("="*40 + "\n")
            # =========================================================================
            
            if slam is not None: 
                slam_results = slam.process()
            else:
                slam_results = np.zeros((length, 7))
                slam_results[:, 3] = 1.0    # Unit quaternion
        
            # Extract image features
            # TODO: Merge this into the previous while loop with an online bbox smoothing.
            tracking_results = extractor.run(video, tracking_results)
            logger.info('Complete Data preprocessing!')
            
            # Save the processed data
            joblib.dump(tracking_results, osp.join(output_pth, 'tracking_results.pth'))
            joblib.dump(slam_results, osp.join(output_pth, 'slam_results.pth'))
            logger.info(f'Save processed data at {output_pth}')
        
        # If the processed data already exists, load the processed data
        else:
            tracking_results = joblib.load(osp.join(output_pth, 'tracking_results.pth'))
            slam_results = joblib.load(osp.join(output_pth, 'slam_results.pth'))
            logger.info(f'Already processed data exists at {output_pth} ! Load the data .')
    
    # Build dataset
    dataset = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)
    
    # run WHAM
    results = defaultdict(dict)
    
    n_subjs = len(dataset)
    for subj in range(n_subjs):

        with torch.no_grad():
            if cfg.FLIP_EVAL:
                # Forward pass with flipped input
                flipped_batch = dataset.load_data(subj, True)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = flipped_batch
                flipped_pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
                
                # Forward pass with normal input
                batch = dataset.load_data(subj)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
                
                # Merge two predictions
                flipped_pose, flipped_shape = flipped_pred['pose'].squeeze(0), flipped_pred['betas'].squeeze(0)
                pose, shape = pred['pose'].squeeze(0), pred['betas'].squeeze(0)
                flipped_pose, pose = flipped_pose.reshape(-1, 24, 6), pose.reshape(-1, 24, 6)
                avg_pose, avg_shape = avg_preds(pose, shape, flipped_pose, flipped_shape)
                avg_pose = avg_pose.reshape(-1, 144)
                avg_contact = (flipped_pred['contact'][..., [2, 3, 0, 1]] + pred['contact']) / 2
                
                # Refine trajectory with merged prediction
                network.pred_pose = avg_pose.view_as(network.pred_pose)
                network.pred_shape = avg_shape.view_as(network.pred_shape)
                network.pred_contact = avg_contact.view_as(network.pred_contact)
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)
            
            else:
                # data
                batch = dataset.load_data(subj)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                
                # inference
                pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
        
        # if False:
        if args.run_smplify:
            smplify = TemporalSMPLify(smpl, img_w=width, img_h=height, device=cfg.DEVICE)
            input_keypoints = dataset.tracking_results[_id]['keypoints']
            pred = smplify.fit(pred, input_keypoints, **kwargs)
            
            with torch.no_grad():
                network.pred_pose = pred['pose']
                network.pred_shape = pred['betas']
                network.pred_cam = pred['cam']
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)
        
        # ========= Store results ========= #
        pred_body_pose = matrix_to_axis_angle(pred['poses_body']).cpu().numpy().reshape(-1, 69)
        pred_root = matrix_to_axis_angle(pred['poses_root_cam']).cpu().numpy().reshape(-1, 3)
        pred_root_world = matrix_to_axis_angle(pred['poses_root_world']).cpu().numpy().reshape(-1, 3)
        pred_pose = np.concatenate((pred_root, pred_body_pose), axis=-1)
        pred_pose_world = np.concatenate((pred_root_world, pred_body_pose), axis=-1)
        pred_trans = (pred['trans_cam'] - network.output.offset).cpu().numpy()
        
        results[_id]['pose'] = pred_pose
        results[_id]['trans'] = pred_trans
        results[_id]['pose_world'] = pred_pose_world
        results[_id]['trans_world'] = pred['trans_world'].cpu().squeeze(0).numpy()
        results[_id]['betas'] = pred['betas'].cpu().squeeze(0).numpy()
        results[_id]['verts'] = (pred['verts_cam'] + pred['trans_cam'].unsqueeze(1)).cpu().numpy()
        results[_id]['frame_ids'] = frame_id
    
    if save_pkl:
        joblib.dump(results, osp.join(output_pth, "wham_output.pkl"))
     
    # Visualize
    # if visualize:
    #     from lib.vis.run_vis import run_vis_on_demo
    #     with torch.no_grad():
    #         run_vis_on_demo(cfg, video, results, output_pth, network.smpl, vis_global=run_global)
    if visualize:
        from lib.vis.run_vis import run_vis_on_demo
        import subprocess
        
        with torch.no_grad():
            for _id in results.keys():
                logger.info(f"Generating full-length render for Fragment ID: {_id}")
                
                # 1. Create a specific sub-folder for this fragment
                id_output_pth = osp.join(output_pth, f"fragment_{_id}")
                os.makedirs(id_output_pth, exist_ok=True)
                
                single_subj_result = {_id: results[_id]}
                
                # 2. Render the full video (WHAM's default behavior)
                run_vis_on_demo(cfg, video, single_subj_result, id_output_pth, network.smpl, vis_global=run_global)
                
                # =========================================================================
                # 3. NEW PATCH: THE AUTO-CROPPER
                # =========================================================================
                # Find the video WHAM just generated
                generated_videos = glob(osp.join(id_output_pth, '*.mp4'))
                if len(generated_videos) > 0:
                    raw_render = generated_videos[0]
                    cropped_render = osp.join(id_output_pth, f"cropped_id{_id}.mp4")
                    
                    # Calculate exact timestamps based on frame indices
                    frames = results[_id]['frame_ids']
                    start_frame = int(min(frames))
                    end_frame = int(max(frames))
                    
                    start_time = start_frame / fps
                    duration = (end_frame - start_frame + 1) / fps
                    
                    logger.info(f"Cropping ID {_id} from {start_time:.2f}s to {start_time+duration:.2f}s")
                    
                    # FFmpeg command: Re-encode to ensure exact frame-level cuts
                    cmd = [
                        'ffmpeg', '-y', 
                        '-ss', str(start_time), 
                        '-t', str(duration),
                        '-i', raw_render, 
                        '-c:v', 'libx264', '-crf', '23', '-preset', 'fast', 
                        cropped_render
                    ]
                    
                    # Execute FFmpeg silently
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    # Delete the uncropped full video to save disk space
                    os.remove(raw_render)
                    logger.info(f"Saved precise cropped clip: {cropped_render}")
                # =========================================================================        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video', type=str, 
                        default='examples/demo_video.mp4', 
                        help='input video path or youtube link')

    parser.add_argument('--output_pth', type=str, default='output/demo', 
                        help='output folder to write results')
    
    parser.add_argument('--calib', type=str, default=None, 
                        help='Camera calibration file path')

    parser.add_argument('--estimate_local_only', action='store_true',
                        help='Only estimate motion in camera coordinate if True')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the output mesh if True')
    
    parser.add_argument('--save_pkl', action='store_true',
                        help='Save output as pkl file')
    
    parser.add_argument('--run_smplify', action='store_true',
                        help='Run Temporal SMPLify for post processing')

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')
    
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')    
    
    # ========= Load WHAM ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()
    
    # Output folder
    sequence = '.'.join(args.video.split('/')[-1].split('.')[:-1])
    output_pth = osp.join(args.output_pth, sequence)
    os.makedirs(output_pth, exist_ok=True)
    
    run(cfg, 
        args.video, 
        output_pth, 
        network, 
        args.calib, 
        run_global=not args.estimate_local_only, 
        save_pkl=args.save_pkl,
        visualize=args.visualize)
        
    print()
    logger.info('Done !')