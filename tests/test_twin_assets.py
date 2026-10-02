import json
import math
from pathlib import Path
import pytest
import numpy as np

PROCESS_DIR = Path("data/process")
LABELS_DIR = PROCESS_DIR / "labels"
TWIN_DIR = PROCESS_DIR / "twin"
ASSETS_DIR = Path("web/twin/assets")
MANIFEST_FILE = PROCESS_DIR / "manifest.json"

EXPECTED_LINKS = [
    "link0", "link1", "link2", "link3", "link4", 
    "link5", "link6", "link7", "hand", "leftfinger", "rightfinger"
]

def test_directories_exist():
    assert PROCESS_DIR.exists()
    assert TWIN_DIR.exists()
    assert ASSETS_DIR.exists()

def test_glb_assets_load_and_size():
    notice_file = ASSETS_DIR / "NOTICE.txt"
    assert notice_file.exists(), "NOTICE file is missing"
    
    notice_text = notice_file.read_text().lower()
    assert "robosuite" in notice_text, "NOTICE must mention robosuite"
    assert "mit" in notice_text, "NOTICE must mention MIT"
    assert "franka" in notice_text, "NOTICE must mention Franka"
    assert "apache" in notice_text, "NOTICE must mention Apache-2.0"
    
    total_size = 0
    
    for link_name in EXPECTED_LINKS:
        glb_file = ASSETS_DIR / f"{link_name}.glb"
        assert glb_file.exists(), f"Missing GLB file {glb_file.name}"
        
        total_size += glb_file.stat().st_size
        
        # Check GLB magic header
        with open(glb_file, "rb") as f:
            magic = f.read(4)
            assert magic == b"glTF", f"{glb_file.name} is not a valid GLB (missing glTF magic)"
            
        import trimesh
        mesh = trimesh.load(glb_file, force="mesh")
        assert mesh is not None, f"Failed to load {glb_file.name} with trimesh"
            
    assert total_size < 8 * 1024 * 1024, f"Total GLB size {total_size} bytes exceeds 8 MB"

def test_index_json_schema():
    index_file = TWIN_DIR / "index.json"
    assert index_file.exists(), "index.json missing"
    
    with open(index_file) as f:
        index_data = json.load(f)
        
    assert "links" in index_data
    assert len(index_data["links"]) == len(EXPECTED_LINKS)
    
    for i, link_obj in enumerate(index_data["links"]):
        assert link_obj["name"] == EXPECTED_LINKS[i], f"Expected {EXPECTED_LINKS[i]} at index {i}, got {link_obj['name']}"
        assert link_obj["mesh"] == f"{EXPECTED_LINKS[i]}.glb"

def test_twin_json_frames_and_replay_accuracy():
    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)
        
    for ep in manifest["episodes"]:
        ep_id = ep["id"]
        label_file = LABELS_DIR / f"{ep_id}.json"
        twin_file = TWIN_DIR / f"{ep_id}.json"
        
        assert twin_file.exists(), f"Missing twin file for episode {ep_id}"
        
        with open(label_file) as f:
            label_data = json.load(f)
            
        with open(twin_file) as f:
            twin_data = json.load(f)
            
        assert twin_data["links"] == EXPECTED_LINKS
        assert len(twin_data["frames"]) == 180, f"Episode {ep_id} does not have 180 frames"
        
        miss_jump = False
        prev_cube_pos = None
        
        eef_errors = []
        
        for i, twin_frame in enumerate(twin_data["frames"]):
            label_frame = label_data["frames"][i]
            
            # eef_pos check
            twin_eef = np.array(twin_frame["eef_pos"])
            label_eef = np.array(label_frame["eef_pos"])
            eef_err = np.linalg.norm(label_eef - twin_eef)
            eef_errors.append(eef_err)
            
            if i < 40:
                assert eef_err <= 0.025, f"EEF pos error {eef_err} > 0.025m in {ep_id} frame {i}"
            else:
                assert eef_err <= 0.005, f"EEF pos error {eef_err} > 0.005m in {ep_id} frame {i}"
            
            # cube_pos check
            twin_cube = np.array(twin_frame["cube_pos"])
            label_cube = np.array(label_frame["cube_pos"])
            cube_err = np.linalg.norm(label_cube - twin_cube)
            assert cube_err <= 0.005, f"Cube pos error {cube_err} > 0.005m in {ep_id} frame {i}"
            
            # gripper check
            assert twin_frame["gripper_closed"] == label_frame["gripper_closed"], f"Gripper closed mismatch in {ep_id} frame {i}"
            
            if "miss" in ep_id and prev_cube_pos is not None:
                jump = np.linalg.norm(twin_cube - prev_cube_pos)
                if 0.04 <= jump <= 0.06:
                    miss_jump = True
            
            prev_cube_pos = twin_cube
            
            # Unit quaternions check for links
            assert len(twin_frame["link_pos"]) == 11
            assert len(twin_frame["link_quat"]) == 11
            
            for j, quat in enumerate(twin_frame["link_quat"]):
                norm = np.linalg.norm(quat)
                assert math.isclose(norm, 1.0, abs_tol=1e-3), f"Non-unit quaternion {norm} for {EXPECTED_LINKS[j]} in {ep_id} frame {i}"
                
            # Unit quaternion check for cube and eef
            cube_quat_norm = np.linalg.norm(twin_frame["cube_quat"])
            assert math.isclose(cube_quat_norm, 1.0, abs_tol=1e-3), f"Non-unit cube quaternion {cube_quat_norm} in {ep_id} frame {i}"
            
            eef_quat_norm = np.linalg.norm(twin_frame["eef_quat"])
            assert math.isclose(eef_quat_norm, 1.0, abs_tol=1e-3), f"Non-unit eef quaternion {eef_quat_norm} in {ep_id} frame {i}"
                
        # median over frames of the eef error <= 0.001 m per episode
        median_eef_error = np.median(eef_errors)
        assert median_eef_error <= 0.001, f"Median EEF error {median_eef_error} > 0.001m in episode {ep_id}"
                
        if "miss" in ep_id:
            assert miss_jump, f"Miss episode {ep_id} did not have a 4-6cm cube jump"
