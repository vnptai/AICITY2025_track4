#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import subprocess
import time
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Process VisDrone dataset with the complete pipeline')
    parser.add_argument('--input-dir', type=str, default='dataset/visdrone',
                        help='Path to the VisDrone dataset directory')
    parser.add_argument('--output-base-dir', type=str, default='dataset/processed',
                        help='Base directory for all output folders')
    parser.add_argument('--skip-steps', type=str, default='',
                        help='Comma-separated list of steps to skip (e.g., "merge,cut,aug")')
    return parser.parse_args()

def main():
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    output_base_dir = Path(args.output_base_dir)
    
    # Create output directories structure
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Determine which steps to skip
    skip_steps = [s.strip() for s in args.skip_steps.split(',') if s.strip()]
    
    
    # Step 1: Merge classes
    merged_dir = output_base_dir / 'visdrone_merged'
    if 'merge' not in skip_steps:
        print("\n" + "="*80)
        print("STEP 1: Merging classes according to target schema")
        print("="*80)
        cmd = [
            "python", "merge_classes.py",
            "--input-dir", str(input_dir),
            "--output-dir", str(merged_dir)
        ]
        subprocess.run(cmd)
    else:
        print("\nSkipping merge step...")
    
    # Step 2: Cut images
    cut_dir = output_base_dir / 'visdrone_cut'
    if 'cut' not in skip_steps:
        print("\n" + "="*80)
        print("STEP 2: Cutting images to proper dimensions")
        print("="*80)
        cmd = [
            "python", "cut_image.py",
            "--input-dir", str(merged_dir if 'merge' not in skip_steps else input_dir),
            "--output-dir", str(cut_dir)
        ]
        subprocess.run(cmd)
    else:
        print("\nSkipping cut step...")
    
    # Step 3: Augment with fisheye effect
    aug_dir = output_base_dir / 'visdrone_aug'
    if 'aug' not in skip_steps:
        print("\n" + "="*80)
        print("STEP 3: Applying fisheye augmentation")
        print("="*80)
        cmd = [
            "python", "aug_visdrone.py",
            "--input-dir", str(merged_dir if 'merge' not in skip_steps else input_dir),
            "--output-dir", str(aug_dir)
        ]
        subprocess.run(cmd)
    else:
        print("\nSkipping augmentation step...")
    
    # Final output is in two directories
    final_cut_dir = output_base_dir / 'final_cut'
    final_aug_dir = output_base_dir / 'final_aug'
    
    # Create final output directories if needed
    if not (('cut' in skip_steps) and ('aug' in skip_steps)):
        os.makedirs(final_cut_dir, exist_ok=True)
        os.makedirs(final_aug_dir, exist_ok=True)
        
        # Copy the cut images to final_cut directory
        if 'cut' not in skip_steps:
            print("\n" + "="*80)
            print("Creating final cut directory")
            print("="*80)
            os.makedirs(final_cut_dir / 'images', exist_ok=True)
            os.makedirs(final_cut_dir / 'labels', exist_ok=True)
            
            # Copy images and labels
            if (cut_dir / 'images').exists():
                for item in os.listdir(cut_dir / 'images'):
                    s = os.path.join(cut_dir / 'images', item)
                    d = os.path.join(final_cut_dir / 'images', item)
                    if not os.path.exists(d):
                        shutil.copy2(s, d)
            
            if (cut_dir / 'labels').exists():
                for item in os.listdir(cut_dir / 'labels'):
                    s = os.path.join(cut_dir / 'labels', item)
                    d = os.path.join(final_cut_dir / 'labels', item)
                    if not os.path.exists(d):
                        shutil.copy2(s, d)
        
        # Copy the augmented images to final_aug directory
        if 'aug' not in skip_steps:
            print("\n" + "="*80)
            print("Creating final aug directory")
            print("="*80)
            os.makedirs(final_aug_dir / 'images', exist_ok=True)
            os.makedirs(final_aug_dir / 'labels', exist_ok=True)
            
            # Copy images and labels
            if (aug_dir / 'images').exists():
                for item in os.listdir(aug_dir / 'images'):
                    s = os.path.join(aug_dir / 'images', item)
                    d = os.path.join(final_aug_dir / 'images', item)
                    if not os.path.exists(d):
                        shutil.copy2(s, d)
            
            if (aug_dir / 'labels').exists():
                for item in os.listdir(aug_dir / 'labels'):
                    s = os.path.join(aug_dir / 'labels', item)
                    d = os.path.join(final_aug_dir / 'labels', item)
                    if not os.path.exists(d):
                        shutil.copy2(s, d)
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)
    print(f"Input directory: {input_dir}")
    
    if 'cut' not in skip_steps:
        print(f"Final cut images output: {final_cut_dir}")
    if 'aug' not in skip_steps:
        print(f"Final augmented images output: {final_aug_dir}")
    
    print("\nStatistics:")
    if 'cut' not in skip_steps and (final_cut_dir / 'images').exists():
        cut_images = len(os.listdir(final_cut_dir / 'images'))
        cut_labels = len(os.listdir(final_cut_dir / 'labels')) if (final_cut_dir / 'labels').exists() else 0
        print(f"Cut images: {cut_images}, Cut labels: {cut_labels}")
    
    if 'aug' not in skip_steps and (final_aug_dir / 'images').exists():
        aug_images = len(os.listdir(final_aug_dir / 'images'))
        aug_labels = len(os.listdir(final_aug_dir / 'labels')) if (final_aug_dir / 'labels').exists() else 0
        print(f"Augmented images: {aug_images}, Augmented labels: {aug_labels}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\nTotal processing time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
