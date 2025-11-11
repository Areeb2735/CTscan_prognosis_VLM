import os
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Constants
TARGET_SPACING = (0.75, 0.75, 1.5)
TARGET_SIZE = (480, 480, 240)

def resample_image(image, target_spacing=TARGET_SPACING, new_size=None):
    """Resample image to target spacing."""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    original_direction = image.GetDirection()
    original_origin = image.GetOrigin()
    
    if new_size is None:
        # Calculate physical size based on original image
        physical_size = np.array([
            size * spacing 
            for size, spacing in zip(original_size, original_spacing)
        ])
        # Calculate new size to maintain physical size
        new_size = np.ceil(physical_size / np.array(target_spacing)).astype(int)
    
    # Create resampling filter
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize([int(s) for s in new_size])
    resampler.SetOutputDirection(original_direction)
    resampler.SetOutputOrigin(original_origin)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    
    # Use appropriate interpolator
    if image.GetPixelID() in [sitk.sitkUInt8, sitk.sitkInt8]:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)
    
    return resampler.Execute(image)

def extract_roi(pet_image, target_size=TARGET_SIZE):
    """Extract ROI using bottom-up approach."""
    pet_array = sitk.GetArrayFromImage(pet_image)
    
    # Get the last 25% of slices (neck region)
    crop_len = int(0.75 * pet_array.shape[0])
    neck_region = pet_array[crop_len:, :, :]
    
    # Find center of activity in neck region
    threshold = np.percentile(neck_region, 95)
    binary = neck_region > threshold
    if not np.any(binary):
        raise ValueError("No significant activity found in neck region")
    
    # Get center coordinates
    coords = np.array(np.nonzero(binary))
    center = np.mean(coords, axis=1).astype(int)
    center[0] += crop_len  # Adjust Z coordinate
    
    # Calculate ROI bounds (z,y,x)
    z_start = max(0, center[0] - target_size[2]//2)
    z_end = min(pet_array.shape[0], z_start + target_size[2])
    if z_end == pet_array.shape[0]:
        z_start = max(0, z_end - target_size[2])
    if z_start == 0:
        z_end = min(pet_array.shape[0], target_size[2])
    
    y_start = max(0, center[1] - target_size[1]//2)
    y_end = min(pet_array.shape[1], y_start + target_size[1])
    if y_end == pet_array.shape[1]:
        y_start = max(0, y_end - target_size[1])
    if y_start == 0:
        y_end = min(pet_array.shape[1], target_size[1])
    
    x_start = max(0, center[2] - target_size[0]//2)
    x_end = min(pet_array.shape[2], x_start + target_size[0])
    if x_end == pet_array.shape[2]:
        x_start = max(0, x_end - target_size[0])
    if x_start == 0:
        x_end = min(pet_array.shape[2], target_size[0])
    
    # Convert to SimpleITK order (x,y,z)
    sitk_start = [int(x_start), int(y_start), int(z_start)]
    sitk_size = [
        int(x_end - x_start),
        int(y_end - y_start),
        int(z_end - z_start)
    ]
    
    return sitk_start, sitk_size

def crop_to_size(image, start_idx, size, target_size=TARGET_SIZE):
    """Crop image to target size with padding if needed."""
    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetIndex(start_idx)
    roi_filter.SetSize(size)
    
    cropped = roi_filter.Execute(image)
    
    # Pad if needed
    current_size = cropped.GetSize()
    if current_size != target_size:
        pad_filter = sitk.ConstantPadImageFilter()
        pad_lower = [0, 0, 0]
        pad_upper = [max(0, target_size[i] - current_size[i]) for i in range(3)]
        
        pad_filter.SetPadLowerBound(pad_lower)
        pad_filter.SetPadUpperBound(pad_upper)
        pad_filter.SetConstant(0)
        
        cropped = pad_filter.Execute(cropped)
    
    return cropped

def process_case(pet_path, ct_path, mask_path, output_dir):
    """Process a single case."""
    try:
        # Load images
        pet_image = sitk.ReadImage(str(pet_path))
        ct_image = sitk.ReadImage(str(ct_path))
        mask_image = sitk.ReadImage(str(mask_path))
        
        # Calculate physical size based on CT
        ct_physical_size = np.array([
            size * spacing 
            for size, spacing in zip(ct_image.GetSize(), ct_image.GetSpacing())
        ])
        
        # Calculate new size to maintain physical size
        new_size = np.ceil(ct_physical_size / np.array(TARGET_SPACING)).astype(int)
        
        # Resample all images
        pet_resampled = resample_image(pet_image, new_size=new_size)
        ct_resampled = resample_image(ct_image, new_size=new_size)
        mask_resampled = resample_image(mask_image, new_size=new_size)
        
        # Extract ROI
        start_idx, size = extract_roi(pet_resampled)
        
        # Crop all images
        pet_cropped = crop_to_size(pet_resampled, start_idx, size)
        ct_cropped = crop_to_size(ct_resampled, start_idx, size)
        mask_cropped = crop_to_size(mask_resampled, start_idx, size)
        
        # Save results
        case_id = pet_path.stem.split('__')[0]
        sitk.WriteImage(pet_cropped, str(output_dir / f"{case_id}_pt_roi.nii.gz"))
        # sitk.WriteImage(ct_cropped, str(output_dir / f"{case_id}_ct_roi.nii.gz"))
        # sitk.WriteImage(mask_cropped, str(output_dir / f"{case_id}_mask_roi.nii.gz"))
        
        return {
            'case_id': case_id,
            'final_size': pet_cropped.GetSize(),
            'final_spacing': pet_cropped.GetSpacing(),
            'roi_start': start_idx,
            'roi_size': size,
            'status': 'success'
        }
    except Exception as e:
        return {
            'case_id': pet_path.stem.split('__')[0],
            'error': str(e),
            'status': 'failed'
        }

def main():
    # Setup paths
    base_path = Path("/share/sda/mohammadqazi/project/hector/dataset")
    image_path = base_path / "imagesTr"
    label_path = base_path / "labelsTr"
    output_dir = base_path / "processed_samples_all"
    output_dir.mkdir(exist_ok=True)
    
    # Get all PET files
    pet_files = sorted(list(image_path.glob("*__PT.nii.gz")))
    
    # Group by center
    centers = {}
    for pet_file in pet_files:
        center = pet_file.stem.split('-')[0]
        if center not in centers:
            centers[center] = []
        centers[center].append(pet_file)
    
    # Select 3 random samples from each center
    selected_files = []
    for center, files in centers.items():
        print(f"\nCenter {center}: {len(files)} files available")
        # Process all files instead of just 3 samples
        selected_files.extend(files)
        print(f"Selected all {len(files)} files")
    
    # Process selected cases
    results = []
    print(f"\nProcessing {len(selected_files)} samples ({len(centers)} centers)...")
    
    for pet_file in tqdm(selected_files):
        case_id = pet_file.stem.split('__')[0]
        ct_file = image_path / f"{case_id}__CT.nii.gz"
        mask_file = label_path / f"{case_id}_label.nii.gz"
        
        if not ct_file.exists():
            print(f"\nSkipping {case_id}: Missing CT file")
            continue
            
        if not mask_file.exists():
            print(f"\nSkipping {case_id}: Missing mask file")
            continue
        
        print(f"\nProcessing {case_id}...")
        result = process_case(pet_file, ct_file, mask_file, output_dir)
        results.append(result)
        
        if result['status'] == 'success':
            print(f"Success - Final size: {result['final_size']}")
        else:
            print(f"Failed: {result['error']}")
    
    # Save processing log
    with open(output_dir / 'processing_log.txt', 'w') as f:
        f.write("Processing Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total cases processed: {len(results)}\n\n")
        
        successful = sum(1 for r in results if r['status'] == 'success')
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {len(results) - successful}\n\n")
        
        f.write("Details by case:\n")
        f.write("-"*50 + "\n")
        for result in results:
            f.write(f"\nCase: {result['case_id']}\n")
            f.write(f"Status: {result['status']}\n")
            if result['status'] == 'success':
                f.write(f"Final size: {result['final_size']}\n")
                f.write(f"Final spacing: {result['final_spacing']}\n")
                f.write(f"ROI start: {result['roi_start']}\n")
                f.write(f"ROI size: {result['roi_size']}\n")
            else:
                f.write(f"Error: {result['error']}\n")

if __name__ == "__main__":
    main()