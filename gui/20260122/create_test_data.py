"""
Generate test XYZ data and Anomaly Maps
Simulates OLED panel measurement data
"""

import numpy as np
from pathlib import Path


def create_xyz_data(width=512, height=512):
    """
    Create synthetic XYZ image data
    
    Parameters:
    -----------
    width, height : int
        Image dimensions
        
    Returns:
    --------
    xyz_image : ndarray, shape (height, width, 3)
        XYZ color space data (0-255 range)
    """
    # Y channel (luminance) - bright center, darker edges
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    # Luminance decreases from center to edges
    Y_channel = 255 * (1 - distance / max_dist)
    
    # Add noise (like real measurement data)
    noise = np.random.normal(0, 5, (height, width))
    Y_channel = np.clip(Y_channel + noise, 0, 255).astype(np.uint8)
    
    # X, Z channels (color information) - slight variations
    X_channel = np.random.randint(100, 200, (height, width), dtype=np.uint8)
    Z_channel = np.random.randint(80, 180, (height, width), dtype=np.uint8)
    
    # Compose XYZ image (height, width, 3)
    xyz_image = np.stack([X_channel, Y_channel, Z_channel], axis=2)
    
    return xyz_image


def create_anomaly_map(width=512, height=512, num_defects=5):
    """
    Create synthetic Anomaly Score Map
    
    Parameters:
    -----------
    width, height : int
        Image dimensions
    num_defects : int
        Number of defects
        
    Returns:
    --------
    anomaly_map : ndarray, shape (height, width)
        Anomaly score (0.0~1.0 range)
    """
    # Background noise (low anomaly score)
    anomaly_map = np.random.rand(height, width) * 0.3
    
    # Add defect regions (high anomaly score)
    for _ in range(num_defects):
        cx = np.random.randint(50, width - 50)
        cy = np.random.randint(50, height - 50)
        radius = np.random.randint(10, 30)
        
        # Circular defect mask
        y, x = np.ogrid[:height, :width]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        
        # Assign high anomaly score to defect region
        anomaly_map[mask] = np.random.rand() * 0.5 + 0.5  # 0.5~1.0
    
    return anomaly_map.astype(np.float32)


def main():
    """Main function for test data generation"""
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("Test Data Generation")
    print("="*60 + "\n")
    
    # Generate multiple sample files
    num_samples = 5
    
    for i in range(1, num_samples + 1):
        print(f"[{i}/{num_samples}] Generating data...", end=" ")
        
        # Generate XYZ data
        xyz = create_xyz_data()
        
        # Include anomaly map only for even numbers
        if i % 2 == 0:
            anomaly_map = create_anomaly_map()
            
            np.savez_compressed(
                data_dir / f"sample_{i:03d}.npz",
                xyz=xyz,
                anomaly_map=anomaly_map,
                metadata=np.array({
                    'device': f'OLED_Panel_{i}',
                    'date': '2024-01-15',
                    'resolution': xyz.shape[:2],
                    'has_anomaly': True
                }, dtype=object)
            )
            print(f"[OK] sample_{i:03d}.npz (XYZ + Anomaly Map)")
        else:
            np.savez_compressed(
                data_dir / f"sample_{i:03d}.npz",
                xyz=xyz,
                metadata=np.array({
                    'device': f'OLED_Panel_{i}',
                    'date': '2024-01-15',
                    'resolution': xyz.shape[:2],
                    'has_anomaly': False
                }, dtype=object)
            )
            print(f"[OK] sample_{i:03d}.npz (XYZ only)")
    
    print(f"\n{'='*60}")
    print(f"Total {num_samples} test files created in '{data_dir}' folder")
    print("="*60 + "\n")
    
    # Load and test one file
    print("File Loading Test:")
    print("-" * 60)
    
    test_file = data_dir / "sample_001.npz"
    data = np.load(test_file, allow_pickle=True)
    
    print(f"Filename: {test_file.name}")
    print(f"Keys: {list(data.keys())}")
    print(f"XYZ shape: {data['xyz'].shape}")
    print(f"XYZ dtype: {data['xyz'].dtype}")
    print(f"XYZ value range: {data['xyz'].min()} ~ {data['xyz'].max()}")
    
    if 'anomaly_map' in data:
        print(f"Anomaly Map shape: {data['anomaly_map'].shape}")
        print(f"Anomaly Map range: {data['anomaly_map'].min():.3f} ~ {data['anomaly_map'].max():.3f}")
    
    print("-" * 60 + "\n")
    
    print("Next step:")
    print("  python main.py  # Run GUI application")
    print()


if __name__ == "__main__":
    main()
