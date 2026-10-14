import argparse
import json
import os
import sys
import rasterio

# Add the project root to the python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predict import run_prediction
from visualization.visualize import generate_overlay

def main():
    parser = argparse.ArgumentParser(description="Deep Learning Flood Detection & Area Estimation")
    parser.add_argument("--image", type=str, required=True, help="Path to input GeoTIFF satellite image.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model weights (.pth).")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save outputs.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"[*] Running prediction on {args.image}...")
    results = run_prediction(args.image, args.model, device="cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n--- OUTPUT EXAMPLE ---")
    print(f"Flood detected: {'YES' if results['flood_pixels'] > 0 else 'NO'}")
    # Note: confidence could be average probability of flood pixels
    conf = float(results['prob_mask'][results['binary_mask'] == 1].mean()) * 100 if results['flood_pixels'] > 0 else 0
    print(f"Flood confidence: {conf:.1f}%")
    print(f"Flood pixels: {results['flood_pixels']:,}")
    print(f"Flood area: {results['area_km2']:.4f} km²")
    print(f"Flood percentage: {results['flood_percentage']:.2f}%")
    print("----------------------\n")
    
    # Save outputs
    # 1. Mask TIF
    mask_path = os.path.join(args.output_dir, "predicted_mask.tif")
    prof = results['profile'].copy()
    prof.update(dtype=rasterio.uint8, count=1, compress='lzw')
    with rasterio.open(mask_path, 'w', **prof) as dst:
        dst.write(results['binary_mask'], 1)
        
    # 2. Boundary GeoJSON
    geojson_path = os.path.join(args.output_dir, "flood_boundary.geojson")
    with open(geojson_path, 'w') as f:
        json.dump({
            "type": "FeatureCollection",
            "features": results['geojson']
        }, f)
        
    # 3. Prediction JSON
    json_path = os.path.join(args.output_dir, "prediction.json")
    with open(json_path, 'w') as f:
        json.dump({
            "flood_detected": bool(results['flood_pixels'] > 0),
            "confidence_percent": conf,
            "flood_pixels": int(results['flood_pixels']),
            "area_km2": float(results['area_km2']),
            "flood_percentage": float(results['flood_percentage'])
        }, f, indent=4)
        
    # 4. Result Overlay PNG
    png_path = os.path.join(args.output_dir, "result_overlay.png")
    generate_overlay(
        results['original_image'],
        results['binary_mask'],
        results['boundaries'],
        results,
        output_path=png_path
    )
    
    print(f"[*] All outputs saved to {args.output_dir}/")

if __name__ == "__main__":
    import torch # Imported here for the device check
    main()
