import random
import argparse
import os


def generatePointCloudInt(filepath, n, k, T):
    seen = set()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        f.write(f"{k}\n")
        f.write(f"{T}\n")

        while len(seen) < n:
            x = int(random.randint(-10000, 10000))
            y = int(random.randint(-10000, 10000))
            z = int(random.randint(-10000, 10000))
            intensity = int(random.randint(0, 255))
            point = (x, y, z, intensity)
            if point in seen:
                continue
            seen.add(point)
            f.write(f"{x} {y} {z} {intensity}\n")


def generatePointCloudFloat(filepath, n, k, T):
    seen = set()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        f.write(f"{k}\n")
        f.write(f"{T}\n")

        while len(seen) < n:
            x = random.uniform(-10000.0, 10000.0)
            y = random.uniform(-10000.0, 10000.0)
            z = random.uniform(-10000.0, 10000.0)
            intensity = int(random.randint(0, 255))
            point = (round(x, 4), round(y, 4), round(z, 4), intensity)
            if point in seen:
                continue
            seen.add(point)
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {intensity}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic point cloud data")
    parser.add_argument("--n", type=int, default=1_000_000, help="Number of points")
    parser.add_argument("--k", type=int, default=16, help="KNN parameter")
    parser.add_argument("--T", type=int, default=50, help="Iterations")

    args = parser.parse_args()

    filename_int = f"input/int/dataset_n_{args.n}_k_{args.k}_T_{args.T}.txt"
    filename_float = f"input/float/dataset_n_{args.n}_k_{args.k}_T_{args.T}.txt"

    print(f"Generating {args.n} points (k={args.k}, T={args.T})...")
    print(f"Creating '{filename_int}'...")
    generatePointCloudInt(filename_int, args.n, args.k, args.T)
    print(f"Creating '{filename_float}'...")
    generatePointCloudFloat(filename_float, args.n, args.k, args.T)
