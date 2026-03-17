import random
import argparse

def generate_point_cloud(filename, n, k, T, num_clusters=5):
    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        f.write(f"{k}\n")
        f.write(f"{T}\n")
        
        centers = [(random.uniform(-100, 100),
                    random.uniform(-100, 100),
                    random.uniform(-100, 100))
                   for _ in range(num_clusters)]
            
        for _ in range(n):
            cx, cy, cz = random.choice(centers)
            
            x = random.gauss(cx, 15.0)
            y = random.gauss(cy, 15.0)
            z = random.gauss(cz, 15.0)
            intensity = random.randint(0, 255)
            
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {intensity}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic point cloud data")

    parser.add_argument("--n", type=int, default=1_000_000, help="Number of points")
    parser.add_argument("--k", type=int, default=16, help="KNN parameter")
    parser.add_argument("--T", type=int, default=50, help="Iterations")

    args = parser.parse_args()

    filename = f"input/dataset_n_{args.n}_k_{args.k}_T_{args.T}.txt"

    print(f"Generating {args.n} points (k={args.k}, T={args.T}) into '{filename}'...")
    generate_point_cloud(filename, args.n, args.k, args.T)