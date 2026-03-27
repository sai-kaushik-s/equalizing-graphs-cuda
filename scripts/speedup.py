import csv
import sys

def parse_time(time_str):
    if not time_str or time_str == '-':
        return None
    time_str = time_str.strip('"').strip()
    try:
        if 'ms' in time_str:
            return float(time_str.replace('ms', '').strip()) / 1000.0
        elif 'µs' in time_str or 'us' in time_str:
            return float(time_str.replace('µs', '').replace('us', '').strip()) / 1000000.0
        elif 'ns' in time_str:
            return float(time_str.replace('ns', '').strip()) / 1000000000.0
        elif 's' in time_str:
            return float(time_str.replace('s', '').strip())
        else:
            return float(time_str)
    except ValueError:
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/speedup.py <input_csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    
    rows = []
    knn_baselines = {}

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            n = row['n']
            k = row['k']
            algo = row['algo']
            precision = row['precision']
            gpu_time_str = row['gpu_time']
            cpu_time_str = row['cpu_time']

            gpu_time = parse_time(gpu_time_str)
            cpu_time = parse_time(cpu_time_str)

            row['gpu_time_float'] = gpu_time
            row['cpu_time_float'] = cpu_time

            if algo == 'knn':
                knn_baselines[(n, k, precision)] = gpu_time

            rows.append(row)

    new_fieldnames = fieldnames + ['gpu speedup', 'algo gpu speedup']
    
    with open(input_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()

        for row in rows:
            n = row['n']
            k = row['k']
            precision = row['precision']
            gpu_time = row['gpu_time_float']
            cpu_time = row['cpu_time_float']

            cpu_vs_gpu = ""
            if cpu_time is not None and gpu_time is not None and gpu_time > 0:
                cpu_vs_gpu = f"{cpu_time / gpu_time:.4f}"
            row['gpu speedup'] = cpu_vs_gpu

            algo_speedup = "-"
            if row['algo'] != 'knn':
                baseline = knn_baselines.get((n, k, precision))
                if baseline is not None and gpu_time is not None and gpu_time > 0:
                    algo_speedup = f"{baseline / gpu_time:.4f}"
            row['algo gpu speedup'] = algo_speedup

            del row['gpu_time_float']
            del row['cpu_time_float']
            
            writer.writerow(row)

    print(f"Speedup analysis added to {input_file}")

if __name__ == "__main__":
    main()
