import random
import os

def generate_matrix_file(n: int, filename: str, value_range=(-5.0, 5.0), dominance_factor=1.5):
    """
    Generates a random non-singular well-conditioned matrix
    :param n: matrix dimension
    :param filename: output file
    :param value_range: range of element values
    :param dominance_factor: diagonal dominance factor (to ensure matrix is well-conditioned)
    :return: None
    """
    A = []
    for i in range (n):
        row = [random.uniform(*value_range) for _ in range(n)]
        off_diag_sum = sum(abs(x) for j, x in enumerate(row) if j != i)
        row[i] = (off_diag_sum + 1.0) * dominance_factor * (1 if random.random() > 0.5 else -1)
        A.append(row)

    dirpath = os.path.dirname(os.path.abspath(filename))
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

    with open(filename, "w") as f:
        f.write(f"{n}\n")
        for row in A:
            f.write(" ".join(f"{x:.6f}" for x in row) + "\n")
        f.close()


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python generate_matrix.py n filename")
        sys.exit(1)
    try:
        n = int(sys.argv[1])
    except ValueError:
        print("Usage: python generate_matrix.py n filename")
        sys.exit(1)
    filename = sys.argv[2]
    generate_matrix_file(n, filename)
