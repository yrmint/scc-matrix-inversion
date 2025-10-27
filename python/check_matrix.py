import numpy as np

def read_matrix_from_file(filename):
    """
    Reads matrix from file in format:
       n
       a11 a12 ... a1n
       ...
       an1 ... ann
    """
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    n = int(lines[0])
    data = []
    for line in lines[1:]:
        row = list(map(float, line.split()))
        data.append(row)
    A = np.array(data)
    if A.shape != (n, n):
        raise ValueError(f"Wring matrix format in file {filename}: expected {n}x{n}, got {A.shape}")
    return A

def compare_matrices(A_true, A_test, eps=1e-6):
    """Compares two matrices with given eps."""
    diff = np.abs(A_true - A_test)
    max_diff = np.max(diff)
    return max_diff < eps, max_diff

def check_inverse(original_file, inverse_file, eps=1e-6):
    """Checks if the inversion is correct."""
    # read matrices
    A = read_matrix_from_file(original_file)
    A_inv_test = read_matrix_from_file(inverse_file)
    n = A.shape[0]

    # compute inverse matrix
    try:
        A_inv_true = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        print("The matrix is singular, has no inverse.")
        return

    # elementwise comparison
    ok, max_diff = compare_matrices(A_inv_true, A_inv_test, eps)

    # check if A * A_inv_test ~ I
    I_test = A @ A_inv_test
    deviation = np.linalg.norm(I_test - np.eye(n), ord='fro')

    print(f"Maximum elementwise difference: {max_diff:.3e}")
    print(f"Residual norm ||I - A*A_inv||_F = {deviation:.3e}")

    if ok and deviation < eps * np.sqrt(n):
        print("The inverse is correct (within tolerance).")
    else:
        print("The inverse may be inaccurate.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python check_matrix.py original.txt inverse.txt [eps]")
        sys.exit(1)

    original_file = sys.argv[1]
    inverse_file = sys.argv[2]
    eps = float(sys.argv[3]) if len(sys.argv) >= 4 else 1e-6

    check_inverse(original_file, inverse_file, eps)

    # todo: iterate over all subdirectories in ../matrices
