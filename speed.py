import sys

def main():
    # 检查参数数量是否正确（4个参数 + 脚本名 = 5）
    if len(sys.argv) != 5:
        print("Usage: python a.py <df-time> <df-glops> <tf-time> <tf-glops>")
        sys.exit(1)

    try:
        a = float(sys.argv[1])
        b = float(sys.argv[2])
        c = float(sys.argv[3])
        d = float(sys.argv[4])
    except ValueError:
        print("Error: All arguments must be numbers")
        sys.exit(1)

    # 计算并输出结果
    print(f"[CARD] 4, [Dataset]: DriveSeg")
    print(f'DataFlow E2E Time: {a:.4f} s, GFLOPS: {b:.4f} GFLOPS')
    print(f"TensorFlow E2E Time: {c:.4f} s, GFLOPS: {d:.4f} GFLOPS")
    print(f"E2E TIME SPEED UP: {c/a:.2f}")
    print(f"E2E GFLOPS SPEED UP: {b/d:.2f}")

if __name__ == "__main__":
    main()