import json

solvers = ["weighted", "squared-weighted", "ridge", "ridge-closed-form"]
print(f"{'solver':<22}{'cos_sim':>10}{'cosine_median':>16}{'angle_deg_mean':>16}{'mse':>12}")
print("-" * 76)
for s in solvers:
    with open(f"artifacts/garden/roundtrip/eval_{s}.json") as f:
        m = json.load(f)
    print(f"{s:<22}{m['cos_sim']:>10.4f}{m['cosine_median']:>16.4f}{m['angular_error_deg_mean']:>16.2f}{m['mse']:>12.6f}")
