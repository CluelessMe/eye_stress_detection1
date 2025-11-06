import sys
from src import eye_detection, feature_extraction, model_training, real_time_detection

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <module>")
        print("Modules: eye_detection, feature_extraction, model_training, real_time_detection")
        sys.exit(1)

    module = sys.argv[1]
    if module == "eye_detection":
        eye_detection.main()
    elif module == "real_time_detection":
        real_time_detection.real_time_stress_detection()
    else:
        print(f"Module {module} not implemented directly in main.py")
