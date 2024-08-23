import argparse

from torchreid.scripts.builder import build_config, build_torchreid_model_engine
from torchreid.tools.extract_part_based_features import extract_reid_features
from torchreid.scripts.default_config import engine_run_kwargs

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "-s",
        "--sources",
        type=str,
        nargs="+",
        help="source datasets (delimited by space)",
    )
    parser.add_argument(
        "-t",
        "--targets",
        type=str,
        nargs="+",
        help="target datasets (delimited by space)",
    )
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation")
    parser.add_argument("--root", type=str, default="", help="path to data root")
    parser.add_argument(
        "--save_dir", type=str, default="", help="path to output root dir"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )
    parser.add_argument("--job-id", type=int, default=None, help="Slurm job id")
    parser.add_argument(
        "--inference-enabled", type=bool, default=False,
    )
    args = parser.parse_args()

    cfg = build_config(args=args, config_path=args.config_file, display_diff=True)

    engine, model = build_torchreid_model_engine(cfg)
    if not cfg.inference.enabled:
        print(
            "Starting experiment {} with job id {} and creation date {}".format(
                cfg.project.experiment_id, cfg.project.job_id, cfg.project.start_time
            )
        )
        engine.run(**engine_run_kwargs(cfg))
        print(
            "End of experiment {} with job id {} and creation date {}".format(
                cfg.project.experiment_id, cfg.project.job_id, cfg.project.start_time
            )
        )
    else:
        print("Starting inference on external data")
        extract_reid_features(cfg, cfg.inference.input_folder, cfg.data.save_dir, model)

if __name__ == "__main__":
    main()
