import argparse
import sys
from pathlib import Path

# Add project directory to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.utils import ensure_directories, get_logger
from src.stage1_nlp_mining.geocoder import build_candidate_events
from src.stage2_satellite_verification.verify import verify_candidate_events
from src.stage3_feature_engineering.negative_sampling import build_training_dataset
from src.stage4_geofm_finetune.train import run_stage4
from src.stage5_validation.validate import run_stage5

logger = get_logger("main_pipeline")


def main():
    parser = argparse.ArgumentParser(description="NLP-Mined Inventories + GeoFM for Landslide Mapping in Eastern Ghats")
    parser.add_argument("--stage", type=int, choices=[0, 1, 2, 3, 4, 5], help="Run a specific stage (0 to 5)")
    parser.add_argument("--all", action="store_true", help="Run full pipeline from Stage 0 to Stage 5 end-to-end")

    args = parser.parse_args()

    if not args.stage and not args.all:
        parser.print_help()
        sys.exit(1)

    logger.info("Initializing Araku Landslide Mapping Pipeline...")
    ensure_directories()

    if args.stage == 0 or args.all:
        logger.info("========== STAGE 0: Environment & Directory Setup ==========")
        ensure_directories()
        logger.info("Stage 0 Completed.")

    if args.stage == 1 or args.all:
        logger.info("========== STAGE 1: Multilingual NLP Archival Mining ==========")
        build_candidate_events()

    if args.stage == 2 or args.all:
        logger.info("========== STAGE 2: Multi-Sensor Satellite Scar Verification ==========")
        verify_candidate_events()

    if args.stage == 3 or args.all:
        logger.info("========== STAGE 3: Feature Stack & Negative Sampling ==========")
        build_training_dataset()

    if args.stage == 4 or args.all:
        logger.info("========== STAGE 4: GeoFM Fine-Tuning & RF Baselines ==========")
        run_stage4()

    if args.stage == 5 or args.all:
        logger.info("========== STAGE 5: Validation & Uncertainty Mapping ==========")
        run_stage5()

    logger.info("Pipeline Execution Completed Successfully.")


if __name__ == "__main__":
    main()
