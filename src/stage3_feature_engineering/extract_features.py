# Alias module for feature extraction
from src.stage3_feature_engineering.feature_stack import compute_dem_derivatives, extract_feature_vector_at_coords
from src.stage3_feature_engineering.negative_sampling import build_training_dataset

__all__ = ['compute_dem_derivatives', 'extract_feature_vector_at_coords', 'build_training_dataset']
