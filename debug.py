from preprocess.preprocessor import*
import albumentations as A


transform = A.Resize(height=224, width=224)
convert_dataset_to_jpg("data/database", "data/database_jpg", transform)

