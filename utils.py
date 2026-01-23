import os
import numpy as np
import torch
import timm

from PIL import Image
from dotenv import load_dotenv


class Layout_Classifier:
    """
    Class for running layout classification on document images using a pre-trained Swin Transformer model.
    """

    def __init__(self, model_path):
        """
        Load the Torch model and initialize transforms.

        Args:
            model_path (str): Path to the Torch model file.
        """
        self.model = torch.jit.load(model_path)
        self.model.cuda()
        self.model.eval()

        data_config = timm.data.resolve_model_data_config(self.model)
        data_config['input_size'] = (3, 384, 384)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

        self.label_names = ['bullet_points', 'decision', 'margins_left', 'margins_right', 
               'register', 'running_text', 'session', 'supplement', 
               'tabular_classic', 'tabular_exotic', 'tabular_multi', 'tabular_text']

    def run_classifier(self, image_path):
        """
        Classify the layout of the given image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            list[str]: Predicted labels.
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transforms(img).unsqueeze(0).cuda()

        # Model inference
        outputs = self.model(img_tensor)
        probs_tensor = torch.sigmoid(outputs)[0]
        threshold_preds = (probs_tensor > 0.5).float().cpu().tolist()
        probs = probs_tensor.cpu().tolist()

        # Map predictions to label names
        labels = [name for idx, name in enumerate(self.label_names) if threshold_preds[idx] == 1]

        # Ensure at least one main type label is present
        main_types = ['session', 'decision', 'supplement', 'register']
        if not any(t in labels for t in main_types):
            main_probs = [probs[1], probs[4], probs[6], probs[7]]
            max_index = np.argmax(main_probs)
            labels.append(main_types[max_index])

        # Margin logic for session/decision
        if 'session' in labels or 'decision' in labels:
            left_present = 'margins_left' in labels
            right_present = 'margins_right' in labels
            left_prob = probs[2]
            right_prob = probs[3]
            if left_present and right_present:
                # Keep only the margin with higher probability
                if left_prob > right_prob:
                    labels.remove('margins_right')
                else:
                    labels.remove('margins_left')
            elif not left_present and not right_present:
                # Add the margin with higher probability
                if left_prob > right_prob:
                    labels.append('margins_left')
                else:
                    labels.append('margins_right')

        if len(labels) == 0:
            raise Exception(
                f"No labels found for {image_path}. Retrain the classifier with your documents to learn the layout first."
            )

        return labels

def clean_markdown_response(text):
    """
    Clean up Markdown output from Gemini API by removing code block markers and unwanted tags.

    Args:
        text (str): Raw Markdown output.

    Returns:
        str: Cleaned Markdown output.
    """
    
    replacements = replacements = {'```': '','markdown': '','Ã¼': 'ü','Ã¤': 'ä','Ã¶': 'ö'}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def convert_tiff_to_jpg_recursive(input_folder, is_eval, remove_files=False, max_size_kb=5000, quality=95,):
    """
    Recursively convert TIFF files in a folder to JPG format.

    Args:
        input_folder (str): Path to folder containing TIFF files.
        is_eval (bool): If True, process all files regardless of previous results.
        remove_files (bool): If True, remove original TIFF files after conversion.
        max_size_kb (int): Target maximum file size in KB.
        quality (int): JPEG quality (0-95).

    Returns:
        list[str]: List of processed JPG file paths.
    """
    # Supported TIFF extensions
    tiff_extensions = ('.tiff', '.tif', '.jpg')
    print("----------------- DATA PREPROCESSING -----------------")
    # Counter for processed files and paths to files to process downstream
    processed = 0
    errors = 0
    files_to_process = []
    try:
        # Walk through directory tree
        for root, _, files in os.walk(input_folder):
            for filename in files:
                if filename.lower().endswith(tiff_extensions):
                    input_path = os.path.join(root, filename)
                    # Replace extension with .jpg
                    output_filename = os.path.splitext(filename)[0] + '.jpg'
                    output_path = os.path.join(root, output_filename)

                    finalized_md_file_name = os.path.splitext(filename)[0] + '_final.md'
                    finalized_md_file_name_path = os.path.join(root, finalized_md_file_name)

                    if os.path.exists(finalized_md_file_name_path) and not is_eval:
                        # Skip already parsed files, nunless we are evaluating
                        continue

                    if os.path.getsize(input_path) / 1024 < max_size_kb and input_path.endswith(".jpg"):
                        # Skip if jpg already of suitable size
                        files_to_process.append(output_path)
                        continue

                    # Process individual file
                    try:
                        with Image.open(input_path) as img:
                            # Convert to RGB if needed
                            if img.mode in ('RGBA', 'LA', 'P'):
                                img = img.convert('RGB')
                            
                            # Get original dimensions
                            width, height = img.size
                            
                            # Initial save
                            img.save(output_path, 'JPEG', quality=quality)
                            
                            # Check file size
                            file_size = os.path.getsize(output_path) / 1024
                            
                            # Reduce size if necessary
                            while file_size > max_size_kb and width > 100 and height > 100:
                                width = int(width * 0.9)
                                height = int(height * 0.9)
                                resized_img = img.resize((width, height), Image.Resampling.LANCZOS)
                                resized_img.save(output_path, 'JPEG', quality=quality)
                                file_size = os.path.getsize(output_path) / 1024
                            
                            processed += 1
                            print(f"Converted: {input_path} -> {file_size:.2f} KB")

                            # Delete .tiff
                            files_to_process.append(output_path)
                            if remove_files and not input_path.endswith('.jpg'): 
                                os.remove(input_path)
                    
                    except Exception as e:
                        errors += 1
                        print(f"Error processing {input_path}: {str(e)}")

        # Summary
        if processed == 0:
            print("Conversion call had no effect, no files to convert.")
        else:
            print(f"Successfully preprocessed: {processed} files")
        print("\n")
    except Exception as e:
        print(f"Critical error: {str(e)}")
    
    return files_to_process

class EnvLoader:
    """
    Helper class for loading environment variables from .env file.
    """
    def __init__(self):
        """
        Load environment variables from .env file.
        """
        load_dotenv(".env")
        self.env_vars = {
            'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY'),
        }

    def get_env_variable(self, key):
        """
        Get the value of an environment variable.

        Args:
            key (str): Environment variable name.

        Returns:
            str or None: Value of the environment variable, or None if not found.
        """
        return self.env_vars.get(key, None)

    def print_env_variables(self):
        """
        Print all loaded environment variables.
        """
        for key, value in self.env_vars.items():
            print(f"{key}: {value}")
