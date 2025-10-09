import time

from pathlib import Path

from google import genai
from google.genai import types

from concurrent.futures import ThreadPoolExecutor
from few_shot.few_shot_metadata import get_few_shot_samples

from prompts import SYSTEM_PROMPTS
from utils import clean_markdown_response, convert_tiff_to_jpg_recursive, EnvLoader, Layout_Classifier

class DocumentParser:
    """
    Main class for processing historical document images and generating refined Markdown files.
    Handles HTR, layout classification, zero-shot learning, few-shot learning, and post-processing.
    """
    def __init__(self,):
        """
        Initialize the DocumentParser with environment variables, model paths, and API clients.
        """
        self.env = EnvLoader()
        self.base_dir = Path(__file__).parent
        gemini_api_key = self.env.get_env_variable("GEMINI_API_KEY")
        self.few_shot_path = self.env.get_env_variable("FEW_SHOT_FOLDER_PATH")

        self.gemini_client = genai.Client(api_key=gemini_api_key)
        self.layout_classifier = Layout_Classifier(str(self.base_dir / "models" / "layout_classifier_model.pt"))
        self.folder_path = str(self.base_dir / "data")

    def set_folder_path(self, folder_path: str):
        """
        Set the folder path containing images to process.

        Args:
            folder_path (str): Path to the folder with images.
        """
        self.folder_path = folder_path
        
    def parse_files(self, batch_size=10, is_eval=False):
        """
        Run the full document processing pipeline on all images in the folder.

        Args:
            batch_size (int): Number of images to process in each batch.
            is_eval (bool): If True, process all files regardless of previous results.

        Returns:
            dict: Contains 'markdown_files' and 'markdown_names' lists.
        """
        start_time = time.time()
        all_images = convert_tiff_to_jpg_recursive(self.folder_path, is_eval, remove_files=True)
        print("----------------- RUNNING PIPELINE -----------------")
        if not all_images:
            print("Warning: Pipeline run aborted! No .tif files to process.")
            return None

        num_images = len(all_images)
        results = [None] * num_images

        def process_image(image_path, idx):
            """
            Run the pipeline for a single image and select the best markdown output.
            Executes three parallel runs and selects the best result.

            Args:
                image_path (str): Path to the image file.
                idx (int): Index of the image in the batch.

            Returns:
                tuple: (index, selected_markdown or None)
            """
            try:
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [executor.submit(self._run_pipeline, image_path) for _ in range(3)]
                    pipeline_results = [future.result() for future in futures]
                prompt = [
                    "The first markdown file: ", pipeline_results[0],
                    "The second markdown file: ", pipeline_results[1],
                    "The third markdown file: ", pipeline_results[2]
                ]
                selected = self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPTS['sample_selection'],
                        thinking_config=types.ThinkingConfig(thinking_budget=512),
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)),
                    contents=[prompt]
                ).text

                replacements = {'```': '','markdown': '','Ã¼': 'ü','Ã¤': 'ä','Ã¶': 'ö'}
                for old, new in replacements.items(): selected = selected.replace(old, new)

                md_path = image_path.replace(".jpg", "_final.md")
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(selected)
                return idx, selected
            
            except Exception as e:
                print(f"Pipeline failure for input {image_path}: {e}.")
                return idx, None

        def batch_process(images, start_idx):
            """
            Process a batch of images, handling retries for failures.

            Args:
                images (list): List of image paths to process.
                start_idx (int): Starting index for the batch.
            """
            batch_len = len(images)
            print(f"\nProcessing a total of {num_images} documents")
            print('#' * 90)
            print(f"Working on the next {batch_len} documents, {start_idx / num_images * 100:.2f}% completed")
            print('#' * 90 + "\n")
            failed = []
            with ThreadPoolExecutor(max_workers=batch_len * 3) as executor:
                futures = {executor.submit(process_image, img, i + start_idx): i + start_idx for i, img in enumerate(images)}
                for future in futures:
                    idx, result = future.result()
                    if result is not None:
                        results[idx] = result
                    else:
                        failed.append((idx, images[idx - start_idx]))
                # Retry logic
                while failed:
                    retry_imgs = [img for _, img in failed]
                    retry_idxs = [i for i, _ in failed]
                    print(f"Retrying {len(retry_imgs)} failed images...")
                    new_failed = []
                    new_futures = {executor.submit(process_image, img, idx): idx for idx, img in zip(retry_idxs, retry_imgs)}
                    for future in new_futures:
                        idx, result = future.result()
                        if result is not None:
                            results[idx] = result
                        else:
                            new_failed.append((idx, retry_imgs[retry_idxs.index(idx)]))
                    if len(new_failed) == len(failed) and set(new_failed) == set(failed):
                        print(f"No progress made after retrying {len(failed)} images. Stopping retries.")
                        break
                    failed = new_failed

        idx = 0
        while idx < num_images:
            batch = all_images[idx:idx + batch_size]
            batch_process(batch, idx)
            idx += batch_size

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_images if num_images > 0 else 0
        print(f"Transformed {num_images} historical documents in an average of {avg_time:.2f} seconds.")
        return {
            "markdown_files": results,
            "markdown_names": [Path(path).stem + ".md" for path in all_images]
        }

    def _run_pipeline(self, image_path: str):
        """
        Run the full pipeline for a single image: zero-shot, classification, few-shot, post-processing.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Final Markdown output for the image.
        """

        with ThreadPoolExecutor(max_workers=2) as executor:
            
            # Zero-Shot Inference & LDDC Inference
            prediction_futures = executor.submit(self._zero_shot_inference, image_path)
            classification_future = executor.submit(self._classify, image_path)
            
            prediction = prediction_futures.result()
            predicted_labels = classification_future.result() # A label, representing a file class
        
        # Few-Shot Learning
        few_shot_paths = self._greedy_label_matching(predicted_labels, image_path) # Greedily select FS GT samples
        markdown = self._few_shot_inference(prediction, few_shot_paths, image_path)
        
        # Post-Processing
        markdown = self._post_processing(markdown, predicted_labels, image_path)
        return markdown

    def _zero_shot_inference(self, image_path: str):
        """
        Perform zero-shot inference on the image using the Gemini model.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Initial Markdown output.
        """
        return self.gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPTS['sampling_system_prompt'],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)),
            contents=[self._image_to_bytes(image_path)]
        ).text
    
    def _post_processing(self, markdown: str, predicted_labels: list[str], image_path: str):
        """
        Refine the Markdown output based on predicted labels (e.g., margins, registers).

        Args:
            markdown (str): Initial Markdown output.
            predicted_labels (list[str]): List of predicted labels for the image.
            image_path (str): Path to the image file.

        Returns:
            str: Refined Markdown output.
        """

        margins = ['margins_left', 'margins_right']
        registers = ['tabular_multi', 'tabular_classic']

        margin = set(margins) & set(predicted_labels)
        register = set(registers) & set(predicted_labels)

        if margin:

            side = 'left' if list(margin)[0] == 'margins_left' else 'right'
            system_prompt = SYSTEM_PROMPTS['margin_post_processing'].format(side=side)
            prompt = ["The original image: ", self._image_to_bytes(image_path), "The markdown file to refine: ", markdown]

            response = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)),
                contents=prompt
            )

            return clean_markdown_response(response.text)
        
        if register:

            system_prompt = SYSTEM_PROMPTS['register_post_processing']
            prompt = ["The original image: ", self._image_to_bytes(image_path), "The markdown file to refine: ", markdown]

            response = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)),
                contents=prompt
            )

            return clean_markdown_response(response.text)

        return markdown

    def _greedy_label_matching(self, predicted_labels, image_path):
        """
        Match predicted labels to few-shot ground truth samples.

        Args:
            predicted_labels (list[str]): Predicted labels for the image.
            image_path (str): Path to the image file.

        Returns:
            list: List of tuples with ground truth image and markdown paths.
        """
        fs_samples = get_few_shot_samples()
        predicted_set = set(predicted_labels)
        n = len(predicted_set)
        
        # Compute scores: (score, index)
        scores = []
        for idx, sample in enumerate(fs_samples):
            gt_set = set(sample['labels'])
            m = len(gt_set)
            s = len(predicted_set & gt_set)
            score = (s/n + s/m)/2 if n and m else 0
            scores.append((score, idx))

        # Notify if a perfect match is missing in GT for Few-Shot Learning
        scores_test, _ = zip(*scores)
        if not 1.0 in scores_test:
            print("No perfect match for predicted labels: ", predicted_labels)
            print("Image path: ", image_path)
        
        # Get indices of max score
        max_score = max(s[0] for s in scores)
        match_indices = [s[1] for s in scores if s[0] == max_score]
        match_indices = [match_indices[0]]

        # Build paths
        few_shot_paths = []
        base_dir = Path(__file__).parent.resolve()
        for idx in match_indices:
            base_name = fs_samples[idx]['file_path']
            jpg_path = base_dir / 'few_shot' / f'{base_name}.jpg'
            md_path = base_dir / 'few_shot' / f'{base_name}.md'
            few_shot_paths.extend([(jpg_path.absolute(), md_path.absolute())])
        
        return few_shot_paths

    def _classify(self, image_path: str):
        """
        Classify the layout of the image using the layout classifier.

        Args:
            image_path (str): Path to the image file.

        Returns:
            list[str]: Predicted labels.
        """
        return self.layout_classifier.run_classifier(image_path=image_path)
    
    def _few_shot_inference(self, predictions: str, few_shot_paths: list[str], image_path: str):
        """
        Refine predictions using few-shot ground truth samples.

        Args:
            predictions (str): Initial Markdown prediction.
            few_shot_paths (list[str]): List of ground truth sample paths.
            image_path (str): Path to the image file.

        Returns:
            str: Refined Markdown output.
        """
        # Sanity check, abort if no gt samples
        if not few_shot_paths:
            print("Warning: Pipeline run aborted! Apparently not a single Ground Truth file available for Few-Shot Learning")
            print("Pipeline failed for: ", image_path)
            return None

        G = len(few_shot_paths)
        S = len(predictions)

        prompt = []
        prompt.append("The following ground truth files are your guidance on a structural level, adapt to them layout wise, do not copy words from ground truth!")

        for idx, (img_path, md_path) in enumerate(few_shot_paths):
            gt_img = self._image_to_bytes(img_path)
            with open(md_path, 'r', encoding="utf-8") as f:
                gt_md = f.read()
            prompt.append(f"ground_truth_pair{idx}")
            prompt.append(gt_img)
            prompt.append(gt_md)

        prompt.append("The following markdown file: ")
        if isinstance(predictions, (list, tuple)):
            prompt.extend(predictions)
        else:
            prompt.append(predictions)
        prompt.append(" is the predictions of this file: ")
        prompt.append(self._image_to_bytes(image_path))
        prompt.append("apply the structure from the ground truth to the prediction, do not change the prediction's semantics, only the structure.")
        prompt.append("Provide the perfect prediction markdown file as an answer. Do not copy semantics from ground truth!")

        response = self.gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPTS['refinement_system_prompt'].format(G=G, S=S),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)),
            contents=prompt
        )
        
        return clean_markdown_response(response.text)
        
    def _image_to_bytes(self, image_path):
        """
        Convert an image file to bytes for Gemini API input.

        Args:
            image_path (str): Path to the image file.

        Returns:
            types.Part: Gemini API image part.
        """
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        return types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')
    

def main():

    print("Chrono Quill is setting up..")
    parser = DocumentParser()
    parser.parse_files()
        
if __name__ == "__main__":
    main()


    