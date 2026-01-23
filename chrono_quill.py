import time

from typing import List, Tuple
from pathlib import Path

from google import genai
from google.genai import types

from concurrent.futures import ThreadPoolExecutor
from few_shot.few_shot_metadata import get_few_shot_samples

from prompts import PROMPTS
from utils import clean_markdown_response, convert_tiff_to_jpg_recursive, EnvLoader, Layout_Classifier

from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class ChronoQuill:
    """
    Class for processing historical document images and generating refined Markdown files.
    Handles HTR, layout classification, zero-shot learning, few-shot learning, and post-processing.
    """
    def __init__(self,):
        """
        Initialize the ChronoQuill with environment variables, model paths, and API clients.
        """
        self.env = EnvLoader()
        self.base_dir = Path(__file__).parent
        gemini_api_key = self.env.get_env_variable("GEMINI_API_KEY")
        self.few_shot_path = self.env.get_env_variable("FEW_SHOT_FOLDER_PATH")

        self.gemini_client = genai.Client(api_key=gemini_api_key)
        self.layout_classifier = Layout_Classifier(str(self.base_dir / "models" / "layout_classifier_model.pt"))
        self.folder_path = str(self.base_dir / "data")
        

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
            predicted_labels = classification_future.result()
        
        # Few-Shot Learning
        few_shot_paths = self._greedy_label_matching(predicted_labels, image_path)
        markdown = self._few_shot_inference(prediction, few_shot_paths, image_path)
        
        # Post-Processing
        markdown = self._post_processing(markdown, predicted_labels, image_path)
        return markdown

    def _zero_shot_inference(self, image_path: str):
        """
        Perform zero-shot inference on the image using the Gemini.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Initial Markdown output.
        """
        response = self._gemini_inference(system_prompt=PROMPTS['zero_shot'], prompt=[self._image_to_bytes(image_path)])
        return clean_markdown_response(response.text)
    
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

        prompt = ["The original image: ", self._image_to_bytes(image_path), "The markdown file to refine: ", markdown]

        if margin:
            side = 'left' if list(margin)[0] == 'margins_left' else 'right'
            system_prompt = PROMPTS['margin_post_processing'].format(side=side)
            response = self._gemini_inference(system_prompt=system_prompt, prompt=prompt)
            return clean_markdown_response(response.text)
        
        if register:
            system_prompt = PROMPTS['register_post_processing']
            response = self._gemini_inference(system_prompt=system_prompt, prompt=prompt)
            return clean_markdown_response(response.text)

        # No refinement if the documents has no margins or is not a register
        return markdown

    def _greedy_label_matching(self, predicted_labels: List[str], image_path: str) -> List[Tuple[Path, Path]]:
        fs_samples = get_few_shot_samples()
        pred_set = set(predicted_labels)
        n = len(pred_set)

        base_dir = Path(__file__).parent / "few_shot"

        def match_score(sample_labels: List[str]) -> float:
            gt_set = set(sample_labels)
            m = len(gt_set) or 1  # avoid div-by-zero
            intersection = len(pred_set & gt_set)
            return (intersection / n + intersection / m) / 2

        scored = [(match_score(s["labels"]), i, s["file_path"]) for i, s in enumerate(fs_samples)]
        best_score, best_idx, best_name = max(scored, key=lambda x: x[0])

        if best_score < 0.7:
            print(f"Poor label overlap (score={best_score:.3f}) for: {predicted_labels}")
            print(f"Image: {image_path}")

        jpg_path = (base_dir / f"{best_name}.jpg").resolve()
        md_path  = (base_dir / f"{best_name}.md").resolve()

        return [(jpg_path, md_path)]

    def _few_shot_inference(self, prediction: str, few_shot_paths: list[tuple[str, str]], image_path: str):
        """
        Builds a few-shot query based on selected ground truth pairs.
        Alongside the examples, it provides the predicted markdown and its original image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            list[str]: Predicted labels.
        """
        prompt = [PROMPTS['few_shot']['base']]

        for idx, (img_path, md_path) in enumerate(few_shot_paths):
            gt_md = Path(md_path).read_text(encoding="utf-8")
            prompt.extend([f"ground_truth_pair{idx}", self._image_to_bytes(img_path), gt_md])

        prompt.extend([
            PROMPTS['few_shot']['markdown'],
            prediction,
            PROMPTS['few_shot']['binary_image'],
            self._image_to_bytes(image_path),
            PROMPTS['few_shot']['CoT'],
        ])

        response = self._gemini_inference(
            system_prompt=PROMPTS['few_shot']['system_prompt'],
            prompt=prompt,
        )
        return clean_markdown_response(response.text)
    
    def parse_files(self, max_retries=3, concurrent_images=10):
        start_time = time.time()
        all_images = convert_tiff_to_jpg_recursive(self.folder_path, is_eval=False, remove_files=True)
        if not all_images:
            print("No documents to parse..")
            return None
        
        num_images = len(all_images)
        results = [None] * num_images
        failures = {path: 0 for path in all_images}
        queue = deque(enumerate(all_images))  # (idx, path)
        lock = threading.Lock()

        def process_single_image(idx, image_path):
            try:
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [executor.submit(self._run_pipeline, image_path) for _ in range(3)]
                    pipeline_results = [f.result() for f in futures]

                prompt = [
                    "The first markdown file: ", pipeline_results[0],
                    "The second markdown file: ", pipeline_results[1],
                    "The third markdown file: ", pipeline_results[2]
                ]
                response = self._gemini_inference(system_prompt=PROMPTS['sample_selection'], prompt=prompt)
                selected = clean_markdown_response(response.text)

                md_path = image_path.replace(".jpg", "_final.md")
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(selected)

                with lock:
                    results[idx] = selected
                return True
            except Exception as e:
                print(f"Failed {image_path}: {e}")
                return False

        print(f"Processing {num_images} images with {concurrent_images} concurrent workers")

        with ThreadPoolExecutor(max_workers=concurrent_images) as executor:
            active_futures = {}

            while queue or active_futures:
                # Fill worker slots
                while len(active_futures) < concurrent_images and queue:
                    idx, path = queue.popleft()
                    future = executor.submit(process_single_image, idx, path)
                    active_futures[future] = (idx, path, 0)  # retries count

                if not active_futures:
                    break

                # Wait for any worker to finish
                for future in as_completed(list(active_futures.keys())):
                    idx, path, retries = active_futures.pop(future)
                    success = future.result()

                    if success:
                        failures[path] = 0
                    else:
                        retries += 1
                        if retries < max_retries:
                            queue.append((idx, path))
                            failures[path] = retries
                            print(f"Retrying {path} ({retries}/{max_retries})")
                        else:
                            print(f"Hard failure after {max_retries} retries: {path}")
                            raise RuntimeError(f"Image {path} failed {max_retries} times. Aborting.")

                    processed = num_images - len(queue) - len(active_futures)
                    print(f"Progress: {processed}/{num_images} ({processed/num_images*100:.1f}%)")

        total_time = time.time() - start_time
        avg_time = total_time / num_images
        print(f"Success: Processed {num_images} documents in {avg_time:.2f}s in the average.")
        
        return {
            "markdown_files": results,
            "markdown_names": [Path(p).stem + ".md" for p in all_images]
        }
    
    def _classify(self, image_path: str):
        """
        Classify the layout of the image using the layout classifier.

        Args:
            image_path (str): Path to the image file.

        Returns:
            list[str]: Predicted labels.
        """
        return self.layout_classifier.run_classifier(image_path=image_path)
        
    def _image_to_bytes(self, image_path: str):
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
    
    def _gemini_inference(self, system_prompt: str, prompt: str, model_id='gemini-3-flash-preview'):
        """
        Calls Gemini with the specified prompts.

        Args:
            system_prompt (str): Baseline guidline for the model's behaviour.
            prompt (str): Concrete task to execute.
            model_id (str): Defines the model which should be used on Google's side to process the request.

        Returns:
            Response: Provides a Gemini Response
        """
        return self.gemini_client.models.generate_content(
            model=model_id,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt),
            contents=prompt
        )
    
def main():

    print("Chrono Quill is setting up..")
    parser = ChronoQuill()
    parser.parse_files()
        
if __name__ == "__main__":
    main()


    