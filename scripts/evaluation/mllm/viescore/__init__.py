import sys
sys.path.insert(0, 'viescore')

from .viescore_utils import (
    mllm_output_to_dict
)
import math

class VIEScore:
    def __init__(self, backbone="gpt4o", task="t2i", key_id=None, vllm_ture=True) -> None:
        self.task = task
        self.backbone_name = backbone

        if self.task not in ["t2i", "tie", "t2v"]:
            raise ValueError("task must be either 't2i' or 'tie'")

        if self.backbone_name == "qwen25vl":
            from .mllm_tools.qwen25vl_eval import Qwen25VL
            self.model = Qwen25VL(key_id, vllm_ture)
        elif self.backbone_name == "google":
            from .mllm_tools.google_eval import GoogleEval
            self.model = GoogleEval(vllm_ture)
        else:
            raise NotImplementedError("backbone not supported")

        self.VF_prompt = """ 
        ## Role
        You are a professional photographic post-production expert and visual perception analyst.
        Your task is to evaluate whether the retouched image successfully establishes the user-specified target region as the dominant visual center.
        ---
        ## Input
        - One retouched image  
        - One retouch instruction describing the intended focal region  
        ---
        ## Evaluation Objective
        Assess whether the retouched image demonstrates clear, coherent, and professionally structured visual focus aligned with the retouch instruction.
        Your evaluation must be based on perceptual attention, visual hierarchy, and dominance structure — not semantic correctness or pixel-level comparison.
        ---
        ## Evaluation Criteria
        Consider the following aspects:
        1. **Primary Visual Landing Point**  
        Does the viewer’s first perceptual attention naturally fall on the instructed region?
        2. **Hierarchy Clarity**  
        Is there a clear primary–secondary visual structure without competing focal points?
        3. **Contrast & Saliency Control**  
        Is the target region enhanced through brightness, color contrast, clarity, or tonal separation?
        4. **Attention Guidance Mechanisms**  
        Are lighting, contrast shaping, depth cues, tonal sculpting, or vignetting effectively guiding gaze toward the target?
        5. **Dominance Stability**  
        Does the target region remain the strongest perceptual anchor of the image?
        ---
        ## Scoring Standard (0–10)
        - **0–2**: Target region is not visually dominant.
        - **3–5**: Partial emphasis, but visual competition remains.
        - **6–8**: Clear and effective visual focus.
        - **9–10**: Exceptional dominance with a professionally structured visual hierarchy.
        ---
        ## Output Format
        # Return only valid JSON:
        # ```json
        # {
        #     "score": <0-10>,
        #     "reasoning": "Concise explanation focusing on attention redistribution and visual hierarchy."
        # }

        # Editing instruction: <instruction>
        """
        self.PQ_prompt = """ 
        You are a post-production specialist with expertise in enhancing photographic imagery through advanced digital editing techniques. We now need your help to evaluate the performance of an AI-powered image post-editing tool for photography.
        
        INPUTS:
            1. Two images will be provided: The first being the original photographic image and the second being an edited version of the first.
            2. The editing instruction will be provided: The post-editing needs of photographic images expressed by users with no image processing knowledge.

        METRICS (From scale 0 to 10): 
            Content Consistency Score: A second score from 0 to 10 will rate the consistency of image content before and after editing. 
                - Need to compare before and after images to assess content consistency.
                - The edited image should maintain consistency in key visual elements such as the shape of landscapes, human figures (including posture, gender, and appearance), building structures, and other important features.
                - The edited image needs to maintain the consistency of local details, such as letters on clothes, textures of buildings, etc.
                - 0 indicates that the content of the image before and after editing is completely inconsistent. 
                - 10 indicates that the content of the edited image is exactly the same as the original image.
            
        Put the score in a list such that output score = [score1], where 'score1' evaluates the Content Consistency Score.

        You will have to give your output in this way (Keep your reasoning concise and short.):
        {
        "score" : [...],
        "reasoning" : "..."
        }
       
        Editing instruction: <instruction>
            """
        self.PQ_prompt_local = """ 
        You are a post-production specialist with expertise in enhancing photographic imagery through advanced digital editing techniques. We now need your help to evaluate the performance of an AI-powered image post-editing tool for photography.
        
        INPUTS:
            1. Two images will be provided: The first being the original photographic image and the second being an edited version of the first.
            2. The editing instruction will be provided: The post-editing needs of photographic images expressed by users with no image processing knowledge.

        METRICS (From scale 0 to 10): 
            Content Consistency Score: A second score from 0 to 10 will rate the consistency of image content before and after editing. 
                - Need to compare before and after images to assess content consistency.
                - The edited image should maintain consistency in key visual elements such as the shape of landscapes, human figures (including posture, gender, and appearance), building structures, and other important features.
                - The edited image needs to maintain the consistency of local details, such as letters on clothes, textures of buildings, etc.
                - 0 indicates that the content of the image before and after editing is completely inconsistent. 
                - 10 indicates that the content of the edited image is exactly the same as the original image.
            
        IMPORTANT NOTEs:
            1. The focus of this assessment is the performance of AI-enabled post-processing tools for photographic images in response to users' local modification instructions.
            2. The two images are the same localized areas of interest selected by the user in the original and edited images.
            3. User editing instructions involve both global adjustments to the entire image and localized edits to specific regions of interest. 
        
        Put the score in a list such that output score = [score1], where 'score1' evaluates the Content Consistency Score.

        You will have to give your output in this way (Keep your reasoning concise and short.):
        {
        "score" : [...],
        "reasoning" : "..."
        }
        Editing instruction: <instruction>
            """

    def evaluate(self, source_image, edited_image, text_prompt, local_flag=False, extract_all_score=True, echo_output=False):
        if self.task == "tie":
            _VF_prompt = self.VF_prompt.replace("<instruction>", text_prompt) 
            _PQ_prompt = self.PQ_prompt.replace("<instruction>", text_prompt) 
        VF_prompt_final = self.model.prepare_prompt([edited_image], _VF_prompt)
        PQ_prompt_final = self.model.prepare_prompt([source_image, edited_image], _PQ_prompt)
        # print(VF_prompt_final)

        results_dict = {}

        VF_dict = False
        tries = 0
        max_tries = 1
        while VF_dict is False:
            tries += 1
            guess_if_cannot_parse = True if tries > max_tries else False
            result_VF = self.model.get_parsed_output(VF_prompt_final)
            VF_dict = mllm_output_to_dict(result_VF, give_up_parsing=guess_if_cannot_parse)

        if VF_dict == "rate_limit_exceeded":
            print("rate_limit_exceeded") 
            raise ValueError("rate_limit_exceeded")
        results_dict['VF'] = VF_dict

        PQ_dict = False
        tries = 0
        while PQ_dict is False:
            tries += 1
            guess_if_cannot_parse = True if tries > max_tries else False
            result_PQ = self.model.get_parsed_output(PQ_prompt_final)
            PQ_dict = mllm_output_to_dict(
                result_PQ,
                give_up_parsing=guess_if_cannot_parse
            )
        if PQ_dict == "rate_limit_exceeded":
            print("rate limit exceeded")
            raise ValueError("rate_limit_exceeded")
        results_dict["PQ"] = PQ_dict

        if echo_output:
            print("results_dict", results_dict)
        if extract_all_score:
            VF_score = results_dict['VF']['score'][0]
            # PQ_score = results_dict['PQ']['score'][1]
            PQ_score = results_dict['PQ']['score'][0]
            O_score = math.sqrt(VF_score * PQ_score)
            return {
                "VF": float(VF_score),
                "PQ": float(PQ_score),
                "O": float(O_score),
            }

        return results_dict
    
    def evaluate_local(self, source_image, edited_image, text_prompt, local_flag=False, extract_all_score=True, echo_output=False):
        if self.task == "tie":
            _PQ_prompt = self.PQ_prompt_local.replace("<instruction>", text_prompt) 
        PQ_prompt_final = self.model.prepare_prompt([source_image, edited_image], _PQ_prompt)
        # print(VF_prompt_final)

        results_dict = {}

        PQ_dict = False
        tries = 0
        max_tries = 1
        while PQ_dict is False:
            tries += 1
            guess_if_cannot_parse = True if tries > max_tries else False
            result_PQ = self.model.get_parsed_output(PQ_prompt_final)
            PQ_dict = mllm_output_to_dict(
                result_PQ,
                give_up_parsing=guess_if_cannot_parse
            )
        if PQ_dict == "rate_limit_exceeded":
            print("rate limit exceeded")
            raise ValueError("rate_limit_exceeded")
        results_dict["PQ"] = PQ_dict

        if echo_output:
            print("results_dict", results_dict)
        if extract_all_score:
            PQ_score = results_dict['PQ']['score'][0]
            # PQ_score = results_dict['PQ']['score'][1]
            return {
                "PQ": float(PQ_score),
            }
        breakpoint()

        return results_dict
