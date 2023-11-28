# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
os.environ['HF_HOME'] = '/src/hf_models'
os.environ['TORCH_HOME'] = '/src/torch_models'
from cog import BasePredictor, Input, Path
import torch
import whisperx
import json


compute_type="float16"
class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda"
        self.model = whisperx.load_model("large-v2", self.device, language="en", compute_type=compute_type)
        self.alignment_model, self.metadata = whisperx.load_align_model(language_code="en", device=self.device)

        def predict(self, segment: Dict = Input(description="Segment dictionary with text, start, and end times")):
        # Extract values from segment
        text = segment['text']
        start_time = segment['start']
        end_time = segment['end']

        # Split text into words and calculate timestamps
        words = text.split()
        total_duration = end_time - start_time
        duration_per_word = total_duration / len(words)

        # Prepare word-level timestamps
        word_timestamps = []
        current_start = start_time
        for word in words:
            word_end = current_start + duration_per_word
            word_timestamps.append({'word': word, 'start': current_start, 'end': word_end})
            current_start = word_end

        return word_timestamps -> str:
        """Run a single prediction on the model"""
        with torch.inference_mode():
            result = self.model.transcribe(str(audio), batch_size=batch_size) 
            # result is dict w/keys ['segments', 'language']
            # segments is a list of dicts,each dict has {'text': <text>, 'start': <start_time_msec>, 'end': <end_time_msec> }
            print("🔴 DEBUG LOGS MAXROOM")
            print(result['segments'])
            if align_output:
                # NOTE - the "only_text" flag makes no sense with this flag, but we'll do it anyway
                result = whisperx.align(result['segments'], self.alignment_model, self.metadata, str(audio), self.device, return_char_alignments=False)
                # dict w/keys ['segments', 'word_segments']
                # aligned_result['word_segments'] = list[dict], each dict contains {'word': <word>, 'start': <start_time_msec>, 'end': <end_time_msec>, 'score': probability}
                #   it is also sorted
                # aligned_result['segments'] - same as result segments, but w/a ['words'] segment which contains timing information above. 
                # return_char_alignments adds in character level alignments. it is: too many. 
            if only_text:
                return ''.join([val.text for val in result['segments']])
            if debug:
                print(f"max gpu memory allocated over runtime: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB")
        return json.dumps(result['segments'])

