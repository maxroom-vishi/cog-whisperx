import torch
import torchaudio
from cog import BasePredictor, Input, Path
from dataclasses import dataclass
from typing import Any, List

# Predictor class for forced alignment using Wav2Vec2
class Predictor(BasePredictor):
    def setup(self):
        # Setup device and load the Wav2Vec2 model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = self.bundle.get_model().to(self.device)
        self.labels = self.bundle.get_labels()

    def predict(self, audio: Path = Input(description="Audio file path"), transcript: str = Input(description="Transcript text")) -> Any:
        # Load and preprocess audio
        waveform, _ = torchaudio.load(str(audio))
        emissions, _ = self.model(waveform.to(self.device))
        emissions = torch.log_softmax(emissions, dim=-1)
        emission = emissions[0].cpu().detach()

        # Process transcript
        transcript = "|" + "|".join(transcript.split()) + "|"
        dictionary = {c: i for i, c in enumerate(self.labels)}
        tokens = []
        for c in transcript:
            if c in dictionary:
                tokens.append(dictionary[c])
            else:
                # Handle the case where the character is not in the dictionary
                # For example, you can skip the character or raise an error
                pass  # or raise ValueError(f"Character '{c}' not found in the dictionary")


        # Generate the trellis for alignment
        trellis = self.get_trellis(emission, tokens)

        # Perform backtracking to find the most likely path
        path = self.backtrack(trellis, emission, tokens)

        # Segment the path into words
        segments = self.merge_repeats(path, transcript)
        word_segments = self.merge_words(segments)

        # Return segments as a list of dictionaries
        return [word.__dict__ for word in word_segments]

    def get_trellis(self, emission, tokens, blank_id=0):
        num_frame = emission.size(0)
        num_tokens = len(tokens)

        trellis = torch.zeros((num_frame, num_tokens))
        trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
        trellis[0, 1:] = -float("inf")
        trellis[-num_tokens + 1 :, 0] = float("inf")

        for t in range(num_frame - 1):
            trellis[t + 1, 1:] = torch.maximum(
                # Score for staying at the same token
                trellis[t, 1:] + emission[t, blank_id],
                # Score for changing to the next token
                trellis[t, :-1] + emission[t, tokens[1:]],
            )
        return trellis

    def backtrack(self, trellis, emission, tokens, blank_id=0):
        t, j = trellis.size(0) - 1, trellis.size(1) - 1

        path = [Point(j, t, emission[t, blank_id].exp().item())]
        while j > 0:
            # Should not happen but just in case
            assert t > 0

            # 1. Figure out if the current position was stay or change
            # Frame-wise score of stay vs change
            p_stay = emission[t - 1, blank_id]
            p_change = emission[t - 1, tokens[j]]

            # Context-aware score for stay vs change
            stayed = trellis[t - 1, j] + p_stay
            changed = trellis[t - 1, j - 1] + p_change

            # Update position
            t -= 1
            if changed > stayed:
                j -= 1

            # Store the path with frame-wise probability.
            prob = (p_change if changed > stayed else p_stay).exp().item()
            path.append(Point(j, t, prob))

        # Now j == 0, which means, it reached the SoS.
        # Fill up the rest for the sake of visualization
        while t > 0:
            prob = emission[t - 1, blank_id].exp().item()
            path.append(Point(j, t - 1, prob))
            t -= 1

        return path[::-1]

    def merge_repeats(self, path, transcript):
        # Merge repeated labels in the path
        i1, i2 = 0, 0
        segments = []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(
                Segment(
                    transcript[path[i1].token_index],
                    path[i1].time_index,
                    path[i2 - 1].time_index + 1,
                    score,
                )
            )
            i1 = i2
        return segments

    def merge_words(self, segments, separator="|"):
        words = []
        i1, i2 = 0, 0
        while i1 < len(segments):
            if i2 >= len(segments) or segments[i2].label == separator:
                if i1 != i2:
                    segs = segments[i1:i2]
                    word = "".join([seg.label for seg in segs])
                    score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                    words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
                i1 = i2 + 1
                i2 = i1
            else:
                i2 += 1
        return words

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float
    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"
    @property
    def length(self):
        return self.end - self.start

# Additional helper functions (if needed) can be added here
