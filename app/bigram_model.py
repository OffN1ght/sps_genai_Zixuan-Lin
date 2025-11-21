import random
from collections import defaultdict
from typing import List


class BigramModel:
    def __init__(self, corpus: List[str]):
        self.bigram_probs = defaultdict(lambda: defaultdict(int))
        self._train(corpus)

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().replace("\n", " ").split()

    def _train(self, corpus: List[str]):
        for sentence in corpus:
            tokens = self._tokenize(sentence)
            for i in range(len(tokens) - 1):
                w1, w2 = tokens[i], tokens[i + 1]
                self.bigram_probs[w1][w2] += 1

        for w1 in self.bigram_probs:
            total_count = sum(self.bigram_probs[w1].values())
            for w2 in self.bigram_probs[w1]:
                self.bigram_probs[w1][w2] /= total_count

    def generate_text(self, start_word: str, length: int) -> str:
        start_word = start_word.lower()
        if start_word not in self.bigram_probs:
            return f"Start word '{start_word}' not in vocabulary."

        words = [start_word]
        current_word = start_word

        for _ in range(length - 1):
            next_words = list(self.bigram_probs[current_word].keys())
            if not next_words: 
                break
            probs = list(self.bigram_probs[current_word].values())
            current_word = random.choices(next_words, weights=probs, k=1)[0]
            words.append(current_word)

        return " ".join(words)