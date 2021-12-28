from fuzzywuzzy import fuzz


class FuzzyReplacer:
    def __init__(self):
        self.job_vocabulary = {
            "Software": 25,
            "Engineer": 25,
            "Team": 29,
            "Lead": 29,
            "Human": 36,
            "Resources": 36,
            "Manager": 28,
            "Scrum": 32,
            "Master": 32,
        }

    def substitute_Lev(self, word, vocab):
        vocabulary = self.job_vocabulary if vocab == 'job' else {}
        if vocabulary.get(word) is not None:
            return word

        if word in ['l', 'i']:
            return 'I'

        get_LR = self.get_Lev_ratio_calculator(word)
        max_LR = max(map(get_LR, vocabulary.keys()))
        max_LR_words = list(filter(lambda word: get_LR(word) == max_LR, vocabulary.keys()))
        if len(max_LR_words) == 1:
            return max_LR_words[0]

        return max(max_LR_words, key=lambda word: vocabulary[word])

    def get_Lev_ratio_calculator(self, word):
        return lambda other_word: fuzz.ratio(word, other_word)

    def convert_Lev(self, result, vocabulary='job'):
        words = result.split(' ')
        for i in range(len(words)):
            words[i] = self.substitute_Lev(words[i], vocabulary)
        return ' '.join(words)


