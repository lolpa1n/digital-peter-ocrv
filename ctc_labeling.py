import re

from settings import *


class CTCLabeling:
    blank = '*'

    def __init__(self, chars):
        self.chars = [self.blank] + sorted(list([char for char in chars if char != '*']))
        self.char2ind = {c: i for i, c in enumerate(self.chars)}

    def encode(self, text):
        text = self.preprocess(text)
        return [self.char2ind[char] for char in text]

    def decode(self, indexes):
        chars = []
        for i, index in enumerate(indexes):
            if index == self.padding_value:
                continue
            if i == 0:
                chars.append(self.chars[index])
                continue
            if indexes[i - 1] != index:
                chars.append(self.chars[index])
                continue
        text = ''.join(chars).strip()
        text = self.postprocess(text)
        return text

    @staticmethod
    def preprocess(text):
        """ Метод чистки текста перед self.encode  """
        eng2rus = {
            'o': 'о',  # 2, "20_16_0" - eng, "41_10_1" - eng
            'a': 'а',  # 3, "217_40_21" - eng, "188_4_21" - ???, "214_37_11" - eng
            'c': 'с',  # 9, на деле все кейсы русские
            'e': 'е',  # 36, почти все действительно eng | популярный кейс "piter"
            'p': 'р',  # 35, почти все действительно eng | популярный кейс "piter"
            '×': 'х',  # 2 - кажется, что обычная русская х, "332_35_26", "332_35_27"
            # --------------------------------
            # встречаются один раз в train >.<
            # --------------------------------
            '/': '',  # что это? "313_12_9"
            '…': '',  # что это? "47_20_5"
            '|': '',  # что это? 2, но в одном кейсе "380_8_1"
            '–': '',  # что это? 3, но в одном кейсе "416_22_2"
            'ǂ': '',  # кейс с повернутой на 90 картинкой, "265_6_17", символ явно не этот # TODO учесть
            'u': 'и',  # действительно eng "41_10_1"
            'k': 'к',  # 1, явно русская k "188_4_15"
            'і': 'i',
        }
        text = text.strip()
        text = ''.join([eng2rus.get(char, char) for char in text])
        text = re.sub(r'\b[pр]s\b', 'р s', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    @staticmethod
    def postprocess(text):
        """ Метод чистки текста после self.decode  """
        text = text.strip()
        text = text.replace('і', 'i')
        text = text.replace('рit', 'pit')
        text = text.replace('pitеr', 'piter')

        text = text.replace('irе', 'ire')
        text = text.replace('hеr', 'her')

        text = text.replace('mоn', 'mon')
        text = text.replace('siе', 'sie')
        text = text.replace('иr', 'ur')

        text = re.sub(r'точки а\b', 'точки a', text)
        text = re.sub(r'точки е\b', 'точки e', text)
        text = re.sub(r'точки с\b', 'точки c', text)

        text = re.sub(r'точка а\b', 'точка a', text)
        text = re.sub(r'точка е\b', 'точка e', text)
        text = re.sub(r'точка с\b', 'точка c', text)

        text = re.sub(r'разстоянием а\b', 'разстоянием a', text)
        text = re.sub(r'разстоянием е\b', 'разстоянием e', text)
        text = re.sub(r'разстоянием с\b', 'разстоянием c', text)

        text = re.sub(r'разстояние а\b', 'разстояние a', text)
        text = re.sub(r'разстояние е\b', 'разстояние e', text)
        text = re.sub(r'разстояние с\b', 'разстояние c', text)

        text = re.sub(r'линѣи а\b', 'линѣи a', text)
        text = re.sub(r'линѣи е\b', 'линѣи e', text)
        text = re.sub(r'линѣи с\b', 'линѣи c', text)

        text = re.sub(r'линѣi а\b', 'линѣi a', text)
        text = re.sub(r'линѣi е\b', 'линѣi e', text)
        text = re.sub(r'линѣi с\b', 'линѣi c', text)

        text = re.sub(r'линѣя а\b', 'линѣя a', text)
        text = re.sub(r'линѣя е\b', 'линѣя e', text)
        text = re.sub(r'линѣя с\b', 'линѣя c', text)

        text = re.sub(r'линiи а\b', 'линiи a', text)
        text = re.sub(r'линiи е\b', 'линiи e', text)
        text = re.sub(r'линiи с\b', 'линiи c', text)

        text = re.sub(r'линii а\b', 'линii a', text)
        text = re.sub(r'линii е\b', 'линii e', text)
        text = re.sub(r'линii с\b', 'линii c', text)

        text = re.sub(r'линiя а\b', 'линiя a', text)
        text = re.sub(r'линiя е\b', 'линiя e', text)
        text = re.sub(r'линiя с\b', 'линiя c', text)

        ##################################
        # ' )+0123456789[]bdfghilmnrstабвгдежзийклмнопрстуфхцчшщъыьэюяѣ⊕⊗'
        def replace_similar_eng(text, symb_rus, symb_eng):
            text = re.sub(r'([a-z])' + symb_rus + r'([a-z])', r'\g<1>' + symb_eng + r'\g<2>', text)
            text = re.sub(symb_rus + r'([a-z][a-z])', symb_eng + r'\g<1>', text)
            text = re.sub(r'([a-z][a-z])' + symb_rus, r'\g<1>' + symb_eng, text)
            return text

        for symb_rus, symb_eng in [
            ('е', 'e'),
            ('а', 'a'),
            ('р', 'p'),
            ('с', 'c'),
            ('о', 'o'),
            ('х', 'x'),
            ('и', 'u'),
            ('т', 't'),
            ('к', 'k'),
            ('0', 'o'),
        ]:
            text = replace_similar_eng(text, symb_rus, symb_eng)

        ####################################

        ##################################
        # ' )+0123456789[]bdfghilmnrstабвгдежзийклмнопрстуфхцчшщъыьэюяѣ⊕⊗'
        def replace_similar_rus(text, symb_rus, symb_eng):
            text = re.sub(r'([а-яѣ])' + symb_eng + r'([а-яѣ])', r'\g<1>' + symb_rus + r'\g<2>', text)
            text = re.sub(symb_eng + r'([а-яѣ][а-яѣ])', symb_rus + r'\g<1>', text)
            text = re.sub(r'([а-яѣ][а-яѣ])' + symb_eng, r'\g<1>' + symb_rus, text)
            return text

        for symb_rus, symb_eng in [
            ('е', 'e'),
            ('а', 'a'),
            ('р', 'p'),
            ('с', 'c'),
            ('о', 'o'),
            ('х', 'x'),
            ('и', 'u'),
            ('т', 't'),
            ('к', 'k'),
            ('о', '0'),
        ]:
            text = replace_similar_rus(text, symb_rus, symb_eng)
        ####################################

        text = text.replace('р s', 'p s')
        text = text.replace('рs', 'p s')

        # @ddimitrov заверил в слаке, что Петр не употребляет ENG  "C" --> замена не грубая.
        # Заменим тоже:
        text = text.replace('c', 'с')

        text = text.replace('#', '')

        text = re.sub(r'\s+', ' ', text)

        return text

    @property
    def padding_value(self):
        return self.char2ind[self.blank]

    def __len__(self):
        return len(self.chars)


CTC_LABELING = CTCLabeling(OCR_CHARS)
