
import unittest
import sys

import get_accuracy as ga
import utterances
import evaluation

class AccuracyTest(unittest.TestCase):

    def setUp(self):

        self.args_anch_3_3 = ["A", "anch", "3", "3"]
        self.args_anch_3_3_lev = ["A", "anch", "3", "3", "levenshtein", "1"]

        self.to_test = ("DATA/Swedish_MINGLE_dataset/plain/1", "DATA/Swedish_MINGLE_dataset/GOLD/1")
        self.u = utterances.Utterances(self.to_test[0], self.to_test[1])
        self.gold_utterances = self.u._goldutterances
        self.utterances_list = [i for i in self.u._utterances]

        self.utterances_reformatted = list()
        self.ids = list()

        for utterance in self.utterances_list:
            new_utt = utterance[2].split()
            self.utterances_reformatted.append(new_utt)
            self.ids.append((utterance[0], utterance[1]))


    def test_decode_args_fail(self):
        with self.assertRaises(SystemExit) as cm:
            ga.decode_args(["A"])
            self.assertEqual(cm.exception, "Please read README.md for command line usage") 

    def test_decode_args_fail_4(self):
        with self.assertRaises(SystemExit) as cm:
            ga.decode_args(["A", "B", "C", "D"])
            self.assertEqual(cm.exception, "Please read README.md for command line usage")

    def test_decode_args_fail_6(self):
        with self.assertRaises(SystemExit) as cm:
            ga.decode_args(["A", "B", "C", "D", "E", "F"])
            self.assertEqual(cm.exception, "Please read README.md for command line usage")

    def test_decode_args_4(self):
        a = ('anch', 3, 3, None, None)
        self.assertEqual(ga.decode_args(self.args_anch_3_3), a)
    
    def test_decode_args_6(self):
        a = ('anch', 3, 3, 'levenshtein', 1)
        self.assertEqual(ga.decode_args(self.args_anch_3_3_lev), a)

    def test_utterances_len(self):
        """Test that the correct number of utterances were imported"""
        self.assertEqual(len(self.utterances_list), 1032)

    def test_utterances_shape(self):
        self.assertEqual(len(self.utterances_list[0]), 3)

    def test_utterances_content(self):
        a = [u'58.717', u'59.889', u'Jag heter Cornelia']
        self.assertEqual(self.utterances_list[2], a)

    def test_utterances_split(self):
        a = [u'Jag', u'heter', u'Cornelia']
        self.assertEqual(self.utterances_list[2][2].split(), a)

    def test_rest_anch_3_3(self):
        a = [0.7089546948099292,
             0.5307259604882902,
             0.6069794035545357,
             0.14179093896198586,
             0.10614519209765805,
             0.6069794035545357]
        args = self.args_anch_3_3
        args = ga.decode_args(args)

        utt_iter = ga.window(self.utterances_reformatted, args[2])
        id_iter = ga.window(self.ids, args[2])
        ids = [i for i in id_iter]

        ids_and_matches = ga.matches_anchor(utt_iter, args[2], args[3], args[4], False, ids)

        combined = ga.convert_varseta_format(ids_and_matches)
        varseta_eval = evaluation.Evaluation(combined, self.gold_utterances)

        to_test = list()
        to_test.append(varseta_eval.fuzzy_precision)
        to_test.append(varseta_eval.fuzzy_recall)
        to_test.append(varseta_eval.fuzzy_f1)
        to_test.append(varseta_eval.strict_precision)
        to_test.append(varseta_eval.strict_recall)
        to_test.append(varseta_eval.fuzzy_f1)

        self.assertEqual(to_test, a)


    

if __name__ == "__main__":
    unittest.main()
