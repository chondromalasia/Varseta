import sys
import difflib
from itertools import islice

import utterances
import evaluation


def window(seq, n=2):
    """ Returns a sliding window (of width n) over data from the iterable
    s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ... """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def levenshtein_dist(s1, s2):
    """Input two strings, returns levenstein distance between the two"""

    s1 = ' ' + s1
    s2 = ' ' + s2
    d = {}
    S1 = len(s1)
    S2 = len(s2)
    for i in range(S1):
        d[i, 0] = i
    for j in range(S2):
        d[0, j] = j
    for j in range(1, S2):
        for i in range(1, S1):
            if s1[i] == s2[j]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + 1)
    return d[S1 - 1, S2 - 1]


def levenshtein(A, B, overlap):
    """ 2 lists of strings and an integar
    returns the number of strings in common, adjusted by levenstein distance"""
    match_count = 0
    for word in A:
        if isinstance(word, type(None)):
            continue

        for word_ in B:
            if isinstance(word_, type(None)):
                continue
            if levenshtein_dist(word, word_) <= overlap:
                match_count += 1
                break
    return match_count


def matches_wrapper(utterance_A, utterance_B, match_type, minimum_matches, overlap):
    if match_type is None:
        if len(set(utterance_A).intersection(set(utterance_B))) >= minimum_matches:
            return True
    elif match_type == 'levenshtein':
        if levenshtein(utterance_A, utterance_B, overlap) >= minimum_matches:
            return True
    elif match_type == 'difflib':
        # some entries are None
        utterance_A = ' '.join([i for i in utterance_A if i])
        utterance_B = ' '.join([i for i in utterance_B if i])
        distance = difflib.SequenceMatcher(lambda x: x == ' ', utterance_A, utterance_B).ratio()
        if distance >= overlap:
            return True


def matches_incremental(it, minimum_matches, match_type, overlap, return_count=True, ids=None):
    """Given an iterator returns the minimum matches."""

    matches = 0
    matches_list = []

    for count, i in enumerate(it):
        pairs = window(i)
        for k, j in pairs:
            if matches_wrapper(k, j, match_type, minimum_matches, overlap):
                matches += 1
                if ids:
                    matches_list.append((ids[count], i))
                else:
                    matches_list.append(i)

                # stop iterating through pairs when a match is found
                break

    if return_count:
        return matches
    else:
        return matches_list


def matches_anchor(it, minimum_matches, match_type, overlap, return_count=True, ids=None):
    """Returns varation set matches using anchor method"""

    matches = 0
    matches_list = []

    for count, i in enumerate(it):
        utterances = iter(i)
        first = next(utterances)

        for utterance in utterances:
            if matches_wrapper(first, utterance, match_type, minimum_matches, overlap):
                matches += 1
                if ids:
                    matches_list.append((ids[count], i))
                else:
                    matches_list.append(i)

    if return_count:
        return matches
    else:
        return matches_list


def convert_varseta_format(results):
    """ Returns ids and matches in the format for the Varseta evaluation

    [[[(id_1_1, id_1_2), (id_2_1, id_2_2)], ["utterance_1", "utterance_2"]], ...]
    to
    [[['id_1_1', 'id_1_2', 'Utterance_1'],['id_2_1', 'id_2_2', 'utterance_2']], ...]
    """

    return_list = []

    for result in results:
        dummy_list = []

        for id_, match_list in zip(result[0], result[1]):
            combined = list(id_)
            combined.append(' '.join(match_list))
            dummy_list.append(combined)

        return_list.append(dummy_list)

    return return_list


def decode_args(args):
    """Parses commandline args"""

    exit_text = "Please read notes.txt for command line usage"

    if len(args) == 4:
        try:
            return args[1], int(args[2]), int(args[3]), None, None
        except ValueError:
            sys.exit(exit_text)

    elif len(args) == 6:
        if args[4] == "levenshtein":
            try:
                return args[1], int(args[2]), int(args[3]), args[4], int(args[5])
            except ValueError:
                sys.exit(exit_text)

        elif args[4] == "difflib":
            try:
                return args[1], int(args[2]), int(args[3]), args[4], float(args[5])
            except ValueError:
                sys.exit(exit_text)
        else:
            sys.exit(exit_text)

    sys.exit("Please read notes.txt for command line usage")


def main():

    args = decode_args(sys.argv)

    to_dos = [
        ("DATA/Swedish_MINGLE_dataset/plain/1", "DATA/Swedish_MINGLE_dataset/GOLD/1"),
        ("DATA/Swedish_MINGLE_dataset/plain/2", "DATA/Swedish_MINGLE_dataset/GOLD/2"),
        ("DATA/Swedish_MINGLE_dataset/plain/3", "DATA/Swedish_MINGLE_dataset/GOLD/3"),
        ("DATA/Swedish_MINGLE_dataset/plain/4", "DATA/Swedish_MINGLE_dataset/GOLD/4")]

    fuzzy_precisions, strict_precisions, fuzzy_recalls, strict_recalls,\
            fuzzy_f1s, strict_f1s = [], [], [], [], [], []

    for to_do in to_dos:
        print("Finding variation sets in" + to_do[0])
        u = utterances.Utterances(to_do[0], to_do[1])
        gold_utterances = u._goldutterances

        utterances_reformatted = []
        ids = []

        for utterance in u._utterances:
            new_utt = utterance[2].split()
            utterances_reformatted.append(new_utt)
            ids.append((utterance[0], utterance[1]))

        utt_iter = window(utterances_reformatted, args[1])
        id_iter = window(ids, args[1])
        ids = [i for i in id_iter]

        if args[0] == "anch":
            ids_and_matches = matches_anchor(utt_iter, args[2], args[3], args[4], False, ids)
        else:
            ids_and_matches = matches_incremental(utt_iter, args[2], args[3], args[4], False, ids)

        combined = convert_varseta_format(ids_and_matches)

        varseta_eval = evaluation.Evaluation(combined, gold_utterances)

        fuzzy_precisions.append(varseta_eval.fuzzy_precision)
        strict_precisions.append(varseta_eval.strict_precision)
        fuzzy_recalls.append(varseta_eval.fuzzy_recall)
        strict_recalls.append(varseta_eval.strict_recall)
        fuzzy_f1s.append(varseta_eval.fuzzy_f1)
        strict_f1s.append(varseta_eval.strict_f1)

        print('\tFuzzy Precision: {:0.2f}'.format(varseta_eval.fuzzy_precision))
        print('\tFuzzy Recall: {:0.2f}'.format(varseta_eval.fuzzy_recall))
        print('\tFuzzy F1: {:0.2f}'.format(varseta_eval.fuzzy_f1))
        print('')
        print('\tStrict Precision: {:0.2f}'.format(varseta_eval.strict_precision))
        print('\tStrict Recall: {:0.2f}'.format(varseta_eval.strict_recall))
        print('\tStrict F1: {:0.2f}'.format(varseta_eval.strict_f1))
        print('\n')

    avg_fuzzy_precision = sum([i for i in fuzzy_precisions])/len(fuzzy_precisions)

    avg_fuzzy_recall = sum([i for i in fuzzy_recalls])/len(fuzzy_recalls)
    avg_fuzzy_f1 = sum([i for i in fuzzy_f1s])/len(fuzzy_f1s)
    avg_strict_precision = sum([i for i in strict_precisions])/len(strict_precisions)
    avg_strict_recall = sum([i for i in strict_recalls])/len(strict_recalls)
    avg_strict_f1 = sum([i for i in strict_f1s])/len(strict_f1s)


    print('Average Scores:')
    print('Average Fuzzy Precision: {:0.2f}'.format(avg_fuzzy_precision))
    print('Average Fuzzy Recall: {:0.2f}'.format(avg_fuzzy_recall))
    print('Average Fuzzy F1: {:0.2f}'.format(avg_fuzzy_f1))
    print('')
    print('Average Strict Precision: {:0.2f}'.format(avg_strict_precision))
    print('Average Strict Recall: {:0.2f}'.format(avg_strict_recall))
    print('Average Strict F1: {:0.2f}'.format(avg_strict_f1))


if __name__ == "__main__":
    main()
