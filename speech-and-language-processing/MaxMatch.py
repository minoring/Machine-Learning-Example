from nltk.corpus import words


def max_match(sentence, dictionary):
    if sentence == "":
        return []

    for i in range(len(sentence), 1, -1):
        first_word = sentence[:i]
        remainder = sentence[i:]

        if first_word in dictionary:
            return [first_word] + max_match(remainder, dictionary)

    # If no word was found, so make a one-character word
    first_word = sentence[0]
    remainder = sentence[1:]
    return [first_word] + max_match(remainder, dictionary)


if __name__ == "__main__":
    turing = "wecannolyseeashortdistanceahead" 
    print(' '.join(max_match(turing, words.words())))

