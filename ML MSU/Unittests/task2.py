def check(s, filename):
    dict_words = {}
    for word in s.split():
        word = word.lower()
        dict_words[word] = dict_words.get(word, 0) + 1
    sorted_words = sorted(dict_words.items(), key=lambda x: (x[0], x[1]))
    dict_words = dict(sorted_words)
    with open(filename, 'w') as f:    
        for word, count in dict_words.items():
            f.write(f"{word} {count}\n")
