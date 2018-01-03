class BOWtoDictGenerator:
    """
    Provides method to generate dictionary from bag of words
    """
    def load(self, file):
        """
        """
        with open(file, "r") as f:
            lines_bow = f.readlines()

        bow_list = []
        i = 0
        for line in lines_bow:
            doc_dict = {}
            tokens = line.split()
            rating = tokens[0]
            occrs = tokens[1:]
            for occr in occrs:
                word, count = int(occr.split(':')[0]), int(occr.split(':')[1])
                doc_dict[word] = count
            bow_list.append(doc_dict)
            i = i + 1
        return bow_list
