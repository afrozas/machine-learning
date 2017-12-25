import os


class Load():

    """
    """

    def get_occurences(self):
        """
        """
        true_ratings = []
        pred_ratings = []
        pos_prob = {}
        neg_prob = {}
        line_words = []
        alpha = 1
        total_words = 0
        word_count = []
        probability = {}
        num = 89527

        with open('../dataset/train/labeledBow.feat') as bow:
            lines_bow = bow.readlines()

        with open('../dataset/imdb.vocab') as voc:
            vocab = voc.readlines()
        
        with open('../dataset/test/labeledBow.feat') as train_:
            train = train_.readlines()

        for item in lines_bow:
            t_words = item.split()
            line_words.append(t_words)
            rating, words = t_words[0], t_words[1:]
            if int(rating) > 5:
                for word in words:
                    tokens = word.split(':')
                    total_words = total_words + int(tokens[1])
                    if tokens[0] in pos_prob:
                        pos_prob[str(tokens[0])] = pos_prob[
                            str(tokens[0])] + int(tokens[1])
                    else:
                        pos_prob[str(tokens[0])] = int(tokens[1])
            else:
                for word in words:
                    tokens = word.split(':')
                    total_words = total_words + int(tokens[1])
                    if tokens[0] in neg_prob:
                        neg_prob[str(tokens[0])] = neg_prob[
                            str(tokens[0])] + int(tokens[1])
                    else:
                        neg_prob[str(tokens[0])] = int(tokens[1])

        for i in range(0, num):
            index = str(i)
            if index in pos_prob:
                pos = pos_prob[index]
            else:
                pos = alpha
            if index in neg_prob:
                neg = neg_prob[index]
            else:
                neg = alpha
            total = pos + neg
            probability[index] = [float(pos)/total, float(neg)/total]

        true_ratings = [0] * 25000
        pred_ratings = [0] * 25000
        test_index = 0
        for item in train:
            test_index = test_index + 1
            t_words = item.split()
            line_words.append(t_words)
            rating, words = t_words[0], t_words[1:]
            if int(rating) > 5:
                true_ratings[test_index] = 1
            pos_pred = 1
            neg_pred = 1
            for word in words:
                tokens = word.split(':')
                pos_pred = pos_pred * (
                    pow(probability[str(tokens[0])][0], int(tokens[1])))
                neg_pred = neg_pred * (
                    pow(probability[str(tokens[0])][1], int(tokens[1])))
            if pos_pred > neg_pred:
                pred_ratings[test_index] = 1

        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(0, test_index):
            if true_ratings[i] == 1 and pred_ratings[i] == 1:
                tp += 1
            elif true_ratings[i] == 0 and pred_ratings[i] == 1:
                fp += 1
            if true_ratings[i] == 0 and pred_ratings[i] == 0:
                tn += 1
            if true_ratings[i] == 1 and pred_ratings[i] == 0:
                fn += 1

        precision = float(tp)/(tp+fp)
        recall = float(tp)/(tp+fn)
        print("tp fp tn fn : ", tp, fp, tn, fn)
        print("Precision : %.2f%%" % (precision*100))
        print("Recall : %.2f%%" % (recall*100))
        print("F Measure : %.2f%%" % (float(2*precision*recall)/(precision+recall)*100))