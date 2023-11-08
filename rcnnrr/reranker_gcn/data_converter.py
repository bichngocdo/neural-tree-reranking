from rcnnrr.data.data_converters import AbstractDataConverter, convert_to_tensor


class DataConverter(AbstractDataConverter):
    def __init__(self):
        super(AbstractDataConverter, self).__init__()

    def convert(self, batch):
        words, pt_words, chars, tags, _, _, _, _, nb_in_edges, nb_in_labels, nb_out_edges, nb_out_labels, \
            nb_scores, nb_labels, nb_margins = batch

        results = list()
        results.append(convert_to_tensor(words, value=0))
        if pt_words is not None:
            results.append(convert_to_tensor(pt_words, value=0))
        else:
            results.append(None)
        results.append(convert_to_tensor(chars, value=0))
        results.append(convert_to_tensor(tags, value=0))

        results.append(convert_to_tensor(nb_in_edges, value=-1))
        results.append(convert_to_tensor(nb_in_labels, value=0))
        results.append(convert_to_tensor(nb_out_edges, value=-1))
        results.append(convert_to_tensor(nb_out_labels, value=0))

        results.append(convert_to_tensor(nb_labels, value=-1))
        results.append(convert_to_tensor(nb_margins, value=-1))
        return results
