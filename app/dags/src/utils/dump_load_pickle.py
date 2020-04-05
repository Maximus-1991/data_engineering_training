import pickle

def load_data_batch(batch_name):
    f = open(batch_name, "rb")
    batch = pickle.load(f, encoding="latin1")
    f.close()
    return batch

def pickle_data(data, pickle_file_path):
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        var = pickle.load(f)
    return var
