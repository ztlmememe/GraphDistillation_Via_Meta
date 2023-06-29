import dgl
import numpy as np
import pickle

# load features
feat = np.load(root + 'features.npy', allow_pickle = True)

root = DATA_PATH
task_setup = 'Disjoint'/'Shared' # for disjoint and shared label settings
task_mode = True/False # for tissue_PPI
task_n = 1-10 # task number index for tissue_PPI
link_pred_mode = True/False # link prediction or not
mode = 'train'/'val'/'test'

# load graphs in dgl format
with open(root + '/graph_dgl.pkl', 'rb') as f:
    dgl_graph = pickle.load(f)

if task_setup == 'Disjoint':    
    with open(root + 'label.pkl', 'rb') as f:
        info = pickle.load(f)
elif task_setup == 'Shared':
    if task_mode == 'True':
        root = root + '/task' + str(task_n) + '/'
    with open(root + 'label.pkl', 'rb') as f:
        info = pickle.load(f)


if link_pred_mode:
    dictLabels_spt, dictGraphs_spt, dictGraphsLabels_spt = loadCSV(os.path.join(root, mode + '_spt.csv'))
    dictLabels_qry, dictGraphs_qry, dictGraphsLabels_qry = loadCSV(os.path.join(root, mode + '_qry.csv'))
    dictLabels, dictGraphs, dictGraphsLabels = loadCSV(os.path.join(root, mode + '.csv'))  # csv path
else:
    dictLabels, dictGraphs, dictGraphsLabels = loadCSV(os.path.join(root, mode + '.csv'))  # csv path

def loadCSV(csvf):
        dictGraphsLabels = {}
        dictLabels = {}
        dictGraphs = {}

        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[1]
                g_idx = int(filename.split('_')[0])
                label = row[2]
                # append filename to current label

                if g_idx in dictGraphs.keys():
                    dictGraphs[g_idx].append(filename)
                else:
                    dictGraphs[g_idx] = [filename]
                    dictGraphsLabels[g_idx] = {}

                if label in dictGraphsLabels[g_idx].keys():
                    dictGraphsLabels[g_idx][label].append(filename)
                else:
                    dictGraphsLabels[g_idx][label] = [filename]

                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels, dictGraphs, dictGraphsLabels