class matrix_point_plot():

    def __init__(self, node_couple_set, file):
        self.node_couple_set = node_couple_set
        self.output_file = file

    def point_plot(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        node_couple_set = pd.DataFrame(self.node_couple_set, columns=['x', 'y'])
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        plt.scatter(node_couple_set['x'], node_couple_set['y'], c='k', marker='o', s=1)
        plt.scatter(node_couple_set['y'], node_couple_set['x'], c='k', marker='o', s=1)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        ax.set_xlabel('Nodes', fontsize=17, fontname='Times New Roman')
        ax.set_ylabel('Nodes', fontsize=17, fontname='Times New Roman')
        max_node = max(self.node_couple_set)[0]
        plt.xlim(0, max_node)
        plt.ylim(0, max_node)
        plt.tick_params(labelsize=16)
        plt.savefig(self.output_file, dpi=600)
        plt.show()