from metaflow import FlowSpec, step, conda_base, project, get_namespace, conda


@project(name='super_project')
@conda_base(python='3.10.11', libraries={'scikit-learn': '1.5.1'})
class ClassifierTrainFlow(FlowSpec):

    @conda(libraries={"matplotlib": "3.9.1"})
    @step
    def start(self):
        from io import BytesIO
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        from models import ModelKNN, ModelSVM
        import matplotlib.pyplot as plt

        print('NAMESPACE IS', get_namespace()) 

        self.models = [ModelKNN(), ModelSVM()]

        X, y = datasets.load_iris(return_X_y=True)
        self.train_data, self.test_data, self.train_labels, self.test_labels = \
            train_test_split(X, y, test_size=0.2, random_state=0)
        
        plt.scatter(self.train_data[:, 0], self.train_data[:, 1])
        fig = plt.gcf()
        buf = BytesIO()
        fig.savefig(buf)
        self.vis = buf.getvalue()
        
        self.next(self.train, foreach="models")

    @step
    def train(self):
        self.model = self.input
        self.model.fit(self.train_data, self.train_labels)

        self.next(self.eval)

    @step
    def eval(self):
        self.score = self.model.eval(self.test_data, self.test_labels)

        self.next(self.select_best_model)

    @step
    def select_best_model(self, inputs):
        self.results = sorted(
            [(inp.model, inp.model.name, inp.score) for inp in inputs], 
            key=lambda x: -x[-1],
        )
        self.model = self.results[0][0]
        self.next(self.end)

    @step
    def end(self):
        print('Scores:')
        print('\n'.join('%s %f' % res[1:] for res in self.results))

if __name__ == '__main__':
    ClassifierTrainFlow()