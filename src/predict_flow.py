from metaflow import FlowSpec, step, Flow, Parameter, JSONType, conda_base, project, get_namespace


@project(name='super_project_2')
@conda_base(python='3.10.11', libraries={'scikit-learn': '1.5.1'})
class ClassifierPredictFlow(FlowSpec):
    vector = Parameter('vector', type=JSONType, required=True)

    @step
    def start(self):
        print('NAMESPACE IS', get_namespace()) 

        run = Flow('ClassifierTrainFlow').latest_run
        self.train_run_id = run.pathspec

        print("ClassifierTrainFlow run_id:", self.train_run_id)

        self.model = run['end'].task.data.model
        print("Input vector", self.vector)
        
        self.next(self.end)

    @step
    def end(self):
        print("Predicted class", self.model.predict([self.vector])[0])

if __name__ == '__main__':
    ClassifierPredictFlow()