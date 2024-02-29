from support.models_base import ModelEnvironment


def train(model_environment: ModelEnvironment):
    model_environment.fit()
    model_environment.score()
    print(model_environment.get_score_info())