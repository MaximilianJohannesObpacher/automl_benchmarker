#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import autosklearn.classification
import numpy
import os
import pandas
import sys
import timeit

target_name='ken'

def ingest():
    training_data = pandas.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                 'automl_benchmarker/data/numerai_training_data.csv'), header=0)

    tournament_data = pandas.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                'automl_benchmarker/data/numerai_tournament_data.csv'), header=0)

    features = [f for f in list(training_data) if 'feature' in f]
    x = training_data[features]
    y = training_data['target_'+target_name]
    x_tournament = tournament_data[features]
    ids = tournament_data['id']
    return (x, y, x_tournament, ids)


def train(x, y):
    start = timeit.timeit()
    model = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=36000,
        per_run_time_limit=600)
    mid = timeit.timeit()
    print(mid-start)
    model.fit(x, y)
    end = timeit.timeit()
    print(end-mid)
    print(model.show_models())
    return model


def predict(model, x_tournament, ids):
    eps = sys.float_info.epsilon
    y_prediction = model.predict_proba(x_tournament)
    results = numpy.clip(y_prediction[:, 1], 0.0 + eps, 1.0 - eps)
    results_df = pandas.DataFrame(data={'probability_'+target_name: results})
    joined = pandas.DataFrame(ids).join(results_df)
    joined.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'automl_benchmarker/data/prediction_'+ target_name +'.csv'), index=False, float_format='%.16f')


def main():
    x, y, x_tournament, ids = ingest()
    model = train(x, y)
    predict(model, x_tournament.copy(), ids)


if __name__ == '__main__':
    main()