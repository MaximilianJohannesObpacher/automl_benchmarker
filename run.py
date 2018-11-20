#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import autosklearn.classification
import numpy
import os
import pandas
import sys
import timeit

target_name='ken'

def main():
    # Reading the data
    training_data = pandas.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                 'automl_benchmarker/data/numerai_training_data.csv'), header=0)

    tournament_data = pandas.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                   'automl_benchmarker/data/numerai_tournament_data.csv'), header=0)

    features = [f for f in list(training_data) if 'feature' in f]
    x = training_data[features]
    y = training_data['target_' + target_name]
    x_tournament = tournament_data[features]
    ids = tournament_data['id']

    start = timeit.timeit()
    print(start)
    # Settings including crossvalidation
    model = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=7200,
        per_run_time_limit=120,
        tmp_folder='automl_benchmarker/data/autosklearn_cv_'+target_name+'_tmp',
        output_folder='automl_benchmarker/data/autosklearn_cv_'+target_name+'_out',
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds':5}
    )
    end = timeit.timeit()
    # training
    model.fit(x, y)
    print(end - start)
    print(model.show_models())

    #prediction
    eps = sys.float_info.epsilon
    y_prediction = model.predict_proba(x_tournament)
    results = numpy.clip(y_prediction[:, 1], 0.0 + eps, 1.0 - eps)
    results_df = pandas.DataFrame(data={'probability_'+target_name: results})
    joined = pandas.DataFrame(ids).join(results_df)
    joined.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'automl_benchmarker/data/prediction_'+ target_name +'.csv'), index=False, float_format='%.16f')


if __name__ == '__main__':
    main()