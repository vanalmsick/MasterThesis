import tensorflow as tf
import z_helpers as my_helpers
from NN_tensorflow_models import compile_and_fit, evaluate_model



def main_run_baseline_models(train_ds, val_ds, test_ds, val_performance_dict, test_performance_dict, data_props, data, pred_length=4, examples=None):

    class Baseline_last_value(tf.keras.Model):
        def __init__(self, label_index=None):
            super().__init__()
            self.label_index = label_index

        def call(self, inputs):
            if self.label_index is None:
                return inputs
            else:
                result = inputs[:, :, self.label_index]
                result = result[:, :, tf.newaxis]
                final_result = result
                for _ in range(pred_length - 1):
                    final_result = tf.concat((final_result, result), axis=2)
                return final_result

        @property
        def name(self):
            return 'baseline_last'


    baseline = Baseline_last_value(label_index=data.latest_out['columns_lookup']['X'][data.dataset_y_col[0]])
    model_name = baseline.name
    history, mlflow_additional_params = compile_and_fit(train=train_ds, val=val_ds, model=baseline, MAX_EPOCHS=1, model_name=model_name)
    mlflow_additional_params['data_props'] = data_props
    val_performance_dict[model_name] = evaluate_model(model=baseline, tf_data=val_ds)
    test_performance_dict[model_name] = evaluate_model(model=baseline, tf_data=test_ds, mlflow_additional_params=mlflow_additional_params)
    my_helpers.mlflow_last_run_add_param(param_dict=mlflow_additional_params)



    class Baseline_4last_value(tf.keras.Model):
        def __init__(self, label_index=None):
            super().__init__()
            self.label_index = label_index

        def call(self, inputs):
            if self.label_index is None:
                return inputs
            else:
                result = inputs[:, -pred_length:, self.label_index]
                result = result[:, tf.newaxis, :]
                return result

        @property
        def name(self):
            return 'baseline_4last'


    baseline4 = Baseline_4last_value(label_index=data.latest_out['columns_lookup']['X'][data.dataset_y_col[0]])
    model_name4 = baseline4.name
    history, mlflow_additional_params = compile_and_fit(train=train_ds, val=val_ds, model=baseline4, MAX_EPOCHS=1, model_name=model_name4)
    mlflow_additional_params['data_props'] = data_props
    val_performance_dict[model_name4] = evaluate_model(model=baseline4, tf_data=val_ds)
    test_performance_dict[model_name4] = evaluate_model(model=baseline4, tf_data=test_ds, mlflow_additional_params=mlflow_additional_params)
    my_helpers.mlflow_last_run_add_param(param_dict=mlflow_additional_params)


    if examples is not None:
        example_pred = {}
        example_pred[model_name] = baseline.predict(examples['X_ds'])
        example_pred[model_name4] = baseline4.predict(examples['X_ds'])

        return example_pred

