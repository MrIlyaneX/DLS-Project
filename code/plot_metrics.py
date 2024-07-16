import matplotlib.pyplot as plt
import numpy as np


def plot_average_metrics_per_method(data):
    """
    Plots average precision@k, recall@k, and F1-score for each method in bar plots.

    Parameters:
    data (list of dict): List of dictionaries containing 'method' and 'metrics'.
                         Each dictionary is of the format:
                         {'method': method (str - one of these strings 'window_train', 'window_validation', 'detection_train', 'detection_validation'),
                          'metrics': {'precision@k': float, 'recall@k': float, 'F1-score': float}}
    """
    methods = set(d['method'] for d in data)
    metrics_per_methods = {method: {'precision@k': [], 'recall@k': [], 'F1-score': []} for method in methods}

    for entry in data:
        method = entry['method']
        metrics = entry['metrics']
        metrics_per_methods[method]['precision@k'].append(metrics['precision@k'])
        metrics_per_methods[method]['recall@k'].append(metrics['recall@k'])
        metrics_per_methods[method]['F1-score'].append(metrics['F1-score'])

    # Calculate average metrics for each class
    avg_metrics_per_method = {method: {'precision@k': np.mean(metrics_per_methods[method]['precision@k']),
                                   'recall@k': np.mean(metrics_per_methods[method]['recall@k']),
                                   'F1-score': np.mean(metrics_per_methods[method]['F1-score'])}
                             for method in methods}

    # Prepare data for plotting
    metrics_names = ['precision@k', 'recall@k', 'F1-score']
    for method in methods:
        values = [avg_metrics_per_method[method][metric] for metric in metrics_names]

        plt.figure(figsize=(8, 6))
        plt.bar(metrics_names, values, color="#4CAF50", width=0.4)
        plt.title(f'Average metrics for method: {method}')
        plt.ylim(0, 1)
        plt.xlabel('Metrics')
        plt.ylabel('Average value')
        plt.show()


# # Example usage:
# data = [
#     {'method': 'window_validation', 'metrics': {'precision@k': 1.0, 'recall@k': 0.83, 'F1-score': 0.91}},
#     {'method': 'window_validation', 'metrics': {'precision@k': 0.9, 'recall@k': 0.85, 'F1-score': 0.87}},
#     {'method': 'window_train', 'metrics': {'precision@k': 0.8, 'recall@k': 0.75, 'F1-score': 0.77}},
#     {'method': 'window_train', 'metrics': {'precision@k': 0.85, 'recall@k': 0.8, 'F1-score': 0.82}},
#     {'method': 'detection_train', 'metrics': {'precision@k': 0.9, 'recall@k': 0.88, 'F1-score': 0.89}},
#     {'method': 'detection_train', 'metrics': {'precision@k': 0.95, 'recall@k': 0.9, 'F1-score': 0.92}},
#     {'method': 'detection_validation', 'metrics': {'precision@k': 1.0, 'recall@k': 0.9, 'F1-score': 0.95}},
#     {'method': 'detection_validation', 'metrics': {'precision@k': 0.98, 'recall@k': 0.92, 'F1-score': 0.95}},
# ]
#
# plot_average_metrics_per_method(data)
