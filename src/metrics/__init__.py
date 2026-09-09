from src.metrics.based_saliency_metric import BasedSaliency_KL, BasedSaliency_CC, BasedSaliency_SIM

METRIC_REGISTRY = {
    "BasedSaliency_KL": BasedSaliency_KL,
    "BasedSaliency_CC": BasedSaliency_CC,
    "BasedSaliency_SIM": BasedSaliency_SIM,
}


def create_custom_metric(metric_name,**kwargs):
    metric_cls = METRIC_REGISTRY[metric_name]
    metric = metric_cls(**kwargs)
    return metric
