import torchtuples as tt
from pycox.models import LogisticHazard, PMF, DeepHitSingle, PCHazard, MTLR, CoxPH
from sksurv.ensemble import RandomSurvivalForest


model_dict = {
    "LogisticHazard": LogisticHazard,
    "PMF": PMF,
    "DeepHitSingle": DeepHitSingle,
    "PCHazard": PCHazard,
    "MTLR": MTLR,
    "DeepSurv": CoxPH,
    "RSF": RandomSurvivalForest
}


def get_model(model_name, in_features, out_features=None, labtrans=None):
    net = tt.practical.MLPVanilla(in_features, [32, 32], out_features or 1, batch_norm=True, dropout=0.1)
    model_cls = model_dict[model_name]

    if model_name == "DeepSurv":
        model = model_cls(net, tt.optim.Adam(0.01))
    elif model_name == "RSF":
        model = model_cls()
    else:
        model = model_cls(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)

    return model
