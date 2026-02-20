from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def get_lgb_model():
    return LGBMClassifier(
        n_estimators=1600,
        learning_rate=0.02,
        max_depth=8,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1
    )

def get_xgb_model():
    return XGBClassifier(
        n_estimators=972,
        learning_rate=0.08233334476657686,
        max_depth=3,
        subsample=0.6967792979720865,
        colsample_bytree=0.7773146292728021,
        reg_alpha=1.911349598671315,
        reg_lambda=0.6194119678307304,
        tree_method="hist",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

def get_cat_model():
    return CatBoostClassifier(
        iterations=2443,
        learning_rate=0.028617286398439353,
        depth=6,
        l2_leaf_reg=3.5313325975665264,
        bagging_temperature=0.5274409717782269,
        random_strength=0.03843459373261249,
        task_type="GPU",
        thread_count=-1,
        random_seed=42,
        verbose=0
    )
