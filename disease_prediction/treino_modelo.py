import logging
import os
import gc
from datetime import datetime, timedelta
import pandas as pd
import joblib
import oracledb
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, fbeta_score
from airflow.decorators import dag, task
from airflow.hooks.base import BaseHook

# ==============================================================================
# CONFIGURAÇÕES GLOBAIS DE CALIBRAÇÃO E PARAMETRIZAÇÃO
# ==============================================================================
THRESHOLD_DECISAO = 0.3  # Limiar de probabilidade focado em alto Recall
NUM_FOLDS_CV = 3  # Número de quebras para a validação cruzada
LIMITE_AMOSTRAGEM = (
    15000000  # Gatilho de RAM: ativa amostragem se a base superar este valor
)
AMOSTRA_POR_CLASSE = (
    10000000  # Quantidade de registros por classe (target 0 e 1) se amostrado
)

# Hiperparâmetros dos Modelos
N_ESTIMATORS = 100
MAX_DEPTH_RF = 15
MAX_ITER_HIST = 100
RANDOM_STATE = 42
# ==============================================================================

logger = logging.getLogger(__name__)


@dag(
    schedule="@monthly",
    start_date=datetime(2026, 6, 1),
    catchup=False,
    tags=["assistencial", "machine_learning", "treinamento"],
    default_args={"retries": 1, "retry_delay": timedelta(minutes=10)},
)
def pipeline_retreino_cardiaco():

    @task
    def extrair_historico_rotulado():
        logger.info("Iniciando extração do histórico de atendimentos rotulados...")

        connection = BaseHook.get_connection("oracle_conn")
        dsn = f"{connection.host}:{connection.port}/{connection.schema}"

        try:
            logger.info("Tentando conectar ao banco Oracle...")
            oracledb.init_oracle_client()
            conn = oracledb.connect(
                user=connection.login,
                password=connection.password,
                dsn=dsn,
            )
            logger.info("Conectado com sucesso ao Oracle!")
        except oracledb.DatabaseError as e:
            logger.error(f"Erro ao conectar no banco de dados: {e}")
            raise e

        cursor = conn.cursor()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sql_path = os.path.join(base_dir, "query", "ML_RISCO_CARDIACO_HISTORICO.sql")

        with open(sql_path) as f:
            sql = f.read()

        logger.info("Buscando registros no banco de dados...")
        cursor.execute(sql)
        columns = [col[0].lower() for col in cursor.description]
        df = pd.DataFrame(cursor.fetchall(), columns=columns)

        cursor.close()
        conn.close()

        logger.info(f"Dados históricos extraídos: {len(df)} registros.")

        # Proteção de memória RAM baseada na configuração global
        if len(df) > LIMITE_AMOSTRAGEM:
            logger.info(
                f"Base superior a {LIMITE_AMOSTRAGEM}. Aplicando amostragem estratificada..."
            )
            df = df.groupby("target", group_keys=False).apply(
                lambda x: x.sample(
                    min(len(x), AMOSTRA_POR_CLASSE), random_state=RANDOM_STATE
                )
            )
            logger.info(
                f"Base reduzida para {len(df)} registros para preservação de memória."
            )

        data_dir = os.path.join(os.path.dirname(base_dir), "data")
        os.makedirs(data_dir, exist_ok=True)

        file_path = os.path.join(data_dir, "historico_treino.parquet")

        logger.info("Persistindo DataFrame em formato Parquet...")
        df.to_parquet(file_path, index=False, engine="pyarrow")

        del df
        gc.collect()
        return file_path

    @task
    def treinar_e_validar_modelo(file_path: str):
        logger.info(f"Carregando dados de {file_path}...")
        df = pd.read_parquet(file_path, engine="pyarrow")

        if df.empty or "target" not in df.columns:
            raise ValueError("Base histórica vazia ou sem coluna target.")

        features = [
            "age",
            "sex",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ]

        # 1. CORREÇÃO DE TIPOS (Evita o erro do XGBoost)
        logger.info("Convertendo todas as features para formato numérico...")
        for col in features:
            # Converte para número e se houver algum lixo textual, vira NaN, que depois é preenchido com 0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Garante que o target seja inteiro
        df["target"] = df["target"].astype(int)

        X = df[features]
        y = df["target"]

        neg_pos_ratio = (len(y) - sum(y)) / sum(y) if sum(y) > 0 else 1

        # 2. CORREÇÃO DO MULTIPROCESSING (n_jobs=1 evita conflito com o Airflow)
        dict_modelos = {
            "RandomForest": RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                class_weight="balanced",
                max_depth=MAX_DEPTH_RF,
                n_jobs=1,  # Alterado
                random_state=RANDOM_STATE,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=N_ESTIMATORS,
                scale_pos_weight=neg_pos_ratio,
                eval_metric="logloss",
                n_jobs=1,  # Alterado
                random_state=RANDOM_STATE,
            ),
            "HistGradientBoosting": HistGradientBoostingClassifier(
                max_iter=MAX_ITER_HIST,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        }

        skf = StratifiedKFold(
            n_splits=NUM_FOLDS_CV, shuffle=True, random_state=RANDOM_STATE
        )
        resultados = {nome: {"auc": [], "f2": []} for nome in dict_modelos.keys()}

        logger.info(
            f"Iniciando Validação Cruzada Estratificada ({NUM_FOLDS_CV} Folds)..."
        )
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Processando Fold {fold + 1}...")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            for nome, model in dict_modelos.items():
                if nome == "XGBoost" and sum(y_train) > 0:
                    model.scale_pos_weight = (len(y_train) - sum(y_train)) / sum(
                        y_train
                    )

                model.fit(X_train, y_train)

                preds_proba = model.predict_proba(X_val)[:, 1]
                preds_bin = (preds_proba >= THRESHOLD_DECISAO).astype(int)

                auc = roc_auc_score(y_val, preds_proba)
                f2 = fbeta_score(y_val, preds_bin, beta=2, zero_division=0)

                resultados[nome]["auc"].append(auc)
                resultados[nome]["f2"].append(f2)

            gc.collect()

        melhor_modelo_nome = None
        maior_f2_medio = -1

        for nome in dict_modelos.keys():
            auc_med = np.mean(resultados[nome]["auc"])
            f2_med = np.mean(resultados[nome]["f2"])
            logger.info(
                f"[{nome}] Média CV -> AUC-ROC: {auc_med:.4f} | F2-Score (Limiar {THRESHOLD_DECISAO}): {f2_med:.4f}"
            )

            if f2_med > maior_f2_medio:
                maior_f2_medio = f2_med
                melhor_modelo_nome = nome

        logger.info(
            f"Modelo selecionado para implantação: {melhor_modelo_nome} (F2: {maior_f2_medio:.4f})"
        )

        modelo_campeao = dict_modelos[melhor_modelo_nome]
        if melhor_modelo_nome == "XGBoost":
            modelo_campeao.scale_pos_weight = neg_pos_ratio

        logger.info(
            f"Treinando modelo final ({melhor_modelo_nome}) com a base completa..."
        )
        modelo_campeao.fit(X, y)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(os.path.dirname(base_dir), "models")
        os.makedirs(models_dir, exist_ok=True)

        target_path = os.path.join(models_dir, "modelo_risco_cardiaco.pkl")
        temp_path = os.path.join(models_dir, "modelo_risco_cardiaco_temp.pkl")

        joblib.dump(modelo_campeao, temp_path)
        os.replace(temp_path, target_path)
        logger.info(
            f"Novo modelo binário ({melhor_modelo_nome}) persistido em: {target_path}"
        )

    dados_treino = extrair_historico_rotulado()
    treinar_e_validar_modelo(dados_treino)


dag_instancia = pipeline_retreino_cardiaco()
