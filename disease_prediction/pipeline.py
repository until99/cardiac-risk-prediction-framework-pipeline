import logging
import os
from datetime import datetime, timedelta
import pandas as pd
import joblib
import oracledb
from airflow.decorators import dag, task
from airflow.hooks.base import BaseHook

logger = logging.getLogger(__name__)
base_dir = os.path.dirname(os.path.abspath(__file__))
sql_filename = "ML_RISCO_CARDIACO_DIARIO"

sql_path = os.path.join(base_dir, "query", f"{sql_filename}.sql")


@dag(
    schedule="0 3 * * *",
    start_date=datetime(2026, 6, 1),
    catchup=False,
    tags=["assistencial", "machine_learning"],
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
)
def pipeline_risco_cardiaco_diario():

    @task
    def extrair_e_processar_dados(**kwargs):
        logger.info("Iniciando a extração de dados do Tasy...")

        conf = kwargs.get("dag_run").conf or {}
        if "dt_inicio" in conf and "dt_fim" in conf:
            dt_inicio = datetime.strptime(conf["dt_inicio"], "%Y-%m-%d")
            dt_fim = datetime.strptime(conf["dt_fim"], "%Y-%m-%d")
            logger.info(
                f"Parâmetros manuais recebidos: {dt_inicio.date()} a {dt_fim.date()}"
            )
        else:
            hoje = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            dt_inicio = hoje - timedelta(days=1)
            dt_fim = hoje
            logger.info(
                f"Parâmetros automáticos (D-1): {dt_inicio.date()} a {dt_fim.date()}"
            )

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

        try:
            with open(sql_path) as f:
                sql = f.read()

            logger.info("Executando a query de extração...")
            cursor.execute(sql, dt_inicio=dt_inicio, dt_fim=dt_fim)

            columns = [col[0].lower() for col in cursor.description]
            df = pd.DataFrame(cursor.fetchall(), columns=columns)
            logger.info(f"Extração concluída. Total de registros: {len(df)}")

        except Exception as e:
            logger.error(f"Erro durante a extração dos dados: {e}")
            raise e
        finally:
            cursor.close()
            conn.close()

        df.fillna(0, inplace=True)
        return df

    @task
    def prever_risco(df: pd.DataFrame, **kwargs):
        logger.info("Iniciando a etapa de predição em lote...")

        if df.empty:
            logger.warning(
                "Nenhum atendimento retornado para o período. Encerrando tarefa."
            )
            return

        modelo_path = os.path.join(
            os.path.dirname(base_dir), "models", "modelo_risco_cardiaco.pkl"
        )
        ds = kwargs.get("ds")

        try:
            logger.info(f"Carregando o modelo a partir de: {modelo_path}")
            modelo = joblib.load(modelo_path)
        except Exception as e:
            logger.error(f"Falha ao carregar o modelo: {e}")
            raise e

        features_model = [
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

        logger.info("Aplicando predições...")
        df["probabilidade_alto_risco"] = modelo.predict_proba(df[features_model])[:, 1]

        # Threshold de 0.3
        df["flag_risco"] = df["probabilidade_alto_risco"].apply(
            lambda x: 1 if x > 0.3 else 0
        )

        df_alto_risco = df[df["flag_risco"] == 1]
        logger.info(
            f"Predição concluída. {len(df_alto_risco)} pacientes classificados como alto risco."
        )

        output_path = os.path.join(base_dir, "data", f"risco_cardiaco_{ds}.csv")
        try:
            df_alto_risco.to_csv(output_path, index=False)
            logger.info(f"Arquivo CSV exportado com sucesso para: {output_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar o arquivo CSV: {e}")
            raise e

    dados_extraidos = extrair_e_processar_dados()
    prever_risco(dados_extraidos)


dag_instancia = pipeline_risco_cardiaco_diario()
