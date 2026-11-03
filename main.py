from __future__ import annotations

import io
import logging
import os
import re
import secrets
import threading
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import oracledb
import tensorflow as tf
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

PASTA_PROJETO = Path(__file__).resolve().parent
CAMINHO_MODELO = PASTA_PROJETO / "modelo" / "model_unquant.tflite"
CAMINHO_ROTULOS = PASTA_PROJETO / "modelo" / "labels.txt"

load_dotenv(PASTA_PROJETO / ".env")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clyvo_api")

app = FastAPI(
    title="CLYVO VET — API de apoio à teleconsulta",
    description="Classifica uma imagem do pet, consulta cuidados no Oracle e sugere perguntas para a consulta veterinária.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def carregar_interpretador() -> tuple[tf.lite.Interpreter, threading.Lock]:
    """Carrega o modelo TensorFlow Lite uma vez e prepara seu interpretador."""
    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {CAMINHO_MODELO}. Coloque model.tflite na pasta modelo."
        )

    interpretador = tf.lite.Interpreter(model_path=str(CAMINHO_MODELO))
    interpretador.allocate_tensors()
    return interpretador, threading.Lock()


def ler_rotulos() -> list[str]:
    if not CAMINHO_ROTULOS.exists():
        raise FileNotFoundError(
            f"Arquivo de rótulos não encontrado em {CAMINHO_ROTULOS}."
        )

    rotulos = []
    for linha in CAMINHO_ROTULOS.read_text(encoding="utf-8-sig").splitlines():
        linha = linha.strip()
        if linha:
            rotulos.append(re.sub(r"^\d+\s+", "", linha))
    if not rotulos:
        raise RuntimeError("O arquivo labels.txt está vazio.")
    return rotulos


def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(caractere for caractere in texto if not unicodedata.combining(caractere))
    return re.sub(r"[^a-z0-9]", "", texto.casefold())


def classificar_imagem(conteudo: bytes) -> tuple[str, float]:
    try:
        with Image.open(io.BytesIO(conteudo)) as arquivo_imagem:
            imagem = arquivo_imagem.convert("RGB")
    except (UnidentifiedImageError, OSError) as erro:
        raise HTTPException(status_code=400, detail="O arquivo enviado não é uma imagem válida.") from erro

    interpretador, cadeado = carregar_interpretador()
    rotulos = ler_rotulos()
    detalhe_entrada = interpretador.get_input_details()[0]
    detalhe_saida = interpretador.get_output_details()[0]
    forma = detalhe_entrada["shape"]
    largura, altura = int(forma[2]), int(forma[1])

    imagem = imagem.resize((largura, altura))
    pixels = np.asarray(imagem, dtype=np.float32)
    entrada = np.expand_dims((pixels / 127.5) - 1.0, axis=0)

    tipo_entrada = detalhe_entrada["dtype"]
    escala_entrada, zero_entrada = detalhe_entrada.get("quantization", (0.0, 0))
    if np.issubdtype(tipo_entrada, np.integer):
        if escala_entrada == 0:
            raise RuntimeError("O modelo usa entrada quantizada sem escala válida.")
        entrada = np.round(entrada / escala_entrada + zero_entrada).astype(tipo_entrada)
    else:
        entrada = entrada.astype(tipo_entrada)

    with cadeado:
        interpretador.set_tensor(detalhe_entrada["index"], entrada)
        interpretador.invoke()
        pontuacoes = interpretador.get_tensor(detalhe_saida["index"])[0]

    tipo_saida = detalhe_saida["dtype"]
    escala_saida, zero_saida = detalhe_saida.get("quantization", (0.0, 0))
    if np.issubdtype(tipo_saida, np.integer) and escala_saida:
        pontuacoes = (pontuacoes.astype(np.float32) - zero_saida) * escala_saida

    if len(pontuacoes) != len(rotulos):
        raise RuntimeError(
            f"O modelo retornou {len(pontuacoes)} classes, mas labels.txt contém {len(rotulos)} rótulos."
        )

    indice = int(np.argmax(pontuacoes))
    return rotulos[indice], float(pontuacoes[indice])


def obter_conexao_oracle():
    usuario = os.getenv("ORACLE_USER")
    senha = os.getenv("ORACLE_PASSWORD")
    endereco = os.getenv("ORACLE_DSN")
    if not usuario or not senha or not endereco:
        raise RuntimeError(
            "Configure ORACLE_USER, ORACLE_PASSWORD e ORACLE_DSN nas variáveis de ambiente."
        )
    return oracledb.connect(user=usuario, password=senha, dsn=endereco)


def buscar_cuidados_especie(raca_prevista: str) -> tuple[str, list[dict[str, Any]], bool]:
    conexao = obter_conexao_oracle()
    try:
        with conexao.cursor() as cursor:
            cursor.execute(
                """
                SELECT id_especie, nome_especie
                FROM TB_CT_ESPECIE
                WHERE genero = 'Canis'
                """
            )
            especies = cursor.fetchall()
            especie_encontrada = next(
                (registro for registro in especies if normalizar(registro[1]) == normalizar(raca_prevista)),
                None,
            )
            correspondencia_exata = especie_encontrada is not None

            consulta_cuidados = """
                SELECT DISTINCT
                    p.nome_predisposicao,
                    p.orientacao_predisposicao,
                    r.nome_restricao,
                    r.orientacao_restricao,
                    r.gravidade_restricao
                FROM TB_CT_CUIDADO c
                JOIN TB_CT_PREDISPOSICAO_ESP p
                  ON p.id_predisposicao = c.id_predisposicao
                JOIN TB_CT_RESTRICAO_ESP r
                  ON r.id_restricao = c.id_restricao
                WHERE c.id_especie = :id_especie
                ORDER BY p.nome_predisposicao, r.gravidade_restricao DESC
            """

            if correspondencia_exata:
                cursor.execute(consulta_cuidados, id_especie=especie_encontrada[0])
                nome_consultado = especie_encontrada[1]
            else:
                cursor.execute(
                    """
                    SELECT DISTINCT
                        p.nome_predisposicao,
                        p.orientacao_predisposicao,
                        r.nome_restricao,
                        r.orientacao_restricao,
                        r.gravidade_restricao
                    FROM TB_CT_ESPECIE e
                    JOIN TB_CT_CUIDADO c ON c.id_especie = e.id_especie
                    JOIN TB_CT_PREDISPOSICAO_ESP p
                      ON p.id_predisposicao = c.id_predisposicao
                    JOIN TB_CT_RESTRICAO_ESP r
                      ON r.id_restricao = c.id_restricao
                    WHERE e.genero = 'Canis'
                    ORDER BY p.nome_predisposicao, r.gravidade_restricao DESC
                    """
                )
                nome_consultado = "Canis (cuidados gerais para cães)"

            cuidados = [
                {
                    "predisposicao": registro[0],
                    "orientacao_predisposicao": registro[1],
                    "restricao": registro[2],
                    "orientacao_restricao": registro[3],
                    "gravidade": registro[4],
                }
                for registro in cursor.fetchall()
            ]
            return nome_consultado, cuidados, correspondencia_exata
    finally:
        conexao.close()


def sugerir_pergunta(condicao: str) -> str:
    termo = normalizar(condicao)
    regras = [
        (("respiratorio", "respiracao"), "O pet apresenta tosse, cansaço fácil, respiração ruidosa ou dificuldade para respirar? Em que situações isso acontece?"),
        (("displasia", "coxofemoral", "articulacao"), "Você percebe dificuldade para levantar, mancar, evitar escadas ou sentir dor ao se movimentar?"),
        (("dermatite", "pele", "dobras", "otite"), "Notou coceira, vermelhidão, secreção, mau cheiro ou irritação na pele ou nas orelhas? Quando começou?"),
        (("obesidade", "gordura"), "Houve mudança recente no peso ou no apetite? Qual alimento e quantidade ele recebe por dia?"),
        (("ansiedade", "estresse", "comportamento"), "Como o pet reage quando fica sozinho ou quando há mudanças na rotina? Houve alteração de comportamento?"),
        (("calor", "temperatura"), "O pet demonstra desconforto no calor, procura locais mais frescos ou apresenta ofegação excessiva?"),
        (("renal", "hidratacao", "desidratacao"), "Você notou mudanças na ingestão de água, na frequência urinária ou na aparência da urina?"),
        (("cardiaco", "cardiaca"), "Ele apresenta cansaço fora do habitual, tosse, desmaios ou menor disposição para passeios?"),
        (("dentario", "dente", "oral"), "Percebe mau hálito, dificuldade para mastigar, salivação ou resistência ao toque na boca?"),
        (("digestiva", "gastrica", "alimentar", "pelos"), "Houve vômito, alteração nas fezes, desconforto após comer ou mudança no apetite?"),
    ]
    for palavras_chave, pergunta in regras:
        if any(normalizar(palavra) in termo for palavra in palavras_chave):
            return pergunta
    return f"Você observou algum sinal relacionado a {condicao.lower()}? Quando começou e com que frequência ocorre?"


def montar_perguntas(cuidados: list[dict[str, Any]]) -> list[str]:
    perguntas = []
    for cuidado in cuidados:
        condicao = f"{cuidado['predisposicao']} {cuidado['restricao']}"
        perguntas.append(sugerir_pergunta(condicao))
    return list(dict.fromkeys(perguntas))


def validar_chave_api(x_api_key: str | None = Header(default=None)) -> None:
    chave_configurada = os.getenv("API_TOKEN")
    if chave_configurada and not secrets.compare_digest(x_api_key or "", chave_configurada):
        raise HTTPException(status_code=401, detail="Chave de API inválida.")


@app.get("/health")
def verificar_api():
    return {"status": "ok"}


@app.post("/api/v1/analise", dependencies=[Depends(validar_chave_api)])
def analisar_pet(imagem: UploadFile = File(...)):
    try:
        raca, confianca = classificar_imagem(imagem.file.read())
        nome_consultado, cuidados, correspondencia_exata = buscar_cuidados_especie(raca)
        return {
            "raca_prevista": raca,
            "confianca": confianca,
            "cadastro_consultado": nome_consultado,
            "correspondencia_exata": correspondencia_exata,
            "cuidados": cuidados,
            "perguntas_sugeridas": montar_perguntas(cuidados),
        }
    except HTTPException:
        raise
    except Exception as erro:
        logger.exception("Falha ao analisar o pet")
        raise HTTPException(
            status_code=500,
            detail="Não foi possível analisar a imagem. Confira os logs da API.",
        ) from erro