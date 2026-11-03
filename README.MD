CLYVO VET — API de apoio à teleconsulta

## Integrantes

| Nome | RM | Turma |
|---|---:|---|
| Gabriel Ambrósio Saraiva | RM566552 | 2TDSPV |
| João Victor Vendrameto | RM563665 | 2TDSPV |
| Nicolas de Oliveira Jacob | RM564205 | 2TDSPX |
| Vinicius Romaguera Cardozo | RM562308 | 2TDSPX |
| Yuri Fuzinatto Garzoli Barreto | RM561450 | 2TDSPX |

API em Python que recebe uma imagem do pet, classifica a raça com um modelo TensorFlow Lite, consulta os cuidados associados no Oracle e retorna perguntas sugeridas em JSON. Ela foi preparada para ser consumida pelo backend Java da CLYVO VET.

A demonstração visual do modelo está no projeto Streamlit separado.

> A classificação e as perguntas são recursos de apoio à teleconsulta; não substituem a avaliação nem o diagnóstico do médico-veterinário.

## Funcionalidades

- Recebe imagens JPG, JPEG e PNG por requisição HTTP.
- Classifica a raça do pet usando TensorFlow Lite.
- Retorna a raça prevista e a confiança do modelo.
- Consulta no Oracle os cuidados cadastrados para a raça.
- Quando não encontra cadastro individual, consulta cuidados gerais para cães (`Canis`).
- Sugere perguntas para apoiar a anamnese veterinária.
- Protege o endpoint com uma chave opcional enviada no cabeçalho `X-API-Key`.

## Tecnologias

- Python
- FastAPI e Uvicorn
- TensorFlow Lite
- Oracle Database com `oracledb`
- NumPy, Pillow e `python-dotenv`

## Estrutura do repositório

```text
FastAPI_challengeClyvoVet/
├── main.py
├── requirements.txt
├── README.MD
├── .gitignore
└── modelo/
    ├── model_unquant.tflite
    └── labels.txt
```

A aplicação procura o modelo em `modelo/model_unquant.tflite` e os rótulos em `modelo/labels.txt`. Mantenha esses nomes e caminhos iguais aos do repositório.

## Requisitos

- Python 3.13 recomendado para usar TensorFlow 2.21.0 estável.
- Python 3.14 também está previsto no `requirements.txt`, usando TensorFlow 2.22.0rc0 (versão candidata).
- Acesso ao banco Oracle com usuário, senha e endereço de conexão válidos.

## Executar localmente no Windows

Abra o Prompt de Comando na pasta do projeto.

### 1. Criar e ativar um ambiente virtual

Com Python 3.13 instalado:

```bat
py -3.13 -m venv .venv
.venv\Scripts\activate.bat
```

No PowerShell, use a ativação correspondente:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Instalar as dependências

Com o ambiente virtual ativado:

```bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use `python -m pip` no mesmo terminal em que vai iniciar o servidor, para instalar as bibliotecas no interpretador correto.

### 3. Configurar as variáveis de ambiente

Crie um arquivo `.env` na raiz, ao lado de `main.py`, com as variáveis abaixo e os valores fornecidos para seu banco Oracle:

```dotenv
ORACLE_USER=seu_usuario
ORACLE_PASSWORD=sua_senha
ORACLE_DSN=oracle.fiap.com.br:1521/ORCL
API_TOKEN=crie_uma_chave_aleatoria_longa
```

- `ORACLE_USER`, `ORACLE_PASSWORD` e `ORACLE_DSN` são usados para conectar ao Oracle.
- `API_TOKEN` é uma chave de acesso própria para a API; não use a senha do Oracle.
- Se `API_TOKEN` estiver definido, as chamadas para análise precisam enviar o mesmo valor em `X-API-Key`.

O `.gitignore` do repositório ignora `.env`. **Não envie esse arquivo ao GitHub.** Em produção, configure os mesmos valores na área privada de variáveis ou Secrets do serviço de hospedagem.

### 4. Iniciar o servidor

Na raiz do projeto:

```bat
uvicorn main:app --reload --port 8000
```

Mantenha o terminal aberto enquanto usa a API. Para encerrá-la, pressione **Ctrl+C**.

## Testar pelo Swagger

Com o servidor ativo, abra a documentação interativa:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Para enviar uma imagem:

1. Expanda `POST /api/v1/analise`.
2. Clique em **Try it out**.
3. No campo `imagem`, clique em **Choose File** e escolha uma imagem JPG, JPEG ou PNG.
4. Se aparecer `x-api-key`, informe o valor de `API_TOKEN` sem `Bearer` e sem aspas.
5. Clique em **Execute** e consulte a resposta JSON.

O endpoint de disponibilidade pode ser acessado em [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health). A resposta esperada é `{"status":"ok"}`.

## Integração com o backend Java

O Java deve enviar a imagem como `multipart/form-data` com o nome do campo `imagem`.

- Método: `POST`
- Caminho: `/api/v1/analise`
- Cabeçalho opcional, quando `API_TOKEN` está configurado: `X-API-Key: <chave>`

Exemplo para testar no terminal:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/analise" \
  -H "X-API-Key: SUA_CHAVE_DA_API" \
  -F "imagem=@foto-do-pet.jpg"
```

A API retorna os campos `raca_prevista`, `confianca`, `cadastro_consultado`, `correspondencia_exata`, `cuidados` e `perguntas_sugeridas`. O backend Java pode converter esse JSON em seus próprios objetos de resposta.

`127.0.0.1` só funciona na máquina onde o Uvicorn está rodando. Para um backend Java hospedado em outro servidor, publique esta API em um serviço acessível por ele e configure a URL pública no Java.

## Exemplo de resposta

```json
{
  "raca_prevista": "Pastor Alemão",
  "confianca": 0.9877,
  "cadastro_consultado": "Pastor Alemao",
  "correspondencia_exata": true,
  "cuidados": [
    {
      "predisposicao": "Displasia Coxofemoral",
      "orientacao_predisposicao": "Monitorar articulações e evitar excesso de impacto durante crescimento.",
      "restricao": "Pisos Escorregadios",
      "orientacao_restricao": "Evitar superfícies lisas para animais idosos.",
      "gravidade": "LEVE"
    }
  ],
  "perguntas_sugeridas": [
    "Você percebe dificuldade para levantar, mancar, evitar escadas ou sentir dor ao se movimentar?"
  ]
}
```

## Erros comuns

- **`No module named 'tensorflow'`:** instale as dependências com `python -m pip install -r requirements.txt` no ambiente correto.
- **Modelo não encontrado:** confira se `modelo/model_unquant.tflite` está presente.
- **`401 Chave de API inválida`:** o valor enviado no cabeçalho `X-API-Key` precisa ser igual ao `API_TOKEN` carregado pela API. Reinicie o Uvicorn após alterar `.env`.
- **`ORA-01017`:** o Oracle recusou o usuário ou a senha; revise as credenciais do banco.
- **`404` ao abrir `/`:** a API não tem página inicial. Use `/docs` ou `/health`.
- **`405 Method Not Allowed` em `/api/v1/analise`:** a análise aceita `POST`; envie a imagem pelo `/docs` ou pelo backend Java.

## Segurança

Não coloque senhas ou tokens no código, no README ou no GitHub. Se uma credencial real for compartilhada acidentalmente, troque-a antes de disponibilizar a API.
