# Detalhes do modelo e do treinamento

## Visão geral

O classificador de imagens da CLYVO VET foi criado no Teachable Machine usando aprendizado supervisionado. Durante o treinamento, cada imagem é associada a um rótulo de raça. A partir desses exemplos, o modelo ajusta seus parâmetros para reconhecer padrões visuais que ajudam a distinguir as classes, como características do focinho, das orelhas, da pelagem e das proporções do corpo.

O modelo treinado é exportado em TensorFlow Lite no formato **Floating point**. A API carrega o arquivo `modelo/model_unquant.tflite` e executa a classificação usando o interpretador `tf.lite.Interpreter`.

## Configuração do treinamento

A configuração registrada no Teachable Machine foi:

| Parâmetro | Valor | O que controla |
|---|---:|---|
| Épocas (epochs) | 1.000 | Quantas passagens completas o treinamento faz pelo conjunto de imagens. |
| Tamanho do lote (batch size) | 16 | Quantas imagens são processadas antes de uma atualização dos parâmetros do modelo. |
| Taxa de aprendizado (learning rate) | 0,001 | O tamanho dos ajustes aplicados aos parâmetros durante cada atualização. |

Esses parâmetros controlam o processo de aprendizagem. O modelo ajusta seus parâmetros a cada lote e repete o processo pelas épocas configuradas.

## Como uma imagem é classificada

1. A API recebe o arquivo enviado e abre a imagem em RGB.
2. A imagem é redimensionada para as dimensões de entrada definidas pelo modelo.
3. Os valores dos pixels são convertidos para números de ponto flutuante e normalizados para a faixa esperada no treinamento, de `-1` a `1`.
4. O interpretador TensorFlow Lite processa a imagem e retorna uma pontuação para cada rótulo do arquivo `labels.txt`.
5. A API seleciona o rótulo com maior pontuação e retorna a raça prevista e o valor correspondente em `confianca`.

A ordem dos rótulos em `labels.txt` deve permanecer sincronizada com a ordem das saídas do modelo. Os arquivos de modelo e rótulos são usados em conjunto.

## Integração com Oracle e perguntas

Depois da classificação, a API compara a raça prevista com os registros de espécie no Oracle. A comparação ignora diferenças de acentuação e formatação. Quando encontra o cadastro correspondente, consulta as predisposições e restrições associadas; caso contrário, usa os cuidados gerais para cães (`Canis`).

As perguntas são selecionadas por regras de palavras-chave aplicadas às predisposições e restrições retornadas pelo banco. Assim, o resultado combina a classificação da imagem, os dados do Oracle e perguntas de apoio à anamnese.

## Como interpretar a confiança

O campo `confianca` é a pontuação atribuída pelo modelo à classe prevista para aquela imagem. Por exemplo, `0.9877` corresponde a aproximadamente `98,77%` na interface. Essa pontuação expressa a confiança do classificador naquela previsão; não é uma medida da acurácia geral do modelo nem uma garantia de identificação correta.

A iluminação, o enquadramento, a nitidez e a semelhança visual entre o pet e as imagens usadas no treinamento podem influenciar o resultado. A classificação apoia a teleconsulta e não substitui a avaliação do médico-veterinário.

## Arquivos relacionados

- `main.py`: carrega o modelo, processa a imagem, consulta o Oracle e monta a resposta da API.
- `modelo/model_unquant.tflite`: modelo exportado pelo Teachable Machine.
- `modelo/labels.txt`: rótulos alinhados às saídas do modelo.
- `requirements.txt`: dependências necessárias para executar a API.
