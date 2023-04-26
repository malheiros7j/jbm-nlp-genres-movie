# jbm-nlp-genres-movie - NLP Project for movies genres prediction

Estrutura do Projeto:

- Pasta Notebook: contém notebooks com a solução do projeto em ciclos (o Notebook Final é o Ciclo_03_FINAL.ipynb).
- requirements.txt: arquivo com as dependências necessárias para executar o projeto (recomendável usar um ambiente virtual).
- data: contém a base de dados utilizada no projeto -> CMU Movie Summary Corpus (http://www.cs.cmu.edu/~ark/personas/)
- API_Deploy e bot-telegram: contêm apenas os códigos de deploy do modelo em produção e do bot no Telegram. O link para os repositórios completos de deploy podem ser encontrado na seção 10.0 do Notebook Ciclo_03_FINAL.ipynb.

Como fazer uma previsão:
- 1ª forma: abra e execute o notebook Ciclo_03_FINAL.ipynb. Na seção 9.0, há uma função que solicita uma sinopse, o usuário insere o texto e o modelo retorna os respectivos gêneros atribuídos ao filme.
- 2ª forma: execute apenas a seção de importação e funções e faça a previsão na seção 11.0 usando a API do modelo em produção. É necessário apenas inserir a sinopse do filme e o modelo consulta a API e retorna os gêneros.
- 3ª forma: envie uma mensagem com a sinopse para o bot do Telegram através do link (https://t.me/jbm_genrespred_Bot). O bot retornará a previsão do gênero do filme.

*Observação: o Telegram tem um limite máximo de 4096 caracteres para cada mensagem enviada. Se a sinopse de um filme ultrapassar esse limite, ela será dividida em duas ou mais mensagens. Nesse caso, o modelo fará uma nova previsão para cada uma dessas partes da mensagem.*
