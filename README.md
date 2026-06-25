# 🧠 Pipeline Inteligente de PLN — Triagem & Classificação Automatizada

Este projeto consiste em uma aplicação web interativa desenvolvida com **Streamlit** que implementa um pipeline modular de **Processamento de Linguagem Natural (PLN)**. O sistema foi projetado para atuar no ecossistema de Customer Experience (CX), convertendo textos brutos de reclamações ou avaliações de clientes em insights estruturados e roteamento automático de demandas.

---

## 🎯 Funcionalidades do Pipeline

O motor lógico executa rigorosamente as seguintes atividades sequenciais ao receber um texto em português:

1. **Normalização:** Limpeza inicial convertendo o texto para minúsculas e eliminando pontuações.
2. **Tokenização:** Segmentação do texto limpo em palavras isoladas (tokens) via `nltk.word_tokenize`.
3. **Filtragem de Stopwords:** Identificação e remoção de termos irrelevantes (artigos, preposições) utilizando a base de dados do **NLTK**.
4. **Mapeamento Léxico Crítico:** Monitoramento e contagem de palavras de atrito explícitas (ex: *erro, falha, bug, travou*).
5. **Análise Morfológica (POS Tagging):** Mapeamento gramatical de cada termo utilizando os modelos avançados do **SpaCy** (`pt_core_news_sm`).
6. **Análise Híbrida de Sentimento:** Tradução automatizada do texto para o inglês (via `deep-translator`) para processamento de alta precisão do Score Compound utilizando o **NLTK VADER**, cruzando o score com o léxico nativo em português.
7. **Roteamento Inteligente:** Direcionamento estatístico do chamado para um dos departamentos mapeados (*Suporte Técnico, Financeiro, Cancelamento* ou *Atendimento Geral*).

---

## 🛠️ Tecnologias Utilizadas

* **[Streamlit](https://streamlit.io/):** Interface gráfica e dashboard interativo.
* **[NLTK](https://www.nltk.org/):** Tokenização, Stopwords e Análise de Sentimento (VADER).
* **[SpaCy](https://spacy.io/):** Etiquetagem Morfossintática (POS Tagging) em Português.
* **[Deep Translator](https://github.com/nidhaloff/deep-translator):** Motor de tradução integrado.
* **[Pandas](https://pandas.pydata.org/):** Manipulação, estruturação e ordenação de tabelas de frequência.

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
Certifique-se de ter o **Python 3.8 ou superior** instalado.

### 1. Clonar o Repositório
```bash
git clone [https://github.com/SEU-USUARIO/NOME-DO-REPOSITORIO.git](https://github.com/SEU-USUARIO/NOME-DO-REPOSITORIO.git)
cd NOME-DO-REPOSITORIO
