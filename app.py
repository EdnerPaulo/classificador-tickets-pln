import streamlit as st
import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
from deep_translator import GoogleTranslator
from collections import Counter

# Configuração da página Streamlit
st.set_page_config(page_title="Pipeline PLN - Triagem Inteligente", layout="wide")

# Inicialização e Cache de Recursos de IA/PLN
@st.cache_resource
def iniciar_recursos_pln():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    try:
        nlp = spacy.load("pt_core_news_sm")
    except OSError:
        from spacy.cli import download
        download("pt_core_news_sm")
        nlp = spacy.load("pt_core_news_sm")
    
    # Dicionário explicativo para POS Tags do SpaCy
    pos_traducao = {
        "NOUN": "Substantivo (Nomeia coisas/ideias)", "VERB": "Verbo (Ação/Estado)",
        "ADJ": "Adjetivo (Qualidade/Característica)", "ADV": "Advérbio (Modifica verbo/adj)",
        "PRON": "Pronome (Substitui nome)", "DET": "Determinante (Artigo, etc.)",
        "PROPN": "Nome Próprio (Lugares, Pessoas)", "ADP": "Preposição",
        "CCONJ": "Conjunção Coordenativa", "SCONJ": "Conjunção Subordinativa",
        "NUM": "Numeral", "AUX": "Verbo Auxiliar", "INTJ": "Interjeição"
    }
    return nlp, stopwords.words('portuguese'), SentimentIntensityAnalyzer(), pos_traducao

nlp, stop_words_pt, sia, pos_explicado = iniciar_recursos_pln()

# Interface Gráfica
st.title("🧠 Pipeline Inteligente de PLN — Triagem & Classificação Automatizada")
st.markdown("---")

# Input do Usuário
texto_usuario = st.text_area(
    "Insira a mensagem ou avaliação do cliente em Português:", 
    height=150, 
    placeholder="Exemplo: O aplicativo travou e deu erro na hora de pagar o boleto. Preciso de reembolso ou vou cancelar."
)

if st.button("Executar Pipeline PLN", type="primary"):
    if not texto_usuario.strip():
        st.warning("Por favor, digite um texto para processar o pipeline.")
    else:
        # ==========================================
        # EXECUÇÃO DO PIPELINE EM BACKEND
        # ==========================================
        
        # 1. Texto Original (Capturado)
        texto_original = texto_usuario

        # 2. Normalização
        texto_min = texto_original.lower()
        texto_normalizado = texto_min.translate(str.maketrans('', '', string.punctuation))

        # 3. Tokenização NLTK e Frequência Inicial
        tokens_iniciais = word_tokenize(texto_normalizado, language='portuguese')
        freq_inicial = Counter(tokens_iniciais)

        # 4. Filtragem de Stopwords
        stopwords_encontradas = [w for w in tokens_iniciais if w in stop_words_pt]
        tokens_pos_filtro = [w for w in tokens_iniciais if w not in stop_words_pt]
        texto_limpo_pt = " ".join(tokens_pos_filtro)

        # 5. Mapeamento Léxico Negativo
        palavras_negativas_alvo = ["ruim", "péssimo", "erro", "falha", "problema", "defeito", "travou", "bug"]
        negativas_detectadas = {w: tokens_iniciais.count(w) for w in palavras_negativas_alvo if w in tokens_iniciais}

        # 6. Análise Morfológica (SpaCy)
        doc = nlp(texto_normalizado)
        analise_pos = []
        for token in doc:
            if token.text.strip():
                classe_pt = pos_explicado.get(token.pos_, token.pos_)
                analise_pos.append(f"**{token.text}** → {classe_pt}")

        # 7. Tradução e VADER
        try:
            texto_en = GoogleTranslator(source='pt', target='en').translate(texto_limpo_pt)
            score_vader = sia.polarity_scores(texto_en)
        except:
            score_vader = sia.polarity_scores(texto_limpo_pt) # Fallback se falhar tradução
        
        compound = score_vader['compound']

        # 8. Classificação Híbrida de Sentimento
        tem_negativa_explicita = len(negativas_detectadas) > 0
        if compound <= -0.05 or (tem_negativa_explicita and compound < 0.2):
            sentimento_final = "Negativo"
            justificativa_sentimento = f"Classificado como **Negativo** devido ao Score Compound ({compound:.2f}) e à presença de termos de atrito textuais no léxico em português."
        elif compound >= 0.05:
            sentimento_final = "Positivo"
            justificativa_sentimento = f"Classificado como **Positivo** baseado no Score Compound favorável ({compound:.2f}) balanceado na tradução."
        else:
            sentimento_final = "Neutro"
            justificativa_sentimento = f"Classificado como **Neutro** (Score: {compound:.2f}). O texto apresenta caráter essencialmente informativo ou descritivo."

        # 9. Detecção de Setor (Palavras-chave)
        setores_regras = {
            "Suporte Técnico": ["erro", "falha", "bug", "sistema", "aplicativo", "travou"],
            "Financeiro": ["pagamento", "boleto", "cobrança", "fatura", "reembolso"],
            "Cancelamento": ["cancelar", "cancelamento", "encerrar", "desistir"]
        }
        
        setores_encontrados = {setor: [] for setor in setores_regras}
        score_roteamento = {setor: 0 for setor in setores_regras}

        for token in tokens_iniciais:
            for setor, palavras_chave in setores_regras.items():
                if token in palavras_chave:
                    if token not in setores_encontrados[setor]:
                        setores_encontrados[setor].append(token)
                    score_roteamento[setor] += 1

        # 10. Classificação de Roteamento
        setor_max = max(score_roteamento, key=score_roteamento.get)
        if score_roteamento[setor_max] > 0:
            roteamento_final = setor_max
            justificativa_roteamento = f"Direcionado ao **{setor_max}** por conter maior densidade de termos do departamento ({setores_encontrados[setor_max]})."
        else:
            roteamento_final = "Atendimento Geral"
            justificativa_roteamento = "Nenhum termo técnico dos setores mapeados foi detectado. Encaminhado para a triagem geral humana."

        # 11. Frequência Pós-Filtro (Top 5)
        freq_pos = Counter(tokens_pos_filtro)
        df_freq = pd.DataFrame(freq_pos.items(), columns=['Palavra', 'Frequência'])
        df_freq = df_freq.sort_values(by=['Frequência', 'Palavra'], ascending=[False, True]).reset_index(drop=True)
        top_5 = df_freq.head(5)['Palavra'].tolist()

        # ==========================================
        # EXIBIÇÃO INTERATIVA NA INTERFACE (EXPECTATION)
        # ==========================================
        
        st.header("🔬 Resultados do Processamento do Pipeline")
        
        # 1. Texto original
        st.subheader("1. Texto Original")
        st.info(texto_original)

        # 2. Texto normalizado
        st.subheader("2. Texto Normalizado")
        st.code(texto_normalizado, language=None)

        # 3. Tokenização e Análise Morfológica
        st.subheader("3. Tokenização e Análise Morfológica (SpaCy)")
        st.write("Abaixo, veja a decomposição gramatical de cada termo encontrado:")
        st.write(", ".join(analise_pos))

        # 4. Frequência das palavras
        st.subheader("4. Tabela de Frequência Completa (Ordem Alfabética em Empates)")
        st.dataframe(df_freq, use_container_width=True)

        # 5. Stopwords removidas
        st.subheader("5. Filtragem de Stopwords (NLTK)")
        col_stop1, col_stop2 = st.columns(2)
        with col_stop1:
            st.metric(label="Total de Stopwords Removidas", value=len(stopwords_encontradas))
            st.write("**Lista de Stopwords Identificadas:**", list(set(stopwords_encontradas)))
        with col_stop2:
            st.write("**Texto Limpo Resultante:**")
            st.success(texto_limpo_pt if texto_limpo_pt else "*Nenhuma palavra restante pós-filtro.*")

        # 6. Palavras negativas encontradas
        st.subheader("6. Palavras Negativas Mapeadas")
        if negativas_detectadas:
            for pal, qtd in negativas_detectadas.items():
                st.warning(f"🚨 Termo Crítico Detectado: **{pal}** ({qtd} ocorrência(s))")
        else:
            st.info("Nenhuma palavra negativa mapeada foi detectada.")

        # 7. Classificação de sentimento
        st.subheader("7. Classificação Híbrida de Sentimento")
        if sentimento_final == "Positivo":
            st.success(f"**Resultado:** {sentimento_final}")
        elif sentimento_final == "Negativo":
            st.error(f"**Resultado:** {sentimento_final}")
        else:
            st.warning(f"**Resultado:** {sentimento_final}")
        st.markdown(f"*Justificativa:* {justificativa_sentimento}")

        # 8. Palavras-chave identificadas por Setor
        st.subheader("8. Mapeamento de Palavras-chave por Setor")
        cols_setores = st.columns(3)
        for idx, (setor, termos) in enumerate(setores_encontrados.items()):
            with cols_setores[idx]:
                st.markdown(f"**{setor}**")
                if termos:
                    st.write(termos)
                else:
                    st.caption("Nenhuma palavra-chave detectada para este setor.")

        # 9. Classificação da mensagem (Roteamento)
        st.subheader("9. Roteamento Automatizado da Mensagem")
        st.metric(label="Departamento de Destino", value=roteamento_final)
        st.markdown(f"*Motivo:* {justificativa_roteamento}")

        # 10. Palavras mais frequentes (Top 5)
        st.subheader("10. Top 5 Termos Mais Frequentes (Pós-Filtro)")
        for rank, pal in enumerate(top_5, 1):
            st.markdown(f"{rank}º Lugar: **{pal}**")

        # 11. Conclusão
        st.subheader("11. Conclusão & Insights Acionáveis")
        assunto_principal = " e ".join(top_5[:2]) if len(top_5) >= 2 else (top_5[0] if top_5 else "Assunto não identificado")
        
        conclusao_texto = (
            f"O cliente manifestou um ponto de contato abordando principalmente termos sobre **'{assunto_principal}'**, "
            f"apresentando um sentimento geral avaliado como **{sentimento_final}**. Devido aos gatilhos textuais identificados, "
            f"a solicitação foi roteada automaticamente para a equipe de **{roteamento_final}**. Como ação imediata, recomenda-se "
            f"priorizar o chamado na fila do setor de destino e disparar uma resposta personalizada mitigando os problemas apontados."
        )
        st.write(conclusao_texto)