# imports
import streamlit as st
from rag_pipeline import executar_rag

# page configs
st.set_page_config(
    page_title="🤖 Agente Inteligente de Reviews",
    layout="centered"
)
st.title("🤖 Agente Inteligente de Reviews")
st.write(
    """
    Análise automatizada de avaliações de clientes para apoiar
    decisões operacionais de restaurantes.
    """
)

# inicializa histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# input do usuário
pergunta = st.chat_input(
    "Digite sua pergunta sobre os reviews do restaurante"
)

if pergunta:
    # adiciona pergunta do usuário ao histórico
    st.session_state.messages.append(
        {"role": "user", "content": pergunta}
    )

    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Analisando avaliações..."):
            try:
                insight = executar_rag(pergunta)

                resposta_final = f"### 📊 Insight Gerado\n{insight}\n\n"

                st.markdown(resposta_final)

                # salva resposta do agente no histórico
                st.session_state.messages.append(
                    {"role": "assistant", "content": resposta_final}
                )

            except Exception as e:
                erro = f"Erro ao processar a solicitação: {e}"
                st.error(erro)
                st.session_state.messages.append(
                    {"role": "assistant", "content": erro}
                )