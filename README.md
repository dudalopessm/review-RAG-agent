# 🤖 Agente Inteligente de Reviews de Restaurantes - RAG

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red?style=flat-square&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.1.137-green?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4.6-orange?style=flat-square)

**Análise Inteligente de Avaliações de Clientes para Restaurantes**

[Visualização do Projeto](#-visualização) • [Instalação](#-instalação) • [Uso](#-como-usar) • [Arquitetura](#-arquitetura)

</div>

---

## 📋 Sobre o Projeto

**Agente Inteligente de Reviews** é uma aplicação que utiliza **Retrieval-Augmented Generation (RAG)** para analisar automaticamente avaliações de clientes de restaurantes (especificamente dados do iFood). A solução combina busca semântica com IA generativa para gerar insights operacionais contextualizados.

### 🎯 Funcionalidades Principais

- ✅ **Busca Semântica Inteligente**: Recupera reviews mais relevantes usando embeddings
- ✅ **Geração de Insights**: Análise contextualizada com LLM (OpenAI GPT)
- ✅ **Interface Interativa**: Chat intuitivo com Streamlit
- ✅ **Detecção de Temas Sensíveis**: Identifica questões críticas de saúde e segurança alimentar

---

### 🔄 Fluxo de Processamento

1. **Ingestão** (`ingestion.py`): Lê CSV de reviews e cria embeddings
2. **Armazenamento**: ChromaDB armazena vetores para busca rápida
3. **Recuperação**: RAG recupera top-K reviews semanticamente similares
4. **Análise**: LLM gera insight contextualizado

---

## 📦 Stack Tecnológico

| Componente | Tecnologia | Versão |
|-----------|-----------|--------|
| **Framework Web** | Streamlit | 1.29.0 |
| **LLM** | OpenAI GPT | via API |
| **Vector DB** | ChromaDB | 0.4.6 |
| **Embeddings** | Sentence Transformers | 2.2.2 |
| **Orquestração LLM** | LangChain | 0.1.137 |
| **Linguagem** | Python | 3.12 |

---

## 🚀 Instalação

### Pré-requisitos

- Python 3.12+
- Conta OpenAI com API key ativa
- pip (gerenciador de pacotes Python)

### Passo 1: Clonar o Repositório

```bash
git clone <seu-repositorio>
cd autoral
```

### Passo 2: Criar Ambiente Virtual

```bash
python3.12 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### Passo 3: Instalar Dependências

```bash
pip install -r requirements.txt
```

### Passo 4: Configurar Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
OPENAI_KEY=sk-sua-chave-aqui
REVIEWS=data/raw/reviews_ptbr.csv
CHROMA_PATH=data/chroma_db_data
```

### Passo 5: Ingerir Reviews

```bash
python ingestion.py
```

Este comando:
- Lê o CSV de reviews
- Cria embeddings usando Sentence Transformers
- Armazena no ChromaDB para busca semântica

---

## 💻 Como Usar

### Iniciar a Aplicação

```bash
streamlit run app.py
```

A aplicação abrirá em `http://localhost:8501`

### Exemplo de Uso

1. **Acesse a interface**: Abra o navegador em `http://localhost:8501`
2. **Digite uma pergunta** sobre os reviews:
   - "Quais são os principais problemas de entrega?"
   - "Como é a qualidade dos alimentos?"
   - "Há reclamações sobre higiene?"
3. **Aguarde a análise**: O sistema recupera reviews relevantes e gera insights contextualizados

---

## 📁 Estrutura do Projeto

```
autoral/
├── app.py                      # Interface Streamlit
├── rag_pipeline.py             # Pipeline RAG
├── ingestion.py                # Ingestão de dados
├── requirements.txt            # Dependências
├── .env                        # Variáveis de ambiente (não versionado)
├── README.md                   # Este arquivo
└── data/
    ├── raw/
    │   └── reviews_ptbr.csv    # CSV original de reviews
    └── chroma_db_data/         # Vector database (gerado automaticamente)
        ├── chroma.sqlite3
        └── [collections]/
```

---

## 🔍 Detalhes Técnicos

### RAG Pipeline (`rag_pipeline.py`)

- **Modelo de Embedding**: `sentence-transformers/all-MiniLM-L6-v2`
- **K (Top Reviews)**: 10 documentos mais similares por padrão
- **Detecção de Temas Sensíveis**: Identifica automaticamente questões críticas de saúde/segurança

**Temas Críticos Monitorados:**
- Intoxicação, vômito, diarreia
- Segurança alimentar, higiene
- Alimentos crus, estragados, contaminados
- Incidentes com hospitais



## 🔧 Configuração Avançada

### Modificar K (Número de Reviews Recuperados)

Em `rag_pipeline.py`, linha de `recuperar_reviews()`:

```python
docs_semanticos = retriever.invoke(pergunta)  # Padrão: k=10
```

### Adicionar Novos Temas Sensíveis

Em `rag_pipeline.py`, atualize a lista `TEMAS_SENSIVEIS`:

```python
TEMAS_SENSIVEIS = [
    "saúde",
    "seu novo tema",  # Adicione aqui
    ...
]
```

### Usar Modelo de Embedding Diferente

Em `rag_pipeline.py`:

```python
embeddings = HuggingFaceEmbeddings(
    model_name="seu-modelo-aqui"
)
```

Opções populares:
- `all-MiniLM-L6-v2` (padrão, rápido)
- `all-mpnet-base-v2` (mais acurado, mais lento)
- `multilingual-e5-large` (multilíngue)

---

## 📊 Exemplo de Saída

```
Pergunta: "Há reclamações sobre qualidade da comida?"

📊 Insight Gerado:
A análise dos reviews aponta problemas consistentes com:
- Alimentos chegando frios ou moles
- Questões de frescor em preparações
- Inconsistência na qualidade entre pedidos
```

---

## 🐛 Troubleshooting

| Problema | Solução |
|----------|---------|
| **OPENAI_KEY não encontrada** | Verifique `.env` e restart do app |
| **ChromaDB error** | Execute `python ingestion.py` novamente |
| **Streamlit não inicia** | Verifique se porta 8501 está livre |
| **Embeddings lentos** | Use modelo mais rápido (all-MiniLM-L6-v2) |

---

## 📝 Variáveis de Ambiente

| Variável | Descrição | Exemplo |
|----------|-----------|---------|
| `OPENAI_KEY` | API key do OpenAI | `sk-...` |
| `REVIEWS` | Caminho do CSV | `data/raw/reviews_ptbr.csv` |
| `CHROMA_PATH` | Diretório ChromaDB | `data/chroma_db_data` |

---

## 📚 Referências

- [LangChain Documentation](https://docs.langchain.com/)
- [ChromaDB](https://docs.trychroma.com/)
- [Streamlit](https://docs.streamlit.io/)
- [Sentence Transformers](https://www.sbert.net/)

---

## 📄 Licença

Este projeto é parte de trabalho autoral acadêmico.

---

<div align="center">

**Desenvolvido com ❤️ para análise inteligente de dados**

</div>
